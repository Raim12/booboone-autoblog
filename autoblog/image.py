"""Featured image resolution.

Strategy (configurable via IMAGE_SOURCE):
    free_ai -> free AI image (Pollinations) -> stock fallback -> none  [recommended]
    auto    -> paid Gemini/Imagen AI -> free AI -> stock -> none
    ai      -> paid AI only (Gemini/Imagen; needs billing)
    stock   -> free stock photos only (Openverse)
    none    -> no featured image

"free_ai" gives unique AI-generated images with no API key and no billing.
"ai"/"auto" use the Gemini/Imagen API, which needs a Google account with image
quota (i.e. billing enabled). Nothing here ever raises — a failure just means
the post publishes without an image.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .atlascloud import AtlasImageGenerator
from .imaging import normalize_image
from .openai_image import OpenAIImageGenerator
from .pollinations import PollinationsImageGenerator
from .stock_image import StockImageProvider

logger = logging.getLogger("autoblog")


class AIImageGenerator:
    """Generates an image via Google Gemini image models or Imagen."""

    def __init__(self, api_key: str, model: str, output_dir: Path):
        self.api_key = api_key
        self.model = model
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str, *, slug: str) -> Path | None:
        try:
            if self.model.startswith("imagen"):
                return self._generate_imagen(prompt, slug=slug)
            return self._generate_gemini(prompt, slug=slug)
        except Exception as err:  # noqa: BLE001 - never block publishing
            logger.warning("AI image generation failed (%s).", _short(err))
            return None

    def _generate_gemini(self, prompt: str, *, slug: str) -> Path | None:
        from google.genai import types

        client = self._get_client()
        resp = client.models.generate_content(
            model=self.model,
            contents=(
                f"Create a photorealistic, editorial featured image. {prompt}. "
                "No text, captions, logos, or watermarks."
            ),
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )
        for cand in resp.candidates or []:
            for part in cand.content.parts or []:
                inline = getattr(part, "inline_data", None)
                if inline and inline.data:
                    ext = ".png" if "png" in (inline.mime_type or "") else ".jpg"
                    path = self.output_dir / f"{_safe_slug(slug)}{ext}"
                    path.write_bytes(inline.data)
                    logger.info("Generated AI image (%s) -> %s", self.model, path.name)
                    return path
        logger.warning("AI model returned no image for '%s'.", slug)
        return None

    def _generate_imagen(self, prompt: str, *, slug: str) -> Path | None:
        from google.genai import types

        client = self._get_client()
        result = client.models.generate_images(
            model=self.model,
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio="16:9"),
        )
        images = getattr(result, "generated_images", None) or []
        if not images:
            logger.warning("Imagen returned no image for '%s'.", slug)
            return None
        path = self.output_dir / f"{_safe_slug(slug)}.png"
        path.write_bytes(images[0].image.image_bytes)
        logger.info("Generated AI image (%s) -> %s", self.model, path.name)
        return path


class FeaturedImageProvider:
    """Orchestrates AI + stock fallback according to the configured strategy."""

    def __init__(
        self,
        *,
        source: str,
        gemini_api_key: str,
        image_model: str,
        output_dir: Path,
        openai_api_key: str = "",
        openai_image_model: str = "gpt-image-1-mini",
        openai_image_size: str = "1536x1024",
        openai_image_quality: str = "low",
        atlas_api_key: str = "",
        atlas_image_model: str = "z-image/turbo",
        atlas_image_size: str = "1536*1024",
        atlas_prompt_extend: bool = False,
    ):
        self.source = (source or "free_ai").lower()
        self.output_dir = output_dir
        self._openai = (
            OpenAIImageGenerator(
                openai_api_key,
                openai_image_model,
                output_dir,
                size=openai_image_size,
                quality=openai_image_quality,
            )
            if openai_api_key and self.source == "openai"
            else None
        )
        self._atlas = (
            AtlasImageGenerator(
                atlas_api_key,
                atlas_image_model,
                output_dir,
                size=atlas_image_size,
                prompt_extend=atlas_prompt_extend,
            )
            if atlas_api_key and self.source == "atlas"
            else None
        )
        self._paid_ai = (
            AIImageGenerator(gemini_api_key, image_model, output_dir)
            if gemini_api_key and self.source in {"auto", "ai"}
            else None
        )
        self._free_ai = (
            PollinationsImageGenerator(output_dir)
            if self.source in {"free_ai", "auto", "openai", "atlas"}
            else None
        )
        self._stock = (
            StockImageProvider(output_dir)
            if self.source in {"free_ai", "auto", "stock", "openai", "atlas"}
            else None
        )

    def get(self, *, ai_prompt: str, keyword: str, slug: str) -> Path | None:
        path = self._resolve(ai_prompt=ai_prompt, keyword=keyword, slug=slug)
        # Clean + optimize before it goes to WordPress.
        return normalize_image(path) if path is not None else None

    def _resolve(self, *, ai_prompt: str, keyword: str, slug: str) -> Path | None:
        if self.source == "none":
            return None
        # 0a) AtlasCloud (Z-Image Turbo)
        if self._atlas is not None:
            path = self._atlas.generate(ai_prompt, slug=slug)
            if path is not None:
                return path
        # 0b) OpenAI images
        if self._openai is not None:
            path = self._openai.generate(ai_prompt, slug=slug)
            if path is not None:
                return path
        # 1) paid Gemini/Imagen (only when explicitly enabled + key present)
        if self._paid_ai is not None:
            path = self._paid_ai.generate(ai_prompt, slug=slug)
            if path is not None:
                return path
            if self.source == "ai":
                return None
        # 2) free AI image (Pollinations)
        if self._free_ai is not None:
            path = self._free_ai.generate(ai_prompt, slug=slug)
            if path is not None:
                return path
        # 3) free stock photo
        if self._stock is not None:
            return self._stock.fetch(keyword, slug=slug)
        return None


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "featured"


def _short(err: Exception) -> str:
    return repr(err)[:160]
