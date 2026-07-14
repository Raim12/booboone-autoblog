"""Image generation via AtlasCloud (Z-Image Turbo).

Docs: POST https://api.atlascloud.ai/api/v1/model/generateImage
Pricing (as published): $0.015/image, or $0.03 with prompt_extend enabled.
"""

from __future__ import annotations

import base64
import logging
import re
import time
from pathlib import Path

import requests

logger = logging.getLogger("autoblog")

# AtlasCloud's docs are inconsistent about the path, so we try the documented
# native endpoint first, then the OpenAI-compatible one. Only a *successful*
# generation is billed, so failed attempts here cost nothing.
_ENDPOINTS = [
    "https://api.atlascloud.ai/api/v1/model/generateImage",
    "https://api.atlascloud.ai/v1/images/generations",
]
_PREDICTION = "https://api.atlascloud.ai/api/v1/model/prediction/{id}"


class AtlasImageGenerator:
    def __init__(
        self,
        api_key: str,
        model: str = "z-image/turbo",
        output_dir: Path | None = None,
        *,
        size: str = "1536*1024",  # landscape for featured images
        prompt_extend: bool = False,  # True doubles the price ($0.03/img)
    ):
        self.api_key = api_key
        self.model = model
        self.size = size
        self.prompt_extend = prompt_extend
        self.output_dir = output_dir or Path("generated_images")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, prompt: str, *, slug: str, timeout: int = 180) -> Path | None:
        full_prompt = (
            f"{prompt}. Photorealistic, editorial blog featured image, high quality, "
            "natural lighting. Absolutely no text anywhere in the image: no words, "
            "letters, numbers, captions, logos, watermarks, signage, or labels. "
            "Do not show readable screens, documents, papers, or charts."
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        native_body = {
            "model": self.model,
            "prompt": full_prompt,
            "prompt_extend": self.prompt_extend,
            "size": self.size,
            "seed": -1,
            "enable_base64_output": False,
            "enable_sync_mode": True,  # wait for the image instead of polling
        }
        openai_body = {
            "model": self.model,
            "prompt": full_prompt,
            "size": self.size.replace("*", "x"),
            "n": 1,
        }

        data = None
        for url in _ENDPOINTS:
            body = openai_body if "/v1/images/" in url else native_body
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            except requests.RequestException as err:
                logger.warning("AtlasCloud %s -> %s", url, repr(err)[:90])
                continue
            if resp.status_code == 200:
                data = resp.json()
                break
            logger.warning(
                "AtlasCloud %s -> %s %s", url, resp.status_code, resp.text[:120]
            )

        if data is None:
            logger.warning("AtlasCloud: no endpoint accepted the request.")
            return None

        try:
            image_ref = _extract_output(data)

            # If it came back still processing, poll the prediction endpoint.
            if image_ref is None:
                inner = data.get("data") if isinstance(data.get("data"), dict) else data
                pred_id = inner.get("id") or inner.get("prediction_id")
                if not pred_id:
                    logger.warning(
                        "AtlasCloud: no output and no prediction id: %s", str(data)[:200]
                    )
                    return None
                image_ref = self._poll(pred_id, headers, timeout=timeout)
                if image_ref is None:
                    return None

            image_bytes = self._to_bytes(image_ref)
            if not image_bytes:
                return None

            path = self.output_dir / f"{_safe_slug(slug)}.png"
            path.write_bytes(image_bytes)
            logger.info("Generated AI image (%s) -> %s", self.model, path.name)
            return path
        except Exception as err:  # noqa: BLE001 - never block publishing
            logger.warning("AtlasCloud image generation failed (%s).", repr(err)[:180])
            return None

    def _poll(self, pred_id: str, headers: dict, *, timeout: int) -> str | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(2)
            r = requests.get(_PREDICTION.format(id=pred_id), headers=headers, timeout=60)
            if r.status_code != 200:
                continue
            data = r.json()
            status = (data.get("status") or "").lower()
            out = _extract_output(data)
            if out is not None:
                return out
            if status in {"failed", "error", "canceled"}:
                logger.warning("AtlasCloud prediction %s: %s", status, str(data)[:200])
                return None
        logger.warning("AtlasCloud timed out waiting for the image.")
        return None

    def _to_bytes(self, ref: str) -> bytes | None:
        if ref.startswith("http"):
            r = requests.get(ref, timeout=120)
            r.raise_for_status()
            return r.content
        try:
            return base64.b64decode(ref)
        except Exception:  # noqa: BLE001
            logger.warning("AtlasCloud: could not decode image payload.")
            return None


def _extract_output(data: dict) -> str | None:
    """Pull the first image URL / base64 string out of the response.

    Handles both AtlasCloud's native shape ({"outputs": [...]}) and the
    OpenAI-compatible shape ({"data": [{"url"|"b64_json": ...}]}).
    """
    # AtlasCloud wraps the real payload: {"code":200,"data":{"outputs":[url]}}
    inner = data.get("data")
    if isinstance(inner, dict):
        data = inner

    outputs = (
        data.get("outputs")
        or data.get("output")
        or data.get("images")
        or data.get("data")
    )
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return (
                first.get("url")
                or first.get("image")
                or first.get("b64_json")
                or first.get("image_url")
            )
    if isinstance(outputs, str):
        return outputs
    return None


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "featured"
