"""Central configuration.

All settings are read from environment variables (loaded from a local ``.env``
file if present). This is the single place the client edits to reconfigure the
bot — API keys, WordPress details, and behaviour — without touching any code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (the folder that contains this package).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass
class Config:
    # WordPress
    wp_url: str = ""
    wp_user: str = ""
    wp_app_password: str = ""

    # DeepSeek (text)
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"

    # Gemini / Imagen (images)
    gemini_api_key: str = ""
    image_model: str = "gemini-2.5-flash-image"
    generate_images: bool = True
    # openai | free_ai | auto | ai (gemini) | stock | none
    image_source: str = "free_ai"

    # OpenAI images (cheapest option: gpt-image-1-mini @ low quality)
    openai_api_key: str = ""
    openai_image_model: str = "gpt-image-1-mini"
    openai_image_size: str = "1536x1024"
    openai_image_quality: str = "low"

    # AtlasCloud images (Z-Image Turbo)
    atlas_api_key: str = ""
    atlas_image_model: str = "z-image/turbo"
    atlas_image_size: str = "1536*1024"
    atlas_prompt_extend: bool = False  # True doubles the cost ($0.03/img)

    # Behaviour
    post_status: str = "publish"  # "publish" or "draft"
    max_posts_per_run: int | None = None
    post_author_id: int | None = None

    # Paths
    keywords_file: Path = field(default_factory=lambda: PROJECT_ROOT / "keywords.json")
    category_map_file: Path = field(default_factory=lambda: PROJECT_ROOT / "category_map.json")
    state_file: Path = field(default_factory=lambda: PROJECT_ROOT / "keywords.state.json")
    image_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "generated_images")
    log_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls(
            wp_url=(os.getenv("WP_URL") or "").rstrip("/"),
            wp_user=os.getenv("WP_USER") or "",
            wp_app_password=os.getenv("WP_APP_PASSWORD") or "",
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or "",
            deepseek_model=os.getenv("DEEPSEEK_MODEL") or "deepseek-chat",
            gemini_api_key=os.getenv("GEMINI_API_KEY") or "",
            image_model=os.getenv("IMAGE_MODEL") or "gemini-2.5-flash-image",
            generate_images=_bool(os.getenv("GENERATE_IMAGES"), default=True),
            image_source=(os.getenv("IMAGE_SOURCE") or "free_ai").strip().lower(),
            openai_api_key=os.getenv("OPENAI_API_KEY") or "",
            openai_image_model=os.getenv("OPENAI_IMAGE_MODEL") or "gpt-image-1-mini",
            openai_image_size=os.getenv("OPENAI_IMAGE_SIZE") or "1536x1024",
            openai_image_quality=(os.getenv("OPENAI_IMAGE_QUALITY") or "low").lower(),
            atlas_api_key=os.getenv("ATLAS_API_KEY") or "",
            atlas_image_model=os.getenv("ATLAS_IMAGE_MODEL") or "z-image/turbo",
            atlas_image_size=os.getenv("ATLAS_IMAGE_SIZE") or "1536*1024",
            atlas_prompt_extend=_bool(os.getenv("ATLAS_PROMPT_EXTEND"), default=False),
            post_status=(os.getenv("POST_STATUS") or "publish").strip().lower(),
            max_posts_per_run=_int(os.getenv("MAX_POSTS_PER_RUN")),
            post_author_id=_int(os.getenv("POST_AUTHOR_ID")),
        )
        return cfg

    def validate(self, *, require_images: bool | None = None) -> list[str]:
        """Return a list of human-readable problems (empty means all good)."""
        problems: list[str] = []
        if not self.wp_url:
            problems.append("WP_URL is not set.")
        elif not self.wp_url.startswith(("http://", "https://")):
            problems.append("WP_URL must start with http:// or https://.")
        if not self.wp_user:
            problems.append("WP_USER is not set.")
        if not self.wp_app_password:
            problems.append("WP_APP_PASSWORD is not set.")
        if not self.deepseek_api_key:
            problems.append("DEEPSEEK_API_KEY is not set.")

        wants_images = self.generate_images if require_images is None else require_images
        if wants_images and self.image_source == "ai" and not self.gemini_api_key:
            problems.append(
                "IMAGE_SOURCE=ai but GEMINI_API_KEY is not set "
                "(use IMAGE_SOURCE=stock for free images without a key)."
            )
        if self.image_source not in {
            "openai",
            "atlas",
            "free_ai",
            "auto",
            "ai",
            "stock",
            "none",
        }:
            problems.append(
                "IMAGE_SOURCE must be one of: openai, atlas, free_ai, auto, ai, "
                "stock, none."
            )
        if wants_images and self.image_source == "openai" and not self.openai_api_key:
            problems.append("IMAGE_SOURCE=openai but OPENAI_API_KEY is not set.")
        if wants_images and self.image_source == "atlas" and not self.atlas_api_key:
            problems.append("IMAGE_SOURCE=atlas but ATLAS_API_KEY is not set.")

        if self.post_status not in {"publish", "draft"}:
            problems.append("POST_STATUS must be 'publish' or 'draft'.")
        return problems

    def require_valid(self) -> None:
        problems = self.validate()
        if problems:
            raise ConfigError(
                "Configuration is incomplete:\n  - " + "\n  - ".join(problems)
            )
