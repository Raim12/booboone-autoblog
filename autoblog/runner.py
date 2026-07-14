"""Orchestrates a single run: for each category, generate + publish one article."""

from __future__ import annotations

import json
import logging

from .config import Config
from .content import ContentGenerator
from .image import FeaturedImageProvider
from .tracker import KeywordTracker
from .wordpress import WordPressClient

logger = logging.getLogger("autoblog")


def run(
    cfg: Config,
    *,
    dry_run: bool = False,
    only_category: str | None = None,
    limit: int | None = None,
) -> dict:
    cfg.require_valid()

    tracker = KeywordTracker(cfg.keywords_file, cfg.state_file)

    # Map keyword-doc category names -> the client's existing WordPress category
    # IDs (preferred: duplicate-proof). Values may also be a name string.
    category_map: dict[str, object] = {}
    if cfg.category_map_file.exists():
        raw_map = json.loads(cfg.category_map_file.read_text(encoding="utf-8"))
        category_map = {k: v for k, v in raw_map.items() if not k.startswith("_")}
    content_gen = ContentGenerator(
        cfg.deepseek_api_key, cfg.deepseek_model, cfg.deepseek_base_url
    )
    image_provider = (
        FeaturedImageProvider(
            source=cfg.image_source,
            gemini_api_key=cfg.gemini_api_key,
            image_model=cfg.image_model,
            output_dir=cfg.image_dir,
            openai_api_key=cfg.openai_api_key,
            openai_image_model=cfg.openai_image_model,
            openai_image_size=cfg.openai_image_size,
            openai_image_quality=cfg.openai_image_quality,
            atlas_api_key=cfg.atlas_api_key,
            atlas_image_model=cfg.atlas_image_model,
            atlas_image_size=cfg.atlas_image_size,
            atlas_prompt_extend=cfg.atlas_prompt_extend,
        )
        if cfg.generate_images and not dry_run
        else None
    )

    wp = None
    if not dry_run:
        wp = WordPressClient(cfg.wp_url, cfg.wp_user, cfg.wp_app_password)
        user = wp.test_connection()
        logger.info("Connected to WordPress as '%s'.", user.get("name", cfg.wp_user))
        # Validate mapped category IDs up front so we fail loud, not silently.
        for name, target in category_map.items():
            if isinstance(target, int) and not wp.category_exists(target):
                logger.warning(
                    "category_map: '%s' -> id %s does not exist on the site!",
                    name,
                    target,
                )

    categories = tracker.categories()
    if only_category:
        categories = [c for c in categories if c.lower() == only_category.lower()]
        if not categories:
            logger.error("Category '%s' not found in keywords.json.", only_category)
            return {"published": 0, "skipped": 0, "failed": 0}

    # Determine per-run cap.
    cap = limit if limit is not None else cfg.max_posts_per_run
    published = skipped = failed = 0

    logger.info(
        "Starting run: %d categories, %d keywords remaining overall.",
        len(categories),
        tracker.total_remaining(),
    )

    for category in categories:
        if cap is not None and published >= cap:
            logger.info("Reached per-run cap of %d posts.", cap)
            break

        keyword = tracker.next_unused(category)
        if not keyword:
            logger.info("[%s] No unused keywords left — skipping.", category)
            skipped += 1
            continue

        try:
            logger.info("[%s] Generating article for '%s'.", category, keyword)
            article = content_gen.generate(keyword)

            if dry_run:
                logger.info(
                    "[DRY-RUN] Would publish: '%s' (%d chars) in '%s'.",
                    article.title,
                    len(article.content_html),
                    category,
                )
                published += 1
                continue

            assert wp is not None
            media_id = None
            if image_provider is not None:
                image_path = image_provider.get(
                    ai_prompt=article.image_prompt, keyword=keyword, slug=keyword
                )
                if image_path is not None:
                    # An image problem must never lose the article — publish anyway.
                    try:
                        media_id = wp.upload_media(image_path, alt_text=article.title)
                    except Exception as img_err:  # noqa: BLE001
                        logger.warning(
                            "[%s] Featured image upload failed (%s); "
                            "publishing without it.",
                            category,
                            img_err,
                        )

            # Resolve category: an ID (duplicate-proof) or a name to match/create.
            mapped = category_map.get(category, category)
            if isinstance(mapped, int):
                category_id = mapped
            else:
                category_id = wp.get_or_create_category(str(mapped))
            post = wp.create_post(
                title=article.title,
                content_html=article.content_html,
                category_id=category_id,
                status=cfg.post_status,
                featured_media=media_id,
                excerpt=article.meta_description,
                author_id=cfg.post_author_id,
            )
            tracker.mark_used(
                category, keyword, post_id=post["id"], url=post.get("link", "")
            )
            published += 1
        except Exception as err:  # noqa: BLE001 - one failure shouldn't stop the run
            logger.exception("[%s] Failed on '%s': %s", category, keyword, err)
            failed += 1

    logger.info(
        "Run complete. Published=%d Skipped=%d Failed=%d Remaining=%d",
        published,
        skipped,
        failed,
        tracker.total_remaining(),
    )
    return {"published": published, "skipped": skipped, "failed": failed}
