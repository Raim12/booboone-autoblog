"""Article generation via DeepSeek (OpenAI-compatible API).

Returns a structured article: title, SEO meta description, HTML body, and an
image prompt used later for the featured image.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

from openai import OpenAI

logger = logging.getLogger("autoblog")

# Headings that scream "AI wrote this". We drop the heading tag but keep the text
# beneath it, so the article simply ends on its final paragraph.
_CONCLUSION_HEADINGS = re.compile(
    r"<h([1-6])[^>]*>\s*(?:in\s+)?(?:conclusion|final\s+thoughts?|final\s+word|"
    r"summary|in\s+summary|to\s+sum\s+up|wrapping\s+up|to\s+wrap\s+up|"
    r"closing\s+thoughts?|the\s+bottom\s+line|last\s+thoughts?)\s*[:.!]?\s*</h\1>",
    re.IGNORECASE,
)


def strip_conclusion_headings(html: str) -> str:
    """Remove 'Conclusion'-style headings (the paragraph text is kept)."""
    cleaned, count = _CONCLUSION_HEADINGS.subn("", html)
    if count:
        logger.info("Removed %d conclusion-style heading(s) from the article.", count)
    return cleaned.strip()

SYSTEM_PROMPT = (
    "You are an expert SEO copywriter and content strategist. You write original, "
    "accurate, genuinely helpful long-form blog articles that read naturally and "
    "avoid fluff or keyword stuffing."
)

USER_PROMPT_TEMPLATE = """Write a comprehensive, engaging, SEO-optimised blog post targeting the keyword: "{keyword}".

Requirements:
- 900-1400 words.
- Natural, human tone. Write like an experienced human writer, not an AI.
- Use the keyword and closely related terms naturally (no stuffing).
- Well structured with an intro, multiple <h2> sections, <h3> sub-points where useful, short paragraphs, and at least one <ul> bullet list.

CRITICAL - avoid these AI giveaways:
- Do NOT include a section headed "Conclusion", "In Conclusion", "Final Thoughts",
  "Summary", "Wrapping Up", "Closing Thoughts", or anything similar. The article must
  simply END on a substantive, useful section - no summarising sign-off heading.
- Do NOT open with clichés like "In today's fast-paced world", "In the ever-evolving
  landscape of", "Whether you're a beginner or", or "Let's dive in".
- Do NOT use phrases like "In conclusion", "To sum up", "At the end of the day",
  "It's important to note that", or "When it comes to".
- Do NOT over-use em dashes or begin sentences with "Moreover", "Furthermore",
  "Additionally".
- Vary sentence length. Be specific and concrete: use real examples, numbers, and
  practical detail rather than generic advice.

{internal_links}Return ONLY a JSON object with exactly these fields:
{{
  "title": "An engaging, click-worthy H1 title (do NOT include the word 'title')",
  "meta_description": "A 150-160 character SEO meta description",
  "content_html": "The full article body as clean HTML using <h2>, <h3>, <p>, <ul>, <li>, <strong>. Do NOT include <html>, <head>, <body>, or an <h1> tag.",
  "image_prompt": "A vivid, concrete prompt (max 40 words) for a photorealistic editorial featured image that fits the article. IMPORTANT: describe a scene with NO readable text in it - avoid computer/phone screens, documents, papers, charts, books, signage, labels, or whiteboards, because AI renders text as gibberish. Prefer people, hands, objects, places, nature, or close-up details instead."
}}"""

# Injected into the prompt only when related posts are available.
INTERNAL_LINKS_TEMPLATE = """INTERNAL LINKS (important for SEO - do this carefully):
Below are real, existing articles on this same website. Naturally weave {min_links}-{max_links}
of them into the article body as contextual internal links. Rules:
- Place each link inside a relevant sentence using descriptive anchor text (the
  linked words should describe the target, not "click here" or the raw URL).
- Use proper HTML: <a href="EXACT_URL">natural anchor text</a>, with the EXACT URLs given.
- Only link where it genuinely fits the topic. Do NOT force all of them, do NOT add a
  "Related posts" list, and do NOT mention that these are internal links.
- Spread them through the body, not all in one place.

Existing articles you may link to:
{links}

"""


def _build_links_block(related_posts, *, min_links: int = 3, max_links: int = 5) -> str:
    if not related_posts:
        return ""
    lines = "\n".join(
        f'- "{p["title"]}" -> {p["url"]}' for p in related_posts if p.get("url")
    )
    if not lines:
        return ""
    return INTERNAL_LINKS_TEMPLATE.format(
        min_links=min_links, max_links=max_links, links=lines
    )


@dataclass
class Article:
    keyword: str
    title: str
    meta_description: str
    content_html: str
    image_prompt: str


class ContentGenerator:
    def __init__(self, api_key: str, model: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        keyword: str,
        *,
        related_posts: list[dict] | None = None,
        max_retries: int = 3,
    ) -> Article:
        prompt = USER_PROMPT_TEMPLATE.format(
            keyword=keyword,
            internal_links=_build_links_block(related_posts or []),
        )
        last_err: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.8,
                    stream=False,
                )
                raw = response.choices[0].message.content or ""
                data = json.loads(raw)
                article = Article(
                    keyword=keyword,
                    title=str(data["title"]).strip(),
                    meta_description=str(data.get("meta_description", "")).strip(),
                    content_html=strip_conclusion_headings(
                        str(data["content_html"]).strip()
                    ),
                    image_prompt=str(data.get("image_prompt", keyword)).strip(),
                )
                if not article.title or not article.content_html:
                    raise ValueError("DeepSeek returned an empty title or body.")
                return article
            except Exception as err:  # noqa: BLE001 - retry any transient failure
                last_err = err
                wait = 2 ** attempt
                logger.warning(
                    "DeepSeek generation failed (attempt %d/%d) for '%s': %s. "
                    "Retrying in %ds.",
                    attempt,
                    max_retries,
                    keyword,
                    err,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"DeepSeek failed to generate an article for '{keyword}': {last_err}"
        )
