"""WordPress REST API client (Application Password / Basic Auth).

Handles category resolution (auto-creating categories that don't exist yet),
media upload, and post creation.
"""

from __future__ import annotations

import base64
import html
import json
import logging
import mimetypes
from pathlib import Path

import requests

logger = logging.getLogger("autoblog")


def _extract_json_object(text: str) -> dict | None:
    """Pull the first top-level JSON object out of a response that may have
    non-JSON junk (e.g. PHP warnings) printed before it."""
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except ValueError:
        return None


class WordPressClient:
    def __init__(self, base_url: str, username: str, app_password: str):
        self.api = f"{base_url.rstrip('/')}/wp-json/wp/v2"
        token = base64.b64encode(f"{username}:{app_password}".encode()).decode()
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Basic {token}"})
        self._category_cache: dict[str, int] = {}

    # -- diagnostics --------------------------------------------------------
    def test_connection(self) -> dict:
        """Verify credentials; returns the authenticated user's data."""
        resp = self.session.get(f"{self.api}/users/me", timeout=30)
        resp.raise_for_status()
        return resp.json()

    # -- related posts (for internal linking) -------------------------------
    def find_related_posts(
        self, keyword: str, category_id: int | None = None, *, limit: int = 6
    ) -> list[dict]:
        """Return the most relevant existing posts for a keyword, as
        [{"title": ..., "url": ...}]. WordPress does the search server-side
        (its DB is indexed), so this is one fast query, not a scan of all posts.

        Prefers same-category matches; if too few are found, broadens to a
        site-wide search so most articles still get enough posts to link to.
        """
        results: list[dict] = []
        seen: set[str] = set()

        def _search(cat: int | None) -> list[dict]:
            params = {
                "search": keyword,
                "per_page": limit,
                "status": "publish",
                "orderby": "relevance",
                "_fields": "id,title,link",
            }
            if cat:
                params["categories"] = cat
            try:
                resp = self.session.get(f"{self.api}/posts", params=params, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as err:
                logger.warning("Related-post search failed for '%s': %s", keyword, err)
                return []

        def _collect(posts) -> None:
            for post in posts:
                url = post.get("link", "")
                title = html.unescape(post.get("title", {}).get("rendered", "")).strip()
                if title and url and url not in seen:
                    seen.add(url)
                    results.append({"title": title, "url": url})

        # 1) same-category matches first (most relevant)
        if category_id:
            _collect(_search(category_id))
        # 2) if too few, broaden site-wide to top up to `limit`
        if len(results) < 3:
            _collect(_search(None))
        return results[:limit]

    # -- categories ---------------------------------------------------------
    def category_exists(self, category_id: int) -> bool:
        resp = self.session.get(f"{self.api}/categories/{category_id}", timeout=30)
        return resp.status_code == 200

    def get_or_create_category(self, name: str) -> int:
        if name in self._category_cache:
            return self._category_cache[name]

        target = name.strip().lower()

        # Look for an exact (case-insensitive) match first. Note: the REST API
        # returns names HTML-encoded (e.g. "Business &amp; Finance"), so we
        # unescape before comparing.
        resp = self.session.get(
            f"{self.api}/categories",
            params={"search": name, "per_page": 100},
            timeout=30,
        )
        resp.raise_for_status()
        for cat in resp.json():
            existing = html.unescape(cat.get("name", "")).strip().lower()
            if existing == target:
                self._category_cache[name] = cat["id"]
                return cat["id"]

        # Not found -> create it.
        create = self.session.post(
            f"{self.api}/categories", json={"name": name}, timeout=30
        )
        # If it already exists (race, or an encoding mismatch above), WordPress
        # returns 400 "term_exists" with the existing id — use that.
        if create.status_code == 400:
            data = create.json()
            if data.get("code") == "term_exists":
                existing_id = data.get("data", {}).get("term_id")
                if existing_id:
                    self._category_cache[name] = existing_id
                    return existing_id
        create.raise_for_status()
        cat_id = create.json()["id"]
        logger.info("Created WordPress category '%s' (id=%s)", name, cat_id)
        self._category_cache[name] = cat_id
        return cat_id

    # -- media --------------------------------------------------------------
    def upload_media(self, image_path: Path, *, alt_text: str = "") -> int:
        mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
        payload = image_path.read_bytes()
        headers = {
            "Content-Disposition": f'attachment; filename="{image_path.name}"',
            "Content-Type": mime,
        }

        last_err: Exception | None = None
        for attempt in range(1, 3):
            resp = self.session.post(
                f"{self.api}/media", headers=headers, data=payload, timeout=180
            )
            resp.raise_for_status()
            try:
                media = resp.json()
                break
            except ValueError as err:
                # Some hosts print PHP warnings before the JSON (display_errors on
                # in production), which garbles the body. The upload usually still
                # succeeded — recover the JSON object embedded in the response.
                media = _extract_json_object(resp.text)
                if media is not None:
                    logger.warning(
                        "Media response had leading junk (host PHP warnings); "
                        "recovered the JSON."
                    )
                    break
                last_err = err
                logger.warning(
                    "Media upload returned an unparseable response (attempt %d). "
                    "Retrying.",
                    attempt,
                )
        else:
            raise RuntimeError(
                f"WordPress media upload did not return JSON: {last_err}"
            )
        media_id = media["id"]
        if alt_text:
            # Best-effort: set alt text for SEO/accessibility.
            try:
                self.session.post(
                    f"{self.api}/media/{media_id}",
                    json={"alt_text": alt_text},
                    timeout=30,
                )
            except requests.RequestException:
                pass
        logger.info("Uploaded featured image (media id=%s)", media_id)
        return media_id

    # -- posts --------------------------------------------------------------
    def create_post(
        self,
        *,
        title: str,
        content_html: str,
        category_id: int,
        status: str = "publish",
        featured_media: int | None = None,
        excerpt: str = "",
        author_id: int | None = None,
    ) -> dict:
        payload: dict = {
            "title": title,
            "content": content_html,
            "status": status,
            "categories": [category_id],
        }
        if featured_media:
            payload["featured_media"] = featured_media
        if excerpt:
            payload["excerpt"] = excerpt
        if author_id:
            payload["author"] = author_id

        resp = self.session.post(f"{self.api}/posts", json=payload, timeout=60)
        resp.raise_for_status()
        post = resp.json()
        logger.info(
            "Published post id=%s status=%s -> %s",
            post.get("id"),
            post.get("status"),
            post.get("link", ""),
        )
        return post
