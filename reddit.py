"""Reddit: posts and comments that mention the product.

Works without an account through Reddit's public JSON endpoints. For more
reliable access (Reddit rate-limits anonymous traffic hard), create a free
"script" app at https://www.reddit.com/prefs/apps and set REDDIT_CLIENT_ID and
REDDIT_CLIENT_SECRET; the collector then uses the official OAuth API.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone

import requests

from .. import http
from ..models import Review

NAME = "Reddit"
_token_cache: dict = {}


def _base_and_headers() -> tuple[str, dict]:
    cid, secret = os.getenv("REDDIT_CLIENT_ID"), os.getenv("REDDIT_CLIENT_SECRET")
    if not (cid and secret):
        return "https://www.reddit.com", {}
    if "token" not in _token_cache:
        resp = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(cid, secret), data={"grant_type": "client_credentials"},
            headers={"User-Agent": http.USER_AGENT}, timeout=15,
        )
        resp.raise_for_status()
        _token_cache["token"] = resp.json()["access_token"]
    return "https://oauth.reddit.com", {"Authorization": f"bearer {_token_cache['token']}"}


def _iso(ts) -> str:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).date().isoformat()
    except (TypeError, ValueError):
        return ""


def _post_review(d: dict) -> Review | None:
    body = (d.get("selftext") or "").strip()
    if body in ("[deleted]", "[removed]"):
        body = ""
    text = f"{d.get('title', '')}. {body}".strip()
    if len(text) < 40:
        return None
    return Review(source=NAME, text=text, title=d.get("title", ""),
                  url="https://www.reddit.com" + d.get("permalink", ""),
                  author=d.get("author", ""), date=_iso(d.get("created_utc")),
                  context=f"r/{d.get('subreddit', '')}", upvotes=d.get("score"))


def _flatten_comments(children: list, post: dict, out: list, depth: int = 0) -> None:
    for c in children:
        if c.get("kind") != "t1":
            continue
        d = c.get("data", {})
        body = d.get("body", "")
        if body and body not in ("[deleted]", "[removed]") and len(body) >= 40:
            out.append(Review(
                source=NAME, text=body, title=post.get("title", ""),
                url="https://www.reddit.com" + d.get("permalink", post.get("permalink", "")),
                author=d.get("author", ""), date=_iso(d.get("created_utc")),
                context=f"r/{post.get('subreddit', '')} · {post.get('title', '')[:80]}",
                upvotes=d.get("score"),
            ))
        replies = d.get("replies")
        if depth < 2 and isinstance(replies, dict):
            _flatten_comments(replies.get("data", {}).get("children", []), post, out, depth + 1)


def _subreddit_guess(query: str) -> str:
    return re.sub(r"[^a-z0-9]", "", query.lower())


def collect(query: str, limit: int = 150, **_) -> list[Review]:
    base, headers = _base_and_headers()
    posts: dict[str, dict] = {}

    # 1. site-wide search for the exact phrase
    q = f'"{query}"' if " " in query.strip() else query
    for sort in ("relevance", "top"):
        data = http.get_json(f"{base}/search.json", headers=headers,
                             params={"q": q, "sort": sort, "t": "year", "limit": 50, "type": "link"})
        for child in data.get("data", {}).get("children", []):
            posts.setdefault(child["data"]["id"], child["data"])

    # 2. a dedicated subreddit (r/Notion, r/iphone...) if one exists
    sub = _subreddit_guess(query)
    if sub:
        try:
            data = http.get_json(f"{base}/r/{sub}/top.json", headers=headers,
                                 params={"t": "year", "limit": 30}, retries=0)
            for child in data.get("data", {}).get("children", []):
                posts.setdefault(child["data"]["id"], child["data"])
        except requests.RequestException:
            pass  # no such subreddit

    reviews: list[Review] = []
    for p in posts.values():
        r = _post_review(p)
        if r:
            reviews.append(r)

    # 3. comments from the most-discussed threads
    busiest = sorted(posts.values(), key=lambda p: p.get("num_comments", 0), reverse=True)[:8]
    for p in busiest:
        if len(reviews) >= limit:
            break
        try:
            listing = http.get_json(f"{base}/comments/{p['id']}.json", headers=headers,
                                    params={"limit": 60, "depth": 3, "sort": "top"}, retries=1)
        except requests.RequestException:
            continue
        if isinstance(listing, list) and len(listing) > 1:
            _flatten_comments(listing[1].get("data", {}).get("children", []), p, reviews)

    return reviews[:limit]
