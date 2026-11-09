"""Hacker News comments and Ask/Show HN posts, via the free Algolia API (no key)."""
from __future__ import annotations

import time

from .. import http
from ..models import Review

NAME = "Hacker News"
API = "https://hn.algolia.com/api/v1/search"


def collect(query: str, limit: int = 150, **_) -> list[Review]:
    q = f'"{query}"' if " " in query.strip() else query
    since = int(time.time()) - 2 * 365 * 24 * 3600   # last two years
    reviews: list[Review] = []
    page = 0
    while len(reviews) < limit and page < 4:
        data = http.get_json(API, params={
            "query": q, "tags": "comment", "hitsPerPage": 100, "page": page,
            "numericFilters": f"created_at_i>{since}",
        })
        hits = data.get("hits", [])
        for h in hits:
            text = h.get("comment_text") or ""
            if len(text) < 60:
                continue
            reviews.append(Review(
                source=NAME, text=text, author=h.get("author", ""),
                url=f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                date=(h.get("created_at") or "")[:10],
                title=h.get("story_title") or "", context=h.get("story_title") or "",
                upvotes=h.get("points"),
            ))
        if page + 1 >= data.get("nbPages", 0):
            break
        page += 1

    # Ask HN / Show HN posts with a body are often direct product feedback
    data = http.get_json(API, params={"query": q, "tags": "(ask_hn,show_hn)", "hitsPerPage": 30})
    for h in data.get("hits", []):
        body = h.get("story_text") or ""
        if len(body) >= 60:
            reviews.append(Review(
                source=NAME, text=f"{h.get('title', '')}. {body}", title=h.get("title", ""),
                url=f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                author=h.get("author", ""), date=(h.get("created_at") or "")[:10],
                context=h.get("title", ""), upvotes=h.get("points"),
            ))
    return reviews[:limit]
