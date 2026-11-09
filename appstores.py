"""Apple App Store and Google Play reviews (both free, no key).

Only used when the query is a software product that actually has an app whose
name matches; hardware like "iPhone 18 Pro" is skipped automatically.
"""
from __future__ import annotations

import re

import requests

from .. import http
from ..models import Review


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def name_matches(query: str, app_name: str) -> bool:
    """True when every word of the query appears in the app's name (first part)."""
    head = re.split(r"[:\-–—|]", app_name, maxsplit=1)[0]
    q, n = _tokens(query), _tokens(head)
    return bool(q) and q <= n and len(n) <= len(q) + 3


# ---------------------------------------------------------------- App Store
def collect_app_store(query: str, limit: int = 150, country: str = "us", **_) -> list[Review]:
    found = http.get_json("https://itunes.apple.com/search",
                          params={"term": query, "entity": "software", "limit": 5, "country": country})
    app = next((a for a in found.get("results", []) if name_matches(query, a.get("trackName", ""))), None)
    if not app:
        return []
    app_id, app_name = app["trackId"], app["trackName"]
    reviews: list[Review] = []
    for page in range(1, 11):                      # Apple serves up to 10 pages x 50
        if len(reviews) >= limit:
            break
        url = (f"https://itunes.apple.com/{country}/rss/customerreviews/"
               f"page={page}/id={app_id}/sortby=mostrecent/json")
        try:
            feed = http.get_json(url, retries=1).get("feed", {})
        except (requests.RequestException, ValueError):
            break
        entries = feed.get("entry", [])
        if isinstance(entries, dict):
            entries = [entries]
        if not entries:
            break
        for e in entries:
            rating = e.get("im:rating", {}).get("label")
            if rating is None:                     # app metadata entry, not a review
                continue
            title = e.get("title", {}).get("label", "")
            body = e.get("content", {}).get("label", "")
            reviews.append(Review(
                source="App Store", text=f"{title}. {body}", title=title, rating=float(rating),
                author=e.get("author", {}).get("name", {}).get("label", ""),
                date=(e.get("updated", {}).get("label") or "")[:10],
                url=app.get("trackViewUrl", ""), context=app_name,
            ))
    return reviews[:limit]


# -------------------------------------------------------------- Google Play
def collect_google_play(query: str, limit: int = 150, **_) -> list[Review]:
    try:
        from google_play_scraper import Sort, reviews as gp_reviews, search
    except ImportError:
        return []
    hits = search(query, n_hits=5, lang="en", country="us") or []
    app = next((h for h in hits if h.get("appId") and name_matches(query, h.get("title", ""))), None)
    if not app:
        return []
    out: list[Review] = []
    for sort in (Sort.MOST_RELEVANT, Sort.NEWEST):
        batch, _ = gp_reviews(app["appId"], lang="en", country="us", sort=sort, count=limit // 2 + 1)
        for r in batch:
            if len(r.get("content") or "") < 25:
                continue
            at = r.get("at")
            out.append(Review(
                source="Google Play", text=r["content"], rating=float(r.get("score") or 0) or None,
                author=r.get("userName", ""), date=at.date().isoformat() if at else "",
                url=f"https://play.google.com/store/apps/details?id={app['appId']}",
                context=app.get("title", ""), upvotes=r.get("thumbsUpCount"),
            ))
    return out[:limit]
