"""Reviews from trusted public websites, found with a free DuckDuckGo search.

Two kinds of pages are used:
  * user-opinion pages (GSMArena user opinions) -> kind="customer"
  * editorial reviews from established outlets    -> kind="expert"
Only domains on TRUSTED_DOMAINS are fetched.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

import requests

from .. import http
from ..models import Review

NAME = "Review sites"

TRUSTED_DOMAINS = {
    # editorial / expert reviews
    "theverge.com", "techradar.com", "cnet.com", "tomsguide.com", "pcmag.com", "zdnet.com",
    "engadget.com", "wired.com", "arstechnica.com", "macrumors.com", "9to5mac.com",
    "androidauthority.com", "gsmarena.com", "zapier.com", "nytimes.com", "rtings.com",
    "techcrunch.com", "forbes.com", "pcworld.com", "macworld.com", "digitaltrends.com",
}
OPINION_WORDS = re.compile(
    r"\b(but|however|love|hate|disappoint\w*|wish|problem\w*|issue\w*|great|best|worst|"
    r"annoying|frustrat\w*|expensive|cheap|price|slow|fast|bug\w*|battery|recommend|"
    r"downside|upside|con|pro)\b", re.I)


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _search(query: str, max_results: int) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # older package name
        except ImportError:
            return []
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results) or [])


def _gsmarena_opinions(url: str, html: str, product: str) -> list[Review]:
    from lxml import html as lh
    doc = lh.fromstring(html)
    out = []
    for thread in doc.xpath("//div[contains(@class,'user-thread')]"):
        text = " ".join(p.text_content() for p in thread.xpath(".//p[contains(@class,'uopin')]"))
        author = " ".join(e.text_content() for e in thread.xpath(".//li[contains(@class,'uname')]")).strip()
        date = " ".join(e.text_content() for e in thread.xpath(".//li[contains(@class,'upost')]")).strip()
        if len(text) >= 40:
            out.append(Review(source="GSMArena", text=text, url=url, author=author,
                              date=date[:20], context=product))
    return out


def _article_passages(url: str, html: str) -> list[Review]:
    try:
        import trafilatura
    except ImportError:
        return []
    meta = trafilatura.extract_metadata(html)
    body = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
    title = getattr(meta, "title", "") or ""
    date = getattr(meta, "date", "") or ""
    paras = [p.strip() for p in body.split("\n") if 180 <= len(p.strip()) <= 1500]
    paras = [p for p in paras if OPINION_WORDS.search(p)][:8]
    site = _domain(url)
    return [Review(source=site, text=p, url=url, title=title, date=str(date)[:10],
                   context=title, kind="expert") for p in paras]


def collect(query: str, limit: int = 80, **_) -> list[Review]:
    results = _search(f"{query} review", 25) + _search(f"{query} user opinions problems", 15)
    seen, reviews = set(), []
    for r in results:
        url = r.get("href") or r.get("url") or ""
        dom = _domain(url)
        if not url or url in seen or not any(dom == d or dom.endswith("." + d) for d in TRUSTED_DOMAINS):
            continue
        seen.add(url)
        try:
            html = http.get(url, headers={"Accept": "text/html"}, retries=0).text
        except requests.RequestException:
            continue
        if dom.endswith("gsmarena.com") and "-reviews-" in url:
            reviews.extend(_gsmarena_opinions(url, html, query))
        else:
            reviews.extend(_article_passages(url, html))
        if len(reviews) >= limit:
            break
    return reviews[:limit]
