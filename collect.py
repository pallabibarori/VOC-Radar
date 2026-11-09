"""Run every source in parallel, then dedupe and filter for relevance."""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from .models import Review, fingerprint
from .sources import appstores, hackernews, reddit, web

SOURCES: dict[str, Callable[..., list[Review]]] = {
    "Reddit": reddit.collect,
    "Hacker News": hackernews.collect,
    "App Store": appstores.collect_app_store,
    "Google Play": appstores.collect_google_play,
    "Review sites": web.collect,
}


@dataclass
class CollectionResult:
    query: str
    reviews: list[Review]
    per_source: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    dropped_irrelevant: int = 0
    dropped_duplicates: int = 0


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def is_relevant(review: Review, query: str, context_terms: list[str] | None = None) -> bool:
    hay = f"{review.text} {review.title} {review.context}".lower()
    hay_tokens = set(_tokens(hay))
    q_tokens = _tokens(query)
    if not q_tokens or not all(t in hay_tokens for t in q_tokens):
        return False
    if len(q_tokens) == 1:
        # single dictionary words ("notion", "linear", "monday") are often used generically
        q = re.escape(q_tokens[0])
        total = len(re.findall(rf"\b{q}\b", hay))
        generic = len(re.findall(rf"\b(the|a|an|this|that|any|no|my)\s+{q}\s+(of|that|is)\b", hay))
        dedicated = q_tokens[0] in _tokens(review.context)   # r/Notion, the Notion app...
        if total and generic >= total and not dedicated:
            return False
    if context_terms:
        terms = [t.strip().lower() for t in context_terms if t.strip()]
        if terms and not any(t in hay for t in terms) and q_tokens[0] not in _tokens(review.context):
            return False
    return True


def collect(query: str, sources: list[str] | None = None, limit_per_source: int = 150,
            context_terms: list[str] | None = None,
            progress: Callable[[str, int, str | None], None] | None = None) -> CollectionResult:
    names = sources or list(SOURCES)
    raw: list[Review] = []
    result = CollectionResult(query=query, reviews=[])

    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        futures = {pool.submit(SOURCES[n], query, limit=limit_per_source): n for n in names}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                items = fut.result() or []
                raw.extend(items)
                result.per_source[name] = len(items)
                if progress:
                    progress(name, len(items), None)
            except Exception as exc:  # a failing source must not sink the run
                result.per_source[name] = 0
                result.errors[name] = f"{type(exc).__name__}: {exc}"[:300]
                if progress:
                    progress(name, 0, result.errors[name])

    seen: set[str] = set()
    for r in raw:
        fp = fingerprint(r.text)
        if fp in seen:
            result.dropped_duplicates += 1
            continue
        seen.add(fp)
        if not is_relevant(r, query, context_terms):
            result.dropped_irrelevant += 1
            continue
        result.reviews.append(r)
    return result
