"""Shared data structures."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Review:
    """One piece of customer voice: a review, comment, or post."""

    source: str                      # e.g. "Reddit", "Hacker News", "App Store"
    text: str
    url: str = ""
    author: str = ""
    date: str = ""                   # ISO date string when known
    rating: Optional[float] = None   # 1-5 star rating when the source has one
    title: str = ""
    context: str = ""                # subreddit, thread title, app name...
    upvotes: Optional[int] = None
    kind: str = "customer"           # "customer" or "expert" (editorial review)
    id: str = ""

    # filled in by analysis
    sentiment: str = ""              # positive | negative | mixed | neutral
    sentiment_score: float = 0.0     # -1 .. 1
    pain_points: list = field(default_factory=list)   # cluster labels
    praise: list = field(default_factory=list)        # cluster labels
    persona_hints: list = field(default_factory=list) # roles / segments mentioned

    def __post_init__(self) -> None:
        self.text = clean_text(self.text)
        if not self.id:
            self.id = fingerprint(self.text)[:10]

    def to_dict(self) -> dict:
        return asdict(self)


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")           # strip HTML tags
    text = (text.replace("&#x27;", "'").replace("&quot;", '"').replace("&amp;", "&")
                .replace("&gt;", ">").replace("&lt;", "<").replace("&#x2F;", "/"))
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # markdown links -> label
    return re.sub(r"\s+", " ", text).strip()


def fingerprint(text: str) -> str:
    norm = re.sub(r"[^a-z0-9]+", "", (text or "").lower())[:400]
    return hashlib.sha1(norm.encode()).hexdigest()
