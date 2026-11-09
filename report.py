"""Markdown export of an analysis (and ICPs, when generated)."""
from __future__ import annotations

import json
from pathlib import Path

from .analyze import Analysis
from .models import Review


def load_demo() -> tuple[str, list[Review]]:
    data = json.loads((Path(__file__).parent / "demo_data.json").read_text())
    return data["query"], [Review(**r) for r in data["reviews"]]


def to_markdown(a: Analysis, icps: list[dict] | None = None) -> str:
    total = len(a.reviews) or 1
    sc = a.sentiment_counts
    lines = [f"# Voice of Customer: {a.query}", "",
             f"{len(a.reviews)} feedback items · engine: {a.engine}", "",
             "## Sentiment", "",
             " · ".join(f"{k}: {v} ({v / total:.0%})" for k, v in sorted(sc.items(), key=lambda kv: -kv[1])), "",
             "| Source | Items | Positive | Negative | Mixed | Neutral |", "|---|---|---|---|---|---|"]
    for src, c in a.by_source.items():
        lines.append(f"| {src} | {sum(c.values())} | {c.get('positive', 0)} | {c.get('negative', 0)} "
                     f"| {c.get('mixed', 0)} | {c.get('neutral', 0)} |")
    for title, clusters in (("Pain points", a.pain_clusters), ("What customers love", a.praise_clusters)):
        lines += ["", f"## {title}", ""]
        for c in clusters:
            sev = f" · severity {c['severity']}" if c.get("severity") else ""
            lines.append(f"### {c['label']} — {c['count']} items ({c['share']:.0%}){sev}")
            if c.get("description"):
                lines.append(c["description"])
            for q in c["quotes"][:4]:
                link = f" ([{q['source']}]({q['url']}))" if q.get("url") else f" ({q['source']})"
                lines.append(f"> {q['text']}{link}")
            lines.append("")
    if icps:
        lines += ["## Top 5 ICPs", ""]
        for i in icps:
            lines += [f"### {i.get('rank', '')}. {i.get('name', '')}",
                      f"**Who:** {i.get('who', '')}", "",
                      f"**Share of voice:** {i.get('estimated_share_of_voice', '')} · "
                      f"**Confidence:** {i.get('confidence', '')}", ""]
            for key, label in (("jobs_to_be_done", "Jobs to be done"), ("top_pains", "Top pains"),
                               ("what_they_love", "What they love"), ("buying_triggers", "Buying triggers"),
                               ("objections", "Objections"), ("where_to_find_them", "Where to find them")):
                vals = i.get(key) or []
                if vals:
                    lines.append(f"**{label}:** " + "; ".join(map(str, vals)))
            if i.get("messaging_angle"):
                lines.append(f"**Messaging angle:** {i['messaging_angle']}")
            for e in (i.get("evidence") or [])[:3]:
                lines.append(f"> {e}")
            lines.append("")
    if a.warnings:
        lines += ["## Notes", ""] + [f"- {w}" for w in a.warnings]
    return "\n".join(lines)
