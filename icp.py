"""Generate the top 5 Ideal Customer Profiles from the clustered voice of customer."""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from .analyze import Analysis

ICP_PROMPT = """You are a B2B/B2C go-to-market strategist. Using ONLY the voice-of-customer evidence
below about "{query}", define the TOP 5 Ideal Customer Profiles (ICPs): the segments that get the
most value, are most vocal, or represent the biggest opportunity (including people the current
product fails, where a fix would win them).

Evidence summary ({total} feedback items from {sources}):
Sentiment: {sentiment}

PAIN-POINT CLUSTERS (label | share of feedback | severity | description | sample quotes):
{pains}

PRAISE CLUSTERS:
{praise}

SEGMENTS MENTIONED BY WRITERS (segment | mentions):
{personas}

PAINS AND PRAISE BY SEGMENT (segment -> top pains / top praise):
{crosstab}

Return JSON: {{"icps": [ {{
  "rank": 1,
  "name": "<memorable segment name>",
  "who": "<role / life stage / company type and size, 1-2 sentences>",
  "jobs_to_be_done": ["...", "..."],
  "top_pains": ["<tie to cluster labels above>", ...],
  "what_they_love": ["...", ...],
  "buying_triggers": ["<events that make them look for a product>", ...],
  "objections": ["...", ...],
  "where_to_find_them": ["<communities, channels, events>", ...],
  "messaging_angle": "<one-line value proposition that would land>",
  "evidence": ["<short verbatim quote from the evidence>", "..."],
  "estimated_share_of_voice": "<e.g. ~18% of feedback>",
  "confidence": "High|Medium|Low",
  "why_this_rank": "<one sentence>"
}} ] }}
Rank 1 = most attractive. Do not invent statistics beyond the evidence; mark Low confidence when the
evidence is thin."""


def _crosstab(a: Analysis) -> dict[str, dict[str, Counter]]:
    ct: dict[str, dict[str, Counter]] = defaultdict(lambda: {"pains": Counter(), "praise": Counter()})
    for r in a.reviews:
        for p in r.persona_hints:
            ct[p]["pains"].update(r.pain_points)
            ct[p]["praise"].update(r.praise)
    return ct


def _evidence_block(a: Analysis) -> dict:
    fmt_cluster = lambda c: (f"- {c['label']} | {c['share']:.0%} | {c.get('severity') or '-'} | "
                             f"{c.get('description') or ''} | "
                             + " / ".join(f'"{q["text"][:180]}"' for q in c["quotes"][:3]))
    ct = _crosstab(a)
    cross = "\n".join(
        f"- {seg}: pains={[k for k, _ in v['pains'].most_common(3)]} praise={[k for k, _ in v['praise'].most_common(3)]}"
        for seg, v in sorted(ct.items(), key=lambda kv: -sum(kv[1]['pains'].values()) - sum(kv[1]['praise'].values()))[:12])
    return {
        "query": a.query, "total": len(a.reviews),
        "sources": ", ".join(a.by_source) or "public sources",
        "sentiment": json.dumps(a.sentiment_counts),
        "pains": "\n".join(fmt_cluster(c) for c in a.pain_clusters[:12]) or "(none)",
        "praise": "\n".join(fmt_cluster(c) for c in a.praise_clusters[:8]) or "(none)",
        "personas": "\n".join(f"- {p['segment']} | {p['mentions']}" for p in a.personas[:12]) or "(none)",
        "crosstab": cross or "(none)",
    }


def generate_icps_llm(a: Analysis, cfg) -> list[dict]:
    from . import llm
    out = llm.complete_json(cfg, ICP_PROMPT.format(**_evidence_block(a)), max_tokens=8000)
    icps = out.get("icps", out) if isinstance(out, dict) else out
    return [i for i in icps if isinstance(i, dict)][:5]


def generate_icps_local(a: Analysis) -> list[dict]:
    """Evidence-only fallback: rank segments by how often they speak up, then attach their
    own top pains, praise and quotes. Less nuanced than the LLM version, but free."""
    ct = _crosstab(a)
    total = len(a.reviews) or 1
    seg_reviews: dict[str, list] = defaultdict(list)
    for r in a.reviews:
        for p in r.persona_hints:
            seg_reviews[p].append(r)

    ranked = sorted(seg_reviews.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]
    icps = []
    for rank, (seg, rs) in enumerate(ranked, 1):
        pains = [k for k, _ in ct[seg]["pains"].most_common(4)]
        praise = [k for k, _ in ct[seg]["praise"].most_common(4)]
        neg = sum(1 for r in rs if r.sentiment in ("negative", "mixed"))
        quotes = sorted(rs, key=lambda r: (r.upvotes or 0), reverse=True)[:3]
        n = len(rs)
        icps.append({
            "rank": rank, "name": seg,
            "who": f"{seg}: people whose feedback shows they belong to this segment "
                   f"({n} of {total} feedback items).",
            "jobs_to_be_done": [f"Get value from {a.query} in their {seg.lower()} workflow"],
            "top_pains": pains or ["No dominant pain in this segment"],
            "what_they_love": praise or ["No dominant praise in this segment"],
            "buying_triggers": [], "objections": pains[:2],
            "where_to_find_them": sorted({r.context.split(' · ')[0] for r in rs if r.context})[:5],
            "messaging_angle": (f"Solve {pains[0].lower()} for {seg.lower()}" if pains
                                else f"Double down on {praise[0].lower()}" if praise else ""),
            "evidence": [r.text[:220] for r in quotes],
            "estimated_share_of_voice": f"~{n / total:.0%} of feedback",
            "confidence": "Medium" if n >= 15 else "Low",
            "why_this_rank": f"{n} mentions; {neg / n:.0%} of them negative or mixed.",
        })
    return icps


def generate_icps(a: Analysis, cfg=None) -> tuple[list[dict], str]:
    """Returns (icps, engine_used)."""
    if cfg is not None:
        try:
            return generate_icps_llm(a, cfg), f"llm:{cfg.model}"
        except Exception as exc:
            icps = generate_icps_local(a)
            return icps, f"local (LLM failed: {type(exc).__name__}: {str(exc)[:120]})"
    return generate_icps_local(a), "local"
