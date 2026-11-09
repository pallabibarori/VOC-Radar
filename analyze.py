"""Sentiment scoring and clustering into pain points and praise themes.

Two engines:
  * local - free, offline: VADER sentiment at sentence level, a pain-point
            taxonomy, and TF-IDF + KMeans for themes the taxonomy misses.
  * llm   - an LLM tags every review, then merges the tags into clusters.
            Counts and shares are always computed in code, never by the model.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .models import Review

# ------------------------------------------------------------------ taxonomy
THEMES: dict[str, list[str]] = {
    "Pricing & value": [r"pric\w*", r"expensive", r"overpriced", r"subscription", r"paywall",
                        r"costs? too much", r"\bworth\b", r"\bbilling\b", r"money", r"\bplan\b", r"\bfee\w*", r"afford\w*", r"\$\d+", r"\bvalue\b"],
    "Performance & speed": [r"\bslow\w*", r"\blag(gy|ging|s)?\b(?! behind)", r"sluggish", r"loading", r"performance",
                            r"freez\w*", r"\bhangs\b", r"\bhanging\b", r"takes forever", r"snappy", r"\bfast\b", r"speed"],
    "Bugs & reliability": [r"\bbug\w*", r"crash\w*", r"broken", r"glitch\w*", r"\berror\w*",
                           r"stopped working", r"doesn'?t work", r"unreliable", r"data loss", r"lost (my|all|our|the) \w+", r"reliab\w*", r"stable"],
    "Sync & offline": [r"\bsync\w*", r"offline", r"no internet", r"connectivity", r"conflict\w*"],
    "Ease of use & learning curve": [r"complicated", r"confusing", r"learning curve", r"overwhelm\w*",
                                    r"\bcomplex\w*", r"hard to (use|learn|figure)", r"clutter\w*", r"bloat\w*", r"intuitive", r"easy to use"],
    "Missing features": [r"\bwish\b", r"missing", r"\black\w*", r"no way to", r"doesn'?t (have|support)",
                         r"would be nice", r"please add", r"feature request", r"still no"],
    "Battery & charging": [r"battery", r"\bdrain\w*", r"charg\w*", r"battery life"],
    "Heat & thermals": [r"overheat\w*", r"\bhot\b", r"\bheat\w*", r"thermal\w*", r"throttl\w*"],
    "Camera": [r"camera", r"\bphotos?\b", r"\bvideo\b", r"\blens\w*", r"\bzoom\b", r"telephoto"],
    "Display & screen": [r"\bscreen\b", r"display", r"brightness", r"refresh rate", r"\bpwm\b"],
    "Build & design": [r"build quality", r"scratch\w*", r"\bweight\b", r"heavy", r"camera bump",
                       r"\bdesign\b", r"\bcolou?rs?\b", r"\bbutton\w*", r"durab\w*", r"premium feel"],
    "Customer support": [r"customer (service|support)", r"\bsupport team\b", r"refund\w*", r"\bticket\w*",
                         r"no response", r"never (got|heard) back", r"support was"],
    "Privacy & security": [r"privacy", r"security", r"tracking", r"encrypt\w*", r"(sell|use|train\w* on|using) (my|our) data", r"\bgdpr\b"],
    "Integrations & data portability": [r"integrat\w*", r"\bapi\b", r"\bexport\w*", r"\bimport\w*",
                                        r"lock-?in", r"ecosystem", r"compatib\w*", r"migrat\w*"],
    "Mobile experience": [r"mobile app", r"ios app", r"android app", r"on mobile", r"phone app", r"on my phone"],
    "AI features": [r"\bai\b", r"copilot", r"apple intelligence", r"chatgpt", r"\bllm\w*", r"\bgpt\w*"],
    "Collaboration & permissions": [r"permission\w*", r"\bsharing\b", r"collaborat\w*", r"real-?time editing",
                                    r"\badmin\w*", r"\bguests?\b", r"multiplayer"],
    "Templates": [r"template\w*"],
    "Flexibility & customization": [r"flexib\w*", r"customi[sz]\w*", r"versatil\w*", r"building blocks", r"lego"],
    "All-in-one workspace": [r"all-in-one", r"replaced (\w+ ){0,2}(tools|apps)", r"one place", r"everything in one"],
    "Databases & views": [r"database\w*", r"board view", r"relations?\b", r"formula\w*", r"kanban", r"timeline\w*"],
    "Editor & writing": [r"\beditor\b", r"code blocks?", r"markdown", r"formatting"],
    "Search & navigation": [r"\bsearch\w*", r"navigat\w*", r"find (anything|things|stuff)"],
}
_THEME_RX = {k: re.compile("|".join(v), re.I) for k, v in THEMES.items()}

COMPLAINT_RX = re.compile(
    r"\b(too|wish|missing|no way|doesn'?t|don'?t|can'?t|cannot|won'?t|isn'?t|aren'?t|never|"
    r"stopped|still no|lacks?|lacking|worse|worst|terrible|awful|hate|annoy\w*|frustrat\w*|"
    r"disappoint\w*|useless|unusable|accidentally|buried|confus\w*|clunky|deal ?breaker|problem\w*|issue\w*|fail\w*|broke\w*|overpriced|expensive|slow)\b", re.I)

PERSONAS: dict[str, str] = {
    "Students": r"\bstudents?\b|\bcollege\b|\buniversity\b|\bexams?\b|\bclass(es)?\b|\bschool\b",
    "Software developers": r"\bdevelopers?\b|\bengineers?\b|\bprogrammers?\b|\bdevs?\b|\bcoding\b",
    "Designers & creatives": r"\bdesigners?\b|\bcreatives?\b|\bartists?\b|\billustrat\w*",
    "Product & project managers": r"\bproduct managers?\b|\bproject managers?\b|\bPMs?\b|\bsprints?\b|\broadmaps?\b",
    "Founders & startups": r"\bfounders?\b|\bstartups?\b|\bco-?founder\b|\bsolo ?preneur\w*",
    "Small businesses": r"\bsmall business\w*\b|\bmy (business|shop|store|agency)\b|\bSMBs?\b",
    "Freelancers & consultants": r"\bfreelanc\w*|\bconsultants?\b|\bclients?\b",
    "Enterprise & IT admins": r"\benterprise\b|\bIT (team|admin|department)\b|\bcompliance\b|\bSSO\b|\bSOC ?2\b",
    "Teams & companies": r"\bour team\b|\bmy team\b|\bteammates?\b|\bcoworkers?\b|\bcompany\b",
    "Writers & researchers": r"\bwriters?\b|\bwriting\b|\bresearch(ers?)?\b|\bthesis\b|\bnovels?\b",
    "Marketers & content creators": r"\bmarketers?\b|\bmarketing\b|\bcontent creators?\b|\byoutubers?\b|\bcreators?\b",
    "Personal productivity users": r"\bhabit\w*\b|\bjournal\w*|\bpersonal (use|life)\b|\bsecond brain\b|\bto-?do\b|\bADHD\b",
    "Photographers & videographers": r"\bphotographers?\b|\bvideographers?\b|\bfilmmak\w*|\bshoot(ing)? (video|raw|prores)\b",
    "Gamers": r"\bgam(ing|ers?)\b|\bgenshin\b|\bfps\b",
    "Upgraders from older models": r"\b(coming|upgrad\w*) from\b|\bmy old\b|\bprevious (model|phone)\b",
    "Switchers from competitors": r"\bswitch(ed|ing)? (from|to)\b|\bmoved (from|to)\b|\bcoming from android\b|\bmigrat\w* from\b",
    "Parents & families": r"\bmy (kids?|son|daughter|wife|husband|family)\b|\bparents?\b",
}
_PERSONA_RX = {k: re.compile(v, re.I if not k.startswith("Product") else 0) for k, v in PERSONAS.items()}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Analysis:
    query: str
    engine: str
    reviews: list[Review]
    pain_clusters: list[dict] = field(default_factory=list)
    praise_clusters: list[dict] = field(default_factory=list)
    personas: list[dict] = field(default_factory=list)
    sentiment_counts: dict = field(default_factory=dict)
    by_source: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query, "engine": self.engine,
            "summary": {"total": len(self.reviews), "sentiment": self.sentiment_counts,
                        "by_source": self.by_source},
            "pain_points": self.pain_clusters, "praise": self.praise_clusters,
            "personas": self.personas, "warnings": self.warnings,
            "reviews": [r.to_dict() for r in self.reviews],
        }


# ------------------------------------------------------------------ helpers
_vader = None


def _polarity(text: str) -> float:
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    return _vader.polarity_scores(text)["compound"]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 12]


_CLAUSE_SPLIT = re.compile(r",?\s+\b(?:but|however|although|though|except|yet)\b\s+|;\s+", re.I)


def clauses(text: str) -> list[str]:
    """Sentences split further at contrast words, so "love X, but Y is slow" yields two opinions."""
    out = []
    for s in sentences(text) or [text]:
        out.extend(c.strip() for c in _CLAUSE_SPLIT.split(s) if len(c.strip()) > 8)
    return out


def score_review(r: Review) -> None:
    sents = sentences(r.text) or [r.text]
    pols = [_polarity(s) for s in sents]
    overall = _polarity(r.text[:3000])
    complaints = sum(1 for s in sents if COMPLAINT_RX.search(s))
    if complaints and overall > -0.05:
        overall -= min(0.15 * complaints, 0.45)          # VADER misses "too expensive", "still no X"
    if r.rating is not None:                             # stars are the strongest signal
        overall = 0.5 * overall + 0.5 * ((r.rating - 3) / 2)
    strong_pos = any(p >= 0.45 for p in pols)
    strong_neg = any(p <= -0.35 for p in pols) or complaints >= 2
    if strong_pos and strong_neg and -0.35 < overall < 0.5:
        label = "mixed"
    elif overall >= 0.25:
        label = "positive"
    elif overall <= -0.1:
        label = "negative"
    else:
        label = "mixed" if (strong_pos and complaints) else "neutral"
    r.sentiment, r.sentiment_score = label, round(max(-1.0, min(1.0, overall)), 3)


def _persona_hints(r: Review) -> list[str]:
    hay = f"{r.text} {r.context}"
    return [k for k, rx in _PERSONA_RX.items() if rx.search(hay)]


def _quote(text: str, limit: int = 260) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def _build_clusters(label_to_items: dict[str, list[tuple[Review, str]]], total: int,
                    descriptions: dict[str, str] | None = None, kind: str = "pain") -> list[dict]:
    clusters = []
    for label, items in label_to_items.items():
        uniq: dict[str, tuple[Review, str]] = {}
        for r, q in items:
            uniq.setdefault(r.id, (r, q))
        rs = [r for r, _ in uniq.values()]
        if not rs:
            continue
        ranked = sorted(uniq.values(), key=lambda t: ((t[0].upvotes or 0), len(t[1])), reverse=True)
        quotes, seen_src = [], Counter()
        for r, q in ranked:                     # spread quotes across sources
            if seen_src[r.source] >= 2 and len(ranked) > 6:
                continue
            seen_src[r.source] += 1
            quotes.append({"text": _quote(q), "source": r.source, "url": r.url,
                           "review_id": r.id, "rating": r.rating})
            if len(quotes) >= 6:
                break
        avg = sum(r.sentiment_score for r in rs) / len(rs)
        n = len(rs)
        clusters.append({
            "label": label, "kind": kind,
            "description": (descriptions or {}).get(label, ""),
            "count": n, "share": round(n / total, 3) if total else 0,
            "avg_sentiment": round(avg, 3),
            "severity": _severity(n, total, avg) if kind == "pain" else None,
            "sources": dict(Counter(r.source for r in rs).most_common()),
            "review_ids": [r.id for r in rs], "quotes": quotes,
        })
    clusters.sort(key=lambda c: c["count"], reverse=True)
    return clusters


def _severity(n: int, total: int, avg: float) -> str:
    share = n / total if total else 0
    score = share * 100 * (1 + max(0.0, -avg))      # frequency x intensity
    return "High" if score >= 12 else "Medium" if score >= 5 else "Low"


def _summarise(a: Analysis) -> None:
    a.sentiment_counts = dict(Counter(r.sentiment for r in a.reviews))
    by_src: dict[str, Counter] = defaultdict(Counter)
    for r in a.reviews:
        by_src[r.source][r.sentiment] += 1
    a.by_source = {k: dict(v) for k, v in by_src.items()}
    pc = Counter(p for r in a.reviews for p in r.persona_hints)
    total = len(a.reviews)
    a.personas = [{"segment": k, "mentions": v, "share": round(v / total, 3) if total else 0}
                  for k, v in pc.most_common()]


# ------------------------------------------------------------------ local engine
def _query_stopwords(query: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", query.lower()))


def _emerging_themes(leftovers: list[Review], query: str, max_k: int = 5) -> dict[str, list[tuple[Review, str]]]:
    if len(leftovers) < 8:
        return {}
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
    stop = list(ENGLISH_STOP_WORDS | _query_stopwords(query) | {
        "just", "like", "really", "use", "using", "used", "get", "got", "think", "know", "thing",
        "things", "make", "way", "want", "does", "did", "don", "ve", "ll", "re", "im", "app", "new",
        "good", "great", "bad", "people", "time", "lot", "pro", "apple", "http", "https", "com", "www"})
    texts = [r.text[:1500] for r in leftovers]
    vec = TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=2, max_df=0.6,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b")
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return {}
    if X.shape[1] < 5:
        return {}
    k = max(2, min(max_k, len(leftovers) // 8))
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    terms = vec.get_feature_names_out()
    out: dict[str, list[tuple[Review, str]]] = {}
    for c in range(k):
        members = [leftovers[i] for i, lab in enumerate(km.labels_) if lab == c]
        if len(members) < 3:
            continue
        top = [terms[i] for i in km.cluster_centers_[c].argsort()[::-1][:3]]
        label = "Other: " + ", ".join(top)
        out[label] = [(r, (sentences(r.text) or [r.text])[0]) for r in members]
    return out


def analyze_local(query: str, reviews: list[Review]) -> Analysis:
    a = Analysis(query=query, engine="local", reviews=reviews)
    pains: dict[str, list] = defaultdict(list)
    praise: dict[str, list] = defaultdict(list)
    leftovers: list[Review] = []
    for r in reviews:
        score_review(r)
        r.persona_hints = _persona_hints(r)
        r.pain_points, r.praise = [], []
        for s in clauses(r.text) or [r.text]:
            themes = [k for k, rx in _THEME_RX.items() if rx.search(s)]
            if not themes:
                continue
            p = _polarity(s)
            is_neg = p <= -0.05 or bool(COMPLAINT_RX.search(s))
            is_pos = (p >= (0.5 if r.sentiment == "negative" else 0.25)
                      and not COMPLAINT_RX.search(s))
            for t in themes:
                if is_neg:
                    if t not in r.pain_points:
                        r.pain_points.append(t)
                        pains[t].append((r, s))
                elif is_pos:
                    if t not in r.praise:
                        r.praise.append(t)
                        praise[t].append((r, s))
        if r.sentiment in ("negative", "mixed") and not r.pain_points:
            leftovers.append(r)

    for label, items in _emerging_themes(leftovers, query).items():
        pains[label] = items
        for r, _ in items:
            r.pain_points.append(label)

    total = len(reviews)
    a.pain_clusters = [c for c in _build_clusters(pains, total, kind="pain") if c["count"] >= 2]
    a.praise_clusters = [c for c in _build_clusters(praise, total, kind="praise") if c["count"] >= 2]
    _summarise(a)
    return a


# ------------------------------------------------------------------ LLM engine
TAG_PROMPT = """You are a voice-of-customer analyst studying feedback about "{query}".
For EACH item below return an object:
  {{"id": "<id>", "relevant": true|false, "sentiment": "positive|negative|mixed|neutral",
    "pain_points": [{{"theme": "<2-5 word problem label>", "quote": "<verbatim excerpt, max 25 words>"}}],
    "praise": [{{"theme": "<2-5 word label>", "quote": "<verbatim excerpt, max 25 words>"}}],
    "persona": ["<who the writer is or what they use it for, e.g. 'student', 'engineering team lead', 'upgrading from iPhone 13'>"]}}
Rules: relevant=false if the item is not actually about {query} (e.g. the common word used generically,
or a different product). Use short, reusable theme labels so similar complaints share a label
("slow on large pages", not a sentence). Empty lists are fine. Persona only when the text supports it.
Return a JSON array with one object per item, same ids.

ITEMS:
{items}"""

MERGE_PROMPT = """You are consolidating voice-of-customer tags about "{query}".
Below are raw theme labels with how often each appeared.
Group them into clusters:
  - at most 10 pain-point clusters, at most 6 praise clusters, at most 8 persona segments.
Return JSON:
{{"pain_points": [{{"label": "<clear 2-5 word name>", "description": "<one sentence on the underlying problem>", "members": ["<raw label>", ...]}}],
  "praise": [{{"label": "...", "description": "...", "members": [...]}}],
  "personas": [{{"label": "<segment name>", "members": [...]}}]}}
Every raw label should land in exactly one cluster of its type; put rare oddballs in the closest cluster.

RAW PAIN LABELS:
{pains}

RAW PRAISE LABELS:
{praise}

RAW PERSONA LABELS:
{personas}"""


def analyze_llm(query: str, reviews: list[Review], cfg, batch_size: int = 60,
                progress=None) -> Analysis:
    from . import llm
    a = Analysis(query=query, engine=f"llm:{cfg.model}", reviews=[])
    tags: dict[str, dict] = {}
    by_id = {r.id: r for r in reviews}
    batches = [reviews[i:i + batch_size] for i in range(0, len(reviews), batch_size)]
    for bi, batch in enumerate(batches):
        if progress:
            progress(f"Tagging batch {bi + 1}/{len(batches)}")
        items = "\n".join(
            f'- id={r.id} | source={r.source}' + (f' | rating={r.rating:g}/5' if r.rating else "")
            + f' | context={r.context[:60]} | text="{r.text[:700]}"' for r in batch)
        out = llm.complete_json(cfg, TAG_PROMPT.format(query=query, items=items), max_tokens=12000)
        for t in out if isinstance(out, list) else out.get("items", []):
            if isinstance(t, dict) and str(t.get("id")) in by_id:
                tags[str(t["id"])] = t

    kept = []
    for r in reviews:
        t = tags.get(r.id)
        score_review(r)                       # numeric score always from VADER/stars
        if t is not None:
            if t.get("relevant") is False:
                continue
            s = str(t.get("sentiment", "")).lower()
            if s in ("positive", "negative", "mixed", "neutral"):
                r.sentiment = s
        kept.append(r)
    a.reviews = kept
    dropped = len(reviews) - len(kept)
    if dropped:
        a.warnings.append(f"LLM marked {dropped} items as not about {query}; they were removed.")
    untagged = sum(1 for r in kept if r.id not in tags)
    if untagged:
        a.warnings.append(f"{untagged} items came back untagged; they count toward sentiment only.")

    def _labels(key: str) -> Counter:
        c = Counter()
        for r in kept:
            for p in (tags.get(r.id, {}).get(key) or []):
                lab = p.get("theme") if isinstance(p, dict) else p
                if lab:
                    c[str(lab).strip().lower()] += 1
        return c

    pain_raw, praise_raw, persona_raw = _labels("pain_points"), _labels("praise"), _labels("persona")
    if progress:
        progress("Merging themes into clusters")
    fmt = lambda c: "\n".join(f"{k} ({v})" for k, v in c.most_common(400)) or "(none)"
    merged = llm.complete_json(cfg, MERGE_PROMPT.format(
        query=query, pains=fmt(pain_raw), praise=fmt(praise_raw), personas=fmt(persona_raw)))

    def _mapping(key: str) -> tuple[dict, dict]:
        m, desc = {}, {}
        for cl in merged.get(key, []) or []:
            label = cl.get("label", "").strip()
            if not label:
                continue
            desc[label] = cl.get("description", "")
            for mem in cl.get("members", []) or []:
                m[str(mem).strip().lower()] = label
        return m, desc

    pain_map, pain_desc = _mapping("pain_points")
    praise_map, praise_desc = _mapping("praise")
    persona_map, _ = _mapping("personas")

    pains: dict[str, list] = defaultdict(list)
    praise: dict[str, list] = defaultdict(list)
    for r in kept:
        t = tags.get(r.id, {})
        r.pain_points, r.praise = [], []
        for key, mapping, bucket, attr in (("pain_points", pain_map, pains, r.pain_points),
                                           ("praise", praise_map, praise, r.praise)):
            for p in t.get(key) or []:
                if not isinstance(p, dict):
                    p = {"theme": str(p), "quote": ""}
                raw = str(p.get("theme", "")).strip().lower()
                label = mapping.get(raw) or (p.get("theme", "").strip().capitalize() or None)
                if label and label not in attr:
                    attr.append(label)
                    bucket[label].append((r, p.get("quote") or r.text))
        r.persona_hints = sorted({persona_map.get(str(p).strip().lower(), str(p).strip().capitalize())
                                  for p in (t.get("persona") or []) if str(p).strip()}) or _persona_hints(r)

    total = len(kept)
    a.pain_clusters = [c for c in _build_clusters(pains, total, pain_desc, "pain") if c["count"] >= 2]
    a.praise_clusters = [c for c in _build_clusters(praise, total, praise_desc, "praise") if c["count"] >= 2]
    _summarise(a)
    return a


def analyze(query: str, reviews: list[Review], cfg=None, progress=None) -> Analysis:
    if cfg is not None and reviews:
        try:
            return analyze_llm(query, reviews, cfg, progress=progress)
        except Exception as exc:
            a = analyze_local(query, reviews)
            a.warnings.append(f"LLM analysis failed ({type(exc).__name__}: {str(exc)[:200]}); "
                              "used the free local engine instead.")
            return a
    return analyze_local(query, reviews)
