"""Collectors (with mocked HTTP), relevance filter, both analysis engines and ICPs."""
import json

import pytest

from voc import analyze, collect, icp, llm, report
from voc.models import Review
from voc.sources import appstores, hackernews, reddit


# ------------------------------------------------------------- fake HTTP payloads
HN = {"nbPages": 1, "hits": [
    {"objectID": "1", "author": "a", "created_at": "2026-05-01T00:00:00Z", "story_title": "Ask HN: notes apps",
     "comment_text": "I moved our team to Notion but the performance on big databases is really bad and slow."},
    {"objectID": "2", "author": "b", "created_at": "2026-05-02T00:00:00Z", "story_title": "Philosophy",
     "comment_text": "I reject the notion that markets are always efficient, it is a simplification at best."},
]}
REDDIT_SEARCH = {"data": {"children": [{"kind": "t3", "data": {
    "id": "abc", "title": "Notion is getting slow", "selftext": "Anyone else seeing slow load times on large pages?",
    "permalink": "/r/Notion/comments/abc/", "author": "u1", "created_utc": 1767225600, "score": 50,
    "subreddit": "Notion", "num_comments": 3}}]}}
REDDIT_COMMENTS = [REDDIT_SEARCH, {"data": {"children": [
    {"kind": "t1", "data": {"body": "Yes, it takes forever to open my wiki pages now, pretty frustrating honestly.",
                            "permalink": "/r/Notion/comments/abc/x/", "author": "u2", "created_utc": 1767225600,
                            "score": 10, "replies": {"data": {"children": [
                                {"kind": "t1", "data": {"body": "Same here, the mobile app is the worst part of it.",
                                                        "author": "u3", "score": 4, "replies": ""}}]}}}},
    {"kind": "more", "data": {}}]}}]
ITUNES_SEARCH = {"results": [{"trackId": 123, "trackName": "Notion: Notes, Tasks, AI",
                              "trackViewUrl": "https://apps.apple.com/app/id123"}]}
ITUNES_FEED = {"feed": {"entry": [
    {"author": {"name": {"label": "x"}}, "im:rating": {"label": "2"}, "title": {"label": "Too slow"},
     "content": {"label": "Love the idea but it is too slow on my phone"}, "updated": {"label": "2026-04-01T00:00:00"}},
]}}


def fake_get_json(url, params=None, **_):
    if "hn.algolia.com" in url:
        return HN if params.get("tags") == "comment" else {"hits": []}
    if "itunes.apple.com/search" in url:
        return ITUNES_SEARCH
    if "customerreviews/page=1/" in url:
        return ITUNES_FEED
    if "customerreviews" in url:
        return {"feed": {}}
    if "/comments/" in url:
        return REDDIT_COMMENTS
    if "reddit.com" in url:
        return REDDIT_SEARCH
    raise AssertionError(url)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr("voc.http.get_json", fake_get_json)
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)


def test_hackernews_parses_comments():
    rs = hackernews.collect("Notion")
    assert len(rs) == 2 and rs[0].url.endswith("id=1")


def test_reddit_posts_and_nested_comments():
    rs = reddit.collect("Notion")
    texts = " ".join(r.text for r in rs)
    assert "takes forever" in texts and "mobile app is the worst" in texts
    assert all(r.source == "Reddit" for r in rs)


def test_app_store_matches_name_and_reads_rating():
    rs = appstores.collect_app_store("Notion")
    assert len(rs) == 1 and rs[0].rating == 2.0
    assert appstores.collect_app_store("iPhone 18 Pro") == []        # no app called that


def test_name_matching():
    assert appstores.name_matches("Notion", "Notion: Notes, Tasks, AI")
    assert not appstores.name_matches("Notion", "Notes Pro: Notion templates for GoodNotes planner")
    assert not appstores.name_matches("iPhone 18 Pro", "Phone Tracker")


def test_collect_dedupes_and_drops_generic_word_use():
    res = collect.collect("Notion", ["Hacker News", "Reddit", "App Store"])
    texts = [r.text for r in res.reviews]
    assert not any("reject the notion" in t for t in texts)         # dictionary use of the word
    assert res.dropped_irrelevant >= 1
    assert res.per_source["Hacker News"] == 2 and not res.errors


def test_collect_survives_a_failing_source(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("blocked")
    monkeypatch.setitem(collect.SOURCES, "Reddit", boom)
    res = collect.collect("Notion", ["Reddit", "Hacker News"])
    assert "Reddit" in res.errors and res.reviews


def test_multiword_product_relevance():
    yes = Review(source="Reddit", text="The iPhone 18 Pro battery is incredible, two days easily.")
    no = Review(source="Reddit", text="My iPhone 15 battery is dying after two years.")
    assert collect.is_relevant(yes, "iPhone 18 Pro") and not collect.is_relevant(no, "iPhone 18 Pro")


# ------------------------------------------------------------- local engine
def test_local_engine_on_sample():
    q, rs = report.load_demo()
    a = analyze.analyze_local(q, rs)
    labels = [c["label"] for c in a.pain_clusters]
    assert labels[0] == "Performance & speed"
    assert {"Sync & offline", "Pricing & value", "Customer support"} <= set(labels)
    assert sum(a.sentiment_counts.values()) == len(rs)
    for c in a.pain_clusters:                                   # shares are real arithmetic
        assert c["share"] == round(c["count"] / len(rs), 3)
        assert c["quotes"]


def test_contrast_clause_counts_on_both_sides():
    r = Review(source="x", text="I love the templates, but the app is painfully slow on large pages.")
    a = analyze.analyze_local("x", [r, Review(source="x", text="Templates are great and beautiful.")])
    assert "Templates" in r.praise and "Performance & speed" in r.pain_points
    assert a.reviews


def test_star_rating_dominates_sentiment():
    r = Review(source="App Store", text="Great idea. Works.", rating=1)
    analyze.score_review(r)
    assert r.sentiment in ("negative", "mixed")


# ------------------------------------------------------------- LLM engine (fake model)
def fake_llm_factory():
    def fake_complete(cfg, prompt, max_tokens=8000):
        if "For EACH item" in prompt:
            ids = [line.split("id=")[1].split(" ")[0] for line in prompt.splitlines() if line.startswith("- id=")]
            out = []
            for i, rid in enumerate(ids):
                out.append({"id": rid, "relevant": i != 0, "sentiment": "negative",
                            "pain_points": [{"theme": "slow large databases" if i % 2 else "slow page loads",
                                             "quote": "slow"}],
                            "praise": [], "persona": ["startup team"] if i % 3 == 0 else []})
            return "```json\n" + json.dumps(out) + "\n```"
        if "consolidating" in prompt:
            return json.dumps({"pain_points": [{"label": "Slow at scale", "description": "Gets slow.",
                                                "members": ["slow large databases", "slow page loads"]}],
                               "praise": [], "personas": [{"label": "Startup teams", "members": ["startup team"]}]})
        if "Ideal Customer Profiles" in prompt:
            assert "Slow at scale" in prompt
            return json.dumps({"icps": [{"rank": n, "name": f"ICP {n}"} for n in range(1, 6)]})
        raise AssertionError(prompt[:80])
    return fake_complete


def test_llm_engine_counts_in_code(monkeypatch):
    monkeypatch.setattr(llm, "complete", fake_llm_factory())
    q, rs = report.load_demo()
    cfg = llm.LLMConfig("openai", "fake", "", "http://x")
    a = analyze.analyze(q, rs, cfg)
    assert a.engine == "llm:fake"
    assert len(a.reviews) == len(rs) - 1                         # one marked irrelevant
    c = a.pain_clusters[0]
    assert c["label"] == "Slow at scale" and c["count"] == len(a.reviews)
    assert a.personas[0]["segment"] == "Startup teams"
    icps, engine = icp.generate_icps(a, cfg)
    assert len(icps) == 5 and engine == "llm:fake"


def test_llm_failure_falls_back_to_local(monkeypatch):
    def broken(*a, **k):
        raise ConnectionError("no route")
    monkeypatch.setattr(llm, "complete", broken)
    q, rs = report.load_demo()
    a = analyze.analyze(q, rs, llm.LLMConfig("anthropic", "m", "k", "https://api.anthropic.com"))
    assert a.engine == "local" and "failed" in a.warnings[0]


def test_local_icps_and_markdown():
    q, rs = report.load_demo()
    a = analyze.analyze_local(q, rs)
    icps, engine = icp.generate_icps(a)
    assert engine == "local" and len(icps) == 5
    md = report.to_markdown(a, icps)
    assert "## Pain points" in md and "## Top 5 ICPs" in md


def test_parse_json_variants():
    assert llm.parse_json('Sure! {"a": 1} hope that helps') == {"a": 1}
    assert llm.parse_json("```json\n[1,2]\n```") == [1, 2]


def test_gsmarena_user_opinions_parser():
    from voc.sources import web
    html = """<html><body><div class="user-thread"><ul><li class="uname2">Anon-1</li>
    <li class="upost"><time>12 Oct 2026</time></li></ul>
    <p class="uopin">Battery life on the 18 Pro is amazing but it gets hot while charging.</p></div></body></html>"""
    rs = web._gsmarena_opinions("https://www.gsmarena.com/apple_iphone_18_pro-reviews-1.php", html, "iPhone 18 Pro")
    assert len(rs) == 1 and "gets hot" in rs[0].text and rs[0].source == "GSMArena"
