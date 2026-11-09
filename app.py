"""VOC Radar: voice-of-customer research from free public sources.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from voc import analyze, collect, icp, llm, report

st.set_page_config(page_title="VOC Radar", page_icon="📣", layout="wide")

SENT_COLORS = {"positive": "#2e9e6b", "mixed": "#d9a400", "neutral": "#9aa3ad", "negative": "#d64545"}
SEV_COLORS = {"High": "#d64545", "Medium": "#e08a2b", "Low": "#9aa3ad"}

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
.quote {border-left: 3px solid #d0d5db; padding: .35rem .8rem; margin: .35rem 0; color: inherit; font-size: .93rem;}
.quote small {opacity: .65;}
.pill {display:inline-block; padding: 1px 9px; border-radius: 999px; font-size: .78rem; font-weight: 600; color: white;}
.icp-rank {font-size: .8rem; opacity: .6; letter-spacing: .06em; text-transform: uppercase;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    engine = st.selectbox("Analysis engine", ["Free local engine (no key)"] + list(llm.PRESETS),
                          help="The local engine is free and offline. An LLM gives sharper clusters "
                               "and much better ICPs. Groq, Gemini and Ollama have free options.")
    cfg = None
    if engine != "Free local engine (no key)":
        preset = llm.PRESETS[engine]
        env_key = os.getenv("ANTHROPIC_API_KEY" if preset["provider"] == "anthropic" else "LLM_API_KEY", "")
        api_key = st.text_input("API key", value=env_key, type="password",
                                help="Not needed for Ollama. Stays in this browser session only.")
        base_url = st.text_input("Base URL", value=preset["base_url"])
        model = st.text_input("Model", value=os.getenv("LLM_MODEL", preset["model"]))
        if model and base_url and (api_key or preset["provider"] == "openai"):
            cfg = llm.LLMConfig(preset["provider"], model, api_key, base_url)
        else:
            st.caption("Add the key and model to use this engine. Until then the local engine runs.")

    st.divider()
    sources = st.multiselect("Sources", list(collect.SOURCES), default=list(collect.SOURCES))
    limit = st.slider("Max items per source", 30, 400, 150, 10)
    context = st.text_input("Disambiguation words (optional)", placeholder="app, workspace, notes",
                            help="For names that are also ordinary words. Items must mention one of these "
                                 "unless they come from a dedicated community like r/Notion.")
    st.divider()
    demo = st.toggle("Use sample data (offline demo)", value=False,
                     help="Invented example comments about Notion, for trying the interface.")


# ------------------------------------------------------------------ header + search
st.title("📣 VOC Radar")
st.caption("Type a product or company. VOC Radar gathers what real customers say on Reddit, Hacker News, "
           "the App Store, Google Play and trusted review sites, then clusters it by sentiment and pain point.")

if "query" not in st.session_state:
    st.session_state.query = ""
auto_run = False
if "pending_query" in st.session_state:          # an example button was clicked
    st.session_state.query = st.session_state.pop("pending_query")
    auto_run = True

c1, c2 = st.columns([5, 1])
with c1:
    query = st.text_input("Product or company", key="query", placeholder="e.g. Notion, iPhone 18 Pro, Figma",
                          label_visibility="collapsed")
with c2:
    go_btn = st.button("Gather feedback", type="primary", use_container_width=True)

ex = st.columns([1, 1.4, 1, 5])
for col, name in zip(ex, ["Notion", "iPhone 18 Pro", "Figma"]):
    if col.button(name, use_container_width=True):
        st.session_state.pending_query = name
        st.rerun()
go_btn = go_btn or auto_run


# ------------------------------------------------------------------ run
def run(query: str) -> None:
    st.session_state.pop("icps", None)
    with st.status("Gathering voice of customer…", expanded=True) as status:
        meta = {}
        if demo:
            q, reviews = report.load_demo()
            st.write(f"Loaded {len(reviews)} sample comments (invented, for demo).")
            meta = {"per_source": {}, "errors": {}, "dropped_irrelevant": 0, "dropped_duplicates": 0}
        else:
            q = query.strip()
            res = collect.collect(q, sources, limit, [c for c in context.split(",") if c.strip()],
                                  progress=lambda n, c, e: st.write(
                                      f"{'⚠️' if e else '✅'} **{n}**: {c} items" + (f" — {e}" if e else "")))
            reviews = res.reviews
            meta = {"per_source": res.per_source, "errors": res.errors,
                    "dropped_irrelevant": res.dropped_irrelevant, "dropped_duplicates": res.dropped_duplicates}
            st.write(f"{len(reviews)} relevant items after removing {res.dropped_duplicates} duplicates "
                     f"and {res.dropped_irrelevant} off-topic mentions.")
        if not reviews:
            status.update(label="No feedback found", state="error")
            st.session_state.analysis = None
            st.session_state.meta = meta
            return
        status.update(label="Clustering by sentiment and pain point…")
        a = analyze.analyze(q, reviews, cfg, progress=st.write)
        st.session_state.analysis, st.session_state.meta = a, meta
        status.update(label=f"Done: {len(a.reviews)} feedback items analysed", state="complete", expanded=False)


if go_btn:
    if not query.strip() and not demo:
        st.warning("Type a product or company first.")
    else:
        run(query)

a: analyze.Analysis | None = st.session_state.get("analysis")
meta = st.session_state.get("meta", {})

if a is None:
    if meta.get("errors"):
        st.error("Every source failed or returned nothing:\n\n" +
                 "\n".join(f"- **{k}**: {v}" for k, v in meta["errors"].items()))
    elif "meta" in st.session_state:
        st.info("No feedback matched. Try the full product name, fewer disambiguation words, or more sources.")
    else:
        st.info("Start with a product or company above, or switch on **sample data** in the sidebar to try it offline.")
    st.stop()


# ------------------------------------------------------------------ summary
total = len(a.reviews)
sc = a.sentiment_counts
pct = lambda k: f"{sc.get(k, 0) / total:.0%}" if total else "0%"
st.subheader(f"Voice of customer: {a.query}")
if demo:
    st.caption("⚠️ Sample data: invented comments for demonstration, not real customer feedback.")
m = st.columns(5)
m[0].metric("Feedback items", total)
m[1].metric("Positive", pct("positive"))
m[2].metric("Negative", pct("negative"))
m[3].metric("Mixed", pct("mixed"))
m[4].metric("Sources", len(a.by_source))
for w in a.warnings:
    st.warning(w)

left, right = st.columns([3, 2])
with left:
    if a.pain_clusters:
        top = a.pain_clusters[:10][::-1]
        fig = go.Figure(go.Bar(
            x=[c["share"] * 100 for c in top], y=[c["label"] for c in top], orientation="h",
            marker_color=[SEV_COLORS.get(c.get("severity"), "#9aa3ad") for c in top],
            text=[f"{c['count']} · {c['share']:.0%}" for c in top], textposition="outside",
            hovertemplate="%{y}<br>%{x:.1f}% of feedback<extra></extra>"))
        fig.update_layout(title="Top pain points (% of all feedback, colour = severity)", height=380,
                          margin=dict(l=10, r=40, t=50, b=10), xaxis_title=None, xaxis_ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)
with right:
    srcs = list(a.by_source)
    fig = go.Figure()
    for s in ["positive", "mixed", "neutral", "negative"]:
        fig.add_bar(name=s, y=srcs, x=[a.by_source[x].get(s, 0) for x in srcs], orientation="h",
                    marker_color=SENT_COLORS[s])
    fig.update_layout(barmode="stack", title="Sentiment by source", height=380,
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------ detail tabs
def quote_html(q: dict) -> str:
    src = q["source"] + (f" · {q['rating']:g}★" if q.get("rating") else "")
    link = f' · <a href="{q["url"]}" target="_blank">open</a>' if q.get("url") else ""
    text = q["text"].replace("<", "&lt;")
    return f'<div class="quote">“{text}”<br><small>{src}{link}</small></div>'


def show_clusters(clusters: list[dict], kind: str) -> None:
    if not clusters:
        st.info("No clear clusters yet. More feedback (raise the per-source limit or add sources) helps.")
        return
    for c in clusters:
        sev = c.get("severity")
        badge = f'<span class="pill" style="background:{SEV_COLORS[sev]}">{sev}</span> ' if sev else ""
        with st.expander(f"**{c['label']}** — {c['count']} items · {c['share']:.0%} of feedback",
                         expanded=c is clusters[0]):
            st.markdown(badge + (c.get("description") or ""), unsafe_allow_html=True)
            st.caption("Sources: " + ", ".join(f"{k} ({v})" for k, v in c["sources"].items())
                       + f" · avg sentiment {c['avg_sentiment']:+.2f}")
            st.markdown("".join(quote_html(q) for q in c["quotes"]), unsafe_allow_html=True)


tabs = st.tabs(["Pain points", "What customers love", "Segments", "All feedback", "Sources & method"])
with tabs[0]:
    show_clusters(a.pain_clusters, "pain")
with tabs[1]:
    show_clusters(a.praise_clusters, "praise")
with tabs[2]:
    if a.personas:
        df = pd.DataFrame(a.personas).rename(columns={"segment": "Segment", "mentions": "Mentions", "share": "Share"})
        st.dataframe(df, hide_index=True, use_container_width=True,
                     column_config={"Share": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
        st.caption("Who is speaking, detected from how writers describe themselves and their use case. "
                   "These feed the ICP generator below.")
    else:
        st.info("No segments detected yet.")
with tabs[3]:
    df = pd.DataFrame([{
        "sentiment": r.sentiment, "score": r.sentiment_score, "source": r.source, "rating": r.rating,
        "pain points": ", ".join(r.pain_points), "praise": ", ".join(r.praise),
        "segments": ", ".join(r.persona_hints), "text": r.text, "context": r.context,
        "date": r.date, "url": r.url, "kind": r.kind} for r in a.reviews])
    f1, f2, f3 = st.columns(3)
    fs = f1.multiselect("Sentiment", sorted(df.sentiment.unique()))
    fsrc = f2.multiselect("Source", sorted(df.source.unique()))
    themes = sorted({p for r in a.reviews for p in r.pain_points + r.praise})
    fth = f3.multiselect("Theme", themes)
    view = df
    if fs:
        view = view[view.sentiment.isin(fs)]
    if fsrc:
        view = view[view.source.isin(fsrc)]
    if fth:
        view = view[view.apply(lambda row: any(t in row["pain points"] or t in row["praise"] for t in fth), axis=1)]
    st.dataframe(view, hide_index=True, use_container_width=True, height=420,
                 column_config={"url": st.column_config.LinkColumn("link", display_text="open"),
                                "text": st.column_config.TextColumn(width="large")})
    st.caption(f"{len(view)} of {len(df)} items")
with tabs[4]:
    st.markdown(f"**Engine:** {a.engine}")
    if meta.get("per_source"):
        st.markdown("**Collected per source:** " + ", ".join(f"{k}: {v}" for k, v in meta["per_source"].items()))
        st.markdown(f"Removed {meta.get('dropped_duplicates', 0)} duplicates and "
                    f"{meta.get('dropped_irrelevant', 0)} off-topic mentions.")
    for k, v in (meta.get("errors") or {}).items():
        st.warning(f"{k}: {v}")
    st.markdown("""
**How it works.** Each source is queried through its free public endpoint. Items are de-duplicated and
filtered so they actually mention the product. Sentiment is scored per sentence (VADER), blended with star
ratings where a store gives them. Pain points and praise are found at clause level, so "love the templates,
but it's slow" counts once on each side. With an LLM engine, the model tags every item and merges the tags
into clusters; all counts and shares are still computed in code.

**Caveats.** Public feedback over-represents vocal users and recent controversies. Treat shares as relative
signal, not market size. App store results appear only when an app with a matching name exists.""")


# ------------------------------------------------------------------ ICPs
st.divider()
h1, h2 = st.columns([4, 1])
h1.subheader("Top 5 Ideal Customer Profiles")
gen = h2.button("Generate ICPs", type="primary", use_container_width=True)
if cfg is None:
    st.caption("Using the free local engine: ICPs are built from detected segments and their own pains. "
               "Pick an LLM engine in the sidebar for richer profiles (jobs, triggers, objections, messaging).")
if gen:
    with st.spinner("Building ICPs from the clustered evidence…"):
        icps, used = icp.generate_icps(a, cfg)
        st.session_state.icps, st.session_state.icp_engine = icps, used

icps = st.session_state.get("icps")
if icps:
    st.caption(f"Engine: {st.session_state.get('icp_engine')}")
    if not icps:
        st.info("Not enough segment signal to build ICPs. Gather more feedback or use an LLM engine.")
    cols = st.columns(2)
    for i, p in enumerate(icps):
        with cols[i % 2].container(border=True):
            st.markdown(f'<div class="icp-rank">ICP #{p.get("rank", i + 1)} · {p.get("confidence", "")} confidence'
                        f' · {p.get("estimated_share_of_voice", "")}</div>', unsafe_allow_html=True)
            st.markdown(f"### {p.get('name', '')}")
            st.write(p.get("who", ""))
            if p.get("messaging_angle"):
                st.markdown(f"**Messaging angle:** {p['messaging_angle']}")
            for key, label in (("jobs_to_be_done", "Jobs to be done"), ("top_pains", "Top pains"),
                               ("what_they_love", "What they love"), ("buying_triggers", "Buying triggers"),
                               ("objections", "Objections"), ("where_to_find_them", "Where to find them")):
                vals = [v for v in (p.get(key) or []) if v]
                if vals:
                    st.markdown(f"**{label}:** " + " · ".join(map(str, vals)))
            ev = p.get("evidence") or []
            if ev:
                with st.expander("Evidence"):
                    for e in ev[:4]:
                        st.markdown(f"> {e}")
            if p.get("why_this_rank"):
                st.caption(p["why_this_rank"])


# ------------------------------------------------------------------ export
st.divider()
data = a.to_dict()
if icps:
    data["icps"] = icps
slug = "".join(ch if ch.isalnum() else "_" for ch in a.query.lower()).strip("_") or "voc"
d1, d2, d3 = st.columns(3)
d1.download_button("Download report (Markdown)", report.to_markdown(a, icps), f"{slug}_voc.md",
                   use_container_width=True)
d2.download_button("Download all data (JSON)", json.dumps(data, indent=2, default=str), f"{slug}_voc.json",
                   use_container_width=True)
d3.download_button("Download feedback (CSV)", pd.DataFrame([r.to_dict() for r in a.reviews]).to_csv(index=False),
                   f"{slug}_feedback.csv", use_container_width=True)
