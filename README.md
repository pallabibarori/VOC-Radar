# 📣 VOC Radar

**Type a product or company. Get what customers love, what hurts, and who your best customers are.**

VOC Radar gathers voice-of-customer feedback from free, trustworthy public sources (Reddit, Hacker News, App Store, Google Play and established review sites), clusters it by sentiment and pain point, and generates your **top 5 Ideal Customer Profiles (ICPs)**.

**[▶ Try the live demo](https://YOUR-USERNAME.github.io/voc-radar/)** · no install, runs on your own Claude account

![VOC Radar dashboard showing sentiment, key insights and top pain points for Notion](docs/dashboard.png)
<sub>Sample dashboard for Notion, built from invented example comments.</sub>

---

## What it does

| Step | What happens |
|---|---|
| 🔎 **Gather** | Searches Reddit, Hacker News, app store reviews, G2, Trustpilot, GSMArena and outlets like The Verge and CNET |
| 🧹 **Clean** | Removes duplicates and off-topic mentions (e.g. "the *notion* that…") |
| 📊 **Cluster** | Groups feedback into pain points (with severity) and praise themes, each with linked quotes |
| 💡 **Summarise** | Plain-language key insights: biggest pain, what wins people over, loudest segment, harshest source |
| 🎯 **Profile** | Generates the top 5 ICPs: who they are, jobs to be done, pains, buying triggers, objections, channels and a messaging angle |

![Top 5 ICP cards with jobs to be done, pains, triggers and messaging](docs/icps.png)

## Two ways to use it

| | 🌐 Web version (`web/`) | 🐍 Python app (`app.py`) |
|---|---|---|
| Best for | Anyone, instantly | Deeper research you can defend with numbers |
| Install | None | Python 3.10+ |
| How it searches | Claude's web search, in the visitor's own Claude chat | Directly from each source's public API |
| Volume | ~40 items per run | Hundreds of items |
| Cost | Visitor's own Claude usage | Free (optional AI key for sharper results) |

### 🌐 Web version

Open the [live demo](https://YOUR-USERNAME.github.io/voc-radar/), type a product, and click **Open live search**. Claude researches it in your own chat. Paste its reply back into the page to get the dashboard, then click **Generate ICPs**.

It's a single HTML file with no server and no build step. You can also open `web/index.html` straight from your computer.

### 🐍 Python app

![Streamlit version of VOC Radar](docs/python-app.png)

```bash
git clone https://github.com/YOUR-USERNAME/voc-radar.git
cd voc-radar
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Switch on **Use sample data** in the sidebar to try it offline.

From the command line:

```bash
python -m voc.cli "Notion" --icp --out notion        # writes notion.json + notion.md
python -m voc.cli "iPhone 18 Pro" --sources "Reddit,Review sites,Hacker News"
```

**Analysis engines.** By default it runs a free, offline engine: sentence-level sentiment, an 18-theme pain-point taxonomy and keyword clustering. For sharper clusters and richer ICPs, pick an AI engine in the sidebar: Claude, or a free option (Groq, Google Gemini, or Ollama on your own machine). Counts and percentages are always calculated in code, never estimated by the AI. See `.env.example` for settings.

## How it works

```
 product name
      │
      ▼
 ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐
 │   Gather     │ → │  Clean       │ → │  Cluster         │ → │  ICPs        │
 │ Reddit, HN,  │   │ dedupe,      │   │ sentiment, pain  │   │ top 5 from   │
 │ stores, web  │   │ relevance    │   │ points, praise   │   │ the evidence │
 └──────────────┘   └──────────────┘   └──────────────────┘   └──────────────┘
```

## Project structure

```
web/index.html        no-install web version (also deployed via GitHub Pages)
app.py                Streamlit dashboard
voc/sources/          collectors: reddit.py, hackernews.py, appstores.py, web.py
voc/collect.py        parallel collection, dedupe, relevance filter
voc/analyze.py        sentiment + clustering (local and AI engines)
voc/icp.py            top-5 ICP generation
voc/cli.py            command line
tests/                17 tests (mocked HTTP + headless UI)
docs/                 screenshots
```

Run the tests with `pytest -q`.

## Limitations

- Public feedback over-represents vocal users and recent controversies. Read shares as relative signals, not market size.
- Reddit rate-limits anonymous access. For reliable runs, add free Reddit app keys (see `.env.example`).
- G2, Capterra, Trustpilot and Amazon aren't scraped by the Python app because their terms forbid it. The web version reads them only through Claude's web search.

## License

[MIT](LICENSE)
