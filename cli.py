"""Command line:  python -m voc.cli "Notion" --icp --out notion_voc
Writes <out>.json (everything) and <out>.md (readable report)."""
from __future__ import annotations

import argparse
import json
import sys

from . import analyze, collect, icp, llm, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Voice-of-customer research from free public sources")
    ap.add_argument("query", nargs="?", help='Product or company, e.g. "Notion" or "iPhone 18 Pro"')
    ap.add_argument("--sources", default=",".join(collect.SOURCES),
                    help="Comma-separated: " + ", ".join(collect.SOURCES))
    ap.add_argument("--limit", type=int, default=150, help="Max items per source")
    ap.add_argument("--context", default="", help='Disambiguation words, e.g. "app,workspace,notes"')
    ap.add_argument("--icp", action="store_true", help="Also generate the top 5 ICPs")
    ap.add_argument("--local", action="store_true", help="Force the free local engine even if an LLM key is set")
    ap.add_argument("--demo", action="store_true", help="Use bundled sample data (no network)")
    ap.add_argument("--out", default="voc_report", help="Output file prefix")
    args = ap.parse_args(argv)

    cfg = None if args.local else llm.LLMConfig.from_env()
    if args.demo:
        query, reviews = report.load_demo()
    else:
        if not args.query:
            ap.error("query is required (or use --demo)")
        query = args.query
        res = collect.collect(query, [s.strip() for s in args.sources.split(",") if s.strip()],
                              args.limit, [c for c in args.context.split(",") if c],
                              progress=lambda n, c, e: print(f"  {n}: {c} items" + (f"  [error: {e}]" if e else ""),
                                                             file=sys.stderr))
        reviews = res.reviews
        print(f"Collected {len(reviews)} relevant items "
              f"({res.dropped_duplicates} duplicates, {res.dropped_irrelevant} off-topic removed)", file=sys.stderr)
    if not reviews:
        print("No feedback found. Try other sources, a broader name, or check your network.", file=sys.stderr)
        return 1

    a = analyze.analyze(query, reviews, cfg, progress=lambda m: print("  " + m, file=sys.stderr))
    icps = None
    if args.icp:
        icps, engine = icp.generate_icps(a, cfg)
        print(f"ICPs generated with {engine}", file=sys.stderr)

    data = a.to_dict()
    if icps:
        data["icps"] = icps
    with open(f"{args.out}.json", "w") as f:
        json.dump(data, f, indent=2, default=str)
    with open(f"{args.out}.md", "w") as f:
        f.write(report.to_markdown(a, icps))
    print(f"Wrote {args.out}.json and {args.out}.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
