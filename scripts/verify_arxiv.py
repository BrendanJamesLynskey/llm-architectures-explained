"""Check every arXiv identifier the data cites at export.arxiv.org and record
its title, first author and submission date in data/sources/arxiv.json.

    python scripts/verify_arxiv.py          # network; rewrites the record

tests/python/test_sources.py (offline) then checks that every cited id is in
the record, so an unchecked paper cannot slip in.
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def cited() -> list[str]:
    ids: set[str] = set()
    for p in (ROOT / "data/curation").glob("*.yaml"):
        ids.update(re.findall(r'paper: "(\d{4}\.\d{4,5})"', p.read_text()))
        ids.update(re.findall(r"arXiv (\d{4}\.\d{4,5})", p.read_text()))
    for p in (ROOT / "data/transcribed").glob("*.json"):
        ids.update(re.findall(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", p.read_text()))
        ids.update(re.findall(r"arXiv (\d{4}\.\d{4,5})", p.read_text()))
    for p in (ROOT / "data/models").glob("*.yaml"):
        m = yaml.safe_load(p.read_text())
        for s in m["sources"].values():
            if s.get("arxiv"):
                ids.add(s["arxiv"])
    return sorted(ids)


def fetch(aid: str) -> dict:
    req = urllib.request.Request(f"https://export.arxiv.org/abs/{aid}", headers={"User-Agent": "llm-architectures-explained"})
    page = urllib.request.urlopen(req, timeout=60).read().decode("utf8", "replace")
    meta = lambda k: [html.unescape(x) for x in re.findall(rf'<meta name="citation_{k}" content="([^"]*)"', page)]  # noqa: E731
    title = meta("title")
    if not title:
        raise RuntimeError(f"{aid}: no title")
    sub = re.search(r"Submitted on (\d{1,2} \w{3} \d{4})", page)
    return {"title": title[0], "first_author": (meta("author") or ["?"])[0], "submitted": sub.group(1) if sub else None}


def main() -> None:
    out = {}
    for aid in cited():
        out[aid] = fetch(aid)
        print(aid, "|", out[aid]["first_author"], "|", out[aid]["title"])
        time.sleep(1.0)
    (ROOT / "data/sources/arxiv.json").write_text(json.dumps({"checked": time.strftime("%Y-%m-%d"), "source": "export.arxiv.org", "papers": out}, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
