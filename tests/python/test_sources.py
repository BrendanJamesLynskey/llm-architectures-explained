"""Every arXiv id the data cites was checked at export.arxiv.org
(scripts/verify_arxiv.py records the title and first author)."""

import json

from helpers import ROOT, models

CHECKED = json.loads((ROOT / "data/sources/arxiv.json").read_text())["papers"]


def test_every_cited_paper_was_checked():
    for m in models():
        for s in m["sources"].values():
            if s.get("arxiv"):
                assert s["arxiv"] in CHECKED, f"{m['id']}: arXiv {s['arxiv']} not verified"


def test_estimate_sources_were_checked():
    for m in models():
        for s in m["sources"].values():
            if s["type"] == "reported-estimate":
                aid = s["url"].rsplit("/", 1)[-1]
                assert aid in CHECKED
