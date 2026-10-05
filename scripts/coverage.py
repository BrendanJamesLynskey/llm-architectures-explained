"""Coverage check: every name in the gallery checklist maps to exactly one
model file, and the extra models are listed.

    python scripts/coverage.py            # print the report, exit 1 on a gap
    python scripts/coverage.py --json     # machine-readable

The checklist (data/coverage/gallery_2026-10-05.txt) holds the gallery's
model names only, one per line; nothing else is taken from the gallery.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def report() -> dict:
    names = [n.strip() for n in (ROOT / "data/coverage/gallery_2026-10-05.txt").read_text().splitlines() if n.strip()]
    models = [yaml.safe_load(p.read_text()) for p in sorted((ROOT / "data/models").glob("*.yaml"))]
    by_gallery: dict[str, list[str]] = {}
    for m in models:
        if m.get("gallery_name"):
            by_gallery.setdefault(m["gallery_name"], []).append(m["id"])
    missing = [n for n in names if n not in by_gallery]
    duplicated = {n: ids for n, ids in by_gallery.items() if len(ids) > 1}
    unknown = sorted(n for n in by_gallery if n not in names)
    extras = sorted(m["id"] for m in models if not m.get("gallery_name"))
    closed = sorted(m["id"] for m in models if not m["open_weights"])
    no_arch = sorted(m["id"] for m in models if m["arch"] is None)
    return {
        "gallery_names": len(names),
        "gallery_covered": len(names) - len(missing),
        "missing": missing,
        "duplicated": duplicated,
        "unknown_gallery_names": unknown,
        "extras": extras,
        "extras_count": len(extras),
        "models_total": len(models),
        "closed": closed,
        "without_arch": no_arch,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = report()
    if a.json:
        print(json.dumps(r, indent=1))
    else:
        print(f"gallery: {r['gallery_covered']}/{r['gallery_names']} names covered")
        print(f"extras: {r['extras_count']} models beyond the gallery: {', '.join(r['extras'])}")
        print(f"total models: {r['models_total']} ({len(r['closed'])} closed, {len(r['without_arch'])} without layer dimensions)")
        for k in ("missing", "duplicated", "unknown_gallery_names"):
            if r[k]:
                print(f"{k}: {r[k]}")
    ok = not r["missing"] and not r["duplicated"] and not r["unknown_gallery_names"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
