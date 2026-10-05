"""Fetch and pin Hugging Face configs, and check pinned ones for drift.

    python scripts/fetch_configs.py --check [--report drift.md]
        For every snapshot in data/hf/: re-fetch config.json at the pinned
        revision (it must still hash to the stored value) and at the branch
        head (if it differs, or the head has moved, report it). Exits 1 only
        if a pinned config no longer matches its snapshot.

    python scripts/fetch_configs.py --pin org/repo [org/repo ...] [--out DIR]
        Pin a repository at its current head: write data/hf/<org>__<repo>.json
        (or DIR/<org>__<repo>.json) with the API metadata and the config. Then
        run build_models.py. The Freshness workflow does this for gated
        repositories with the HF_TOKEN secret (its `pin` input) and uploads the
        snapshots as an artifact, so the token never leaves GitHub.

Gated repositories need HF_TOKEN (the owner's token, with the licence
accepted); without it their configs are reported as unavailable, and the
site uses the transcriptions in data/transcribed/ instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HF = ROOT / "data/hf"
OMIT = ("quantization_config", "chat_template", "processor_config", "image_grid_pinpoints")


def get(url: str) -> tuple[int, bytes]:
    headers = {"User-Agent": "llm-architectures-explained/fetch_configs"}
    tok = os.environ.get("HF_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                time.sleep(10 * (attempt + 1))
                continue
            return e.code, b""
        except OSError:
            if attempt < 3:
                time.sleep(5)
                continue
            return 0, b""
    return 0, b""


def canonical(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def trim(cfg: dict) -> dict:
    def drop(x):
        if isinstance(x, dict):
            for k in OMIT:
                if k in x:
                    x[k] = "<omitted from snapshot>"
            for v in x.values():
                drop(v)
    drop(cfg)
    return cfg


def pin(repo: str, out: Path = HF) -> None:
    s, b = get(f"https://huggingface.co/api/models/{repo}")
    if s != 200:
        raise SystemExit(f"{repo}: API {s}")
    a = json.loads(b)
    files = [x["rfilename"] for x in a.get("siblings", [])]
    fname = "config.json" if "config.json" in files else ("params.json" if "params.json" in files else None)
    cfg = None
    if fname:
        s, b = get(f"https://huggingface.co/{repo}/resolve/{a['sha']}/{fname}")
        cfg = json.loads(b) if s == 200 else None
    snap = {
        "repo": repo,
        "revision": a["sha"],
        "fetched": time.strftime("%Y-%m-%d"),
        "file": fname if cfg is not None else None,
        "api": {"createdAt": a.get("createdAt"), "license": (a.get("cardData") or {}).get("license"),
                "gated": a.get("gated"), "safetensors": a.get("safetensors")},
        "config": None,
    }
    if cfg is not None:
        snap["config_canonical_sha256"] = canonical(cfg)
        snap["config"] = trim(cfg)
    (out / (repo.replace("/", "__") + ".json")).write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
    print(f"pinned {repo} @ {a['sha'][:7]}{'' if cfg is not None else ' (config unavailable)'}")


def check(report: Path | None) -> int:
    rows = []
    broken = 0
    for p in sorted(HF.glob("*.json")):
        snap = json.loads(p.read_text())
        repo, rev = snap["repo"], snap["revision"]
        s, b = get(f"https://huggingface.co/api/models/{repo}")
        head = json.loads(b).get("sha") if s == 200 else None
        status = []
        if snap.get("file") and snap.get("config_canonical_sha256"):
            s1, b1 = get(f"https://huggingface.co/{repo}/resolve/{rev}/{snap['file']}")
            if s1 == 200:
                if canonical(json.loads(b1)) != snap["config_canonical_sha256"]:
                    status.append("PINNED CONFIG CHANGED")
                    broken += 1
            else:
                status.append(f"pinned config unavailable ({s1})")
            if head and head != rev:
                s2, b2 = get(f"https://huggingface.co/{repo}/resolve/{head}/{snap['file']}")
                if s2 == 200 and canonical(json.loads(b2)) != snap["config_canonical_sha256"]:
                    status.append("head config differs: re-pin deliberately")
                else:
                    status.append("head moved, config unchanged")
        elif head is None:
            status.append(f"API {s}")
        rows.append((repo, rev[:7], (head or "?")[:7], "; ".join(status) or "ok"))
        time.sleep(0.5)
    lines = ["| repository | pinned | head | status |", "|---|---|---|---|"] + [f"| {r} | {a} | {h} | {st} |" for r, a, h, st in rows]
    drift = [r for r in rows if "differs" in r[3]]
    lines.insert(0, f"**{len(rows)} pinned configs checked; {len(drift)} with a newer config upstream; {broken} broken.**\n")
    text = "\n".join(lines) + "\n"
    if report:
        report.write_text(text)
    print(text)
    for r in drift:
        print(f"::warning title=Config drift::{r[0]}: {r[3]}")
    return 1 if broken else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--pin", nargs="+")
    ap.add_argument("--out", type=Path, default=HF, help="directory for --pin snapshots (default data/hf)")
    a = ap.parse_args()
    if a.pin:
        for r in a.pin:
            pin(r, a.out)
        return 0
    if a.check:
        return check(a.report)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
