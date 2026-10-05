"""Every field read straight from a config.json is checked against the
pinned snapshot in data/hf/, and the generated files match the build."""

from __future__ import annotations

import re
import subprocess
import sys

import pytest

from helpers import ROOT, models, snapshot, walk_fields

SIMPLE = re.compile(r"^[A-Za-z_][\w]*(\.[\w]+)*$")


def lookup(cfg, path):
    cur = cfg
    for part in path.split("."):
        cur = cur[part]
    return cur


HF = [m for m in models() if m["sources"].get("hf", {}).get("config_available") and not any(
    s.get("transcription") for s in m["sources"].values()) and "arch_from" not in m]


@pytest.mark.parametrize("m", HF, ids=[m["id"] for m in HF])
def test_config_fields_match_snapshot(m):
    cfg = snapshot(m)["config"]
    checked = 0
    for path, f in walk_fields({"facts": m["facts"], "arch": m["arch"]}):
        # Fields derived from a key by a rule (layouts, position summaries)
        # hold a list or a dict; build_models.py --check covers those.
        if isinstance(f["v"], (dict, list)):
            continue
        if f["st"] == "config" and f.get("src") == "hf" and SIMPLE.match(f.get("ref", "")):
            assert lookup(cfg, f["ref"]) == f["v"], f"{m['id']}: {path} = {f['v']!r}, config {f['ref']} differs"
            checked += 1
    assert checked >= 3, f"{m['id']}: only {checked} config fields checked"


def test_generated_files_are_up_to_date():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/build_models.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_coverage_complete():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/coverage.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
    assert "109/109" in r.stdout


def test_snapshots_are_pinned_to_full_commits():
    for m in models():
        hf = m["sources"].get("hf")
        if hf:
            assert re.fullmatch(r"[0-9a-f]{40}", hf["revision"]), m["id"]
            assert hf["revision"] in hf["url"]
