"""Write tests/fixtures/arch_fixtures.json: every model's normalised spec and
what reference/arch_model.py computes for it. The TypeScript port
(src/lib/arch/costModel.ts) must reproduce every number exactly.

    python scripts/make_fixtures.py          # write
    python scripts/make_fixtures.py --check  # fail if the file would change
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "reference"), str(ROOT / "scripts")]
import arch_model as am  # noqa: E402
from build_models import strip  # noqa: E402

CONTEXTS = [1.0, 7.0, 128.0, 1000.0, 4096.0, 32768.0, 131072.0, 1048576.0]
BYTES = [(2.0, 2.0), (1.0, 2.0), (0.5625, 4.0)]  # (KV element bytes, state bytes)
WEIGHT_BYTES = [2.0, 1.0, 0.5]


def spec_of(m: dict) -> dict:
    s = strip(m["arch"])
    s["kind"] = m["kind"]
    return s


def compute(s: dict) -> dict:
    p = am.params(s)
    out = {"params": p.__dict__, "kv": [], "decode_flops": [], "prefill_flops": [], "decode_bytes": []}
    for c in CONTEXTS:
        for kb, sb in BYTES:
            out["kv"].append({"context": c, "kv_elem_bytes": kb, "state_elem_bytes": sb, "out": am.kv_cache(s, c, kb, sb)})
        out["decode_flops"].append({"context": c, "v": am.decode_flops(s, c)})
        out["prefill_flops"].append({"context": c, "v": am.prefill_flops(s, c)})
        for wb in WEIGHT_BYTES:
            out["decode_bytes"].append({"context": c, "weight_elem_bytes": wb, "kv_elem_bytes": 1.0, "out": am.decode_bytes(s, c, wb, 1.0)})
    return out


def build() -> str:
    cases = []
    for p in sorted((ROOT / "data/models").glob("*.yaml")):
        m = yaml.safe_load(p.read_text())
        if m["arch"] is None:
            continue
        s = spec_of(m)
        cases.append({"id": m["id"], "spec": s, "out": compute(s)})
    return json.dumps({"generator": "scripts/make_fixtures.py", "reference": "reference/arch_model.py", "cases": cases}, indent=None, separators=(",", ":")) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    path = ROOT / "tests/fixtures/arch_fixtures.json"
    text = build()
    if a.check:
        if not path.exists() or path.read_text() != text:
            print("fixtures out of date: run python scripts/make_fixtures.py")
            return 1
        print("fixtures up to date")
        return 0
    path.write_text(text)
    print(f"wrote {path.relative_to(ROOT)} ({len(text) // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
