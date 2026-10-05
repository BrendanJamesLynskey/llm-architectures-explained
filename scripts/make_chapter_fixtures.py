"""Write the concept chapters' data and fixtures from reference/chapter_model.py.

* ``src/data/chapters.json``: what the chapter interactives start from: the
  attention variants (specs built on Llama 3 8B's sourced shape), the base
  specs the sliders rebuild, the RoPE settings of a few models (read from their
  pinned config.json snapshots), and the model lists of the context, MTP,
  looping and CED widgets.
* ``tests/fixtures/chapter_fixtures.json``: what the Python reference computes
  for them (and for grids of slider settings). ``tests/unit/chapters/*.test.ts``
  checks that the TypeScript port reproduces every value.

    python scripts/make_chapter_fixtures.py          # write
    python scripts/make_chapter_fixtures.py --check  # fail if either file would change
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
import chapter_model as cm  # noqa: E402
from build_models import strip  # noqa: E402

CONTEXTS = [1024.0, 8192.0, 32768.0, 131072.0, 1048576.0]
BF16 = 2.0


def load(mid: str) -> dict:
    return yaml.safe_load((ROOT / "data/models" / f"{mid}.yaml").read_text())


def spec_of(m: dict) -> dict:
    s = strip(m["arch"])
    s["kind"] = m["kind"]
    return s


def costs(s: dict) -> dict:
    p = am.params(s)
    return {
        "params": p.__dict__,
        "kv": [{"context": c, "out": am.kv_cache(s, c, BF16, BF16)} for c in CONTEXTS],
        "decode_flops": [{"context": c, "v": am.decode_flops(s, c)} for c in CONTEXTS],
        "prefill_flops": [{"context": c, "v": am.prefill_flops(s, c)} for c in CONTEXTS],
        "decode_bytes": [{"context": c, "out": am.decode_bytes(s, c, BF16, BF16)} for c in CONTEXTS],
    }


def rope_entry(mid: str) -> dict:
    m = load(mid)
    s = spec_of(m)
    src = m["sources"]["hf"]
    snap = json.loads((ROOT / src["snapshot"]).read_text())["config"]
    cfg = snap.get("text_config", snap)
    theta = float(cfg["rope_theta"])
    mixer = next(iter(s["mixers"].values()))
    pos = strip(m["facts"]["position"])
    if mixer["type"] == "mla":
        head, fraction, what = int(mixer["qk_rope"]), 1.0, "qk_rope_head_dim (decoupled rotary key)"
    else:
        head, fraction, what = int(mixer["head_dim"]), float(pos["rope_fraction"]), "head_dim"
    return {"id": mid, "name": m["name"], "theta": theta, "head_dim": head, "fraction": fraction,
            "rotated": what, "config": src["url"]}


ROPE_MODELS = ["mistral-7b", "gpt-oss-120b", "olmo-2-7b", "qwen3-8b", "glm-4.5", "qwen3-next-80b-a3b", "deepseek-v3"]
CONTEXT_MODELS = ["llama-3.1-405b", "gemma-3-27b", "deepseek-v3", "deepseek-v3.2", "qwen3-next-80b-a3b",
                  "minimax-text-01", "deepseek-v4-flash"]
MOE_GRID = [(1, 1, 0, 1, 0), (8, 2, 0, 1, 0), (64, 6, 2, 8, 1), (128, 8, 0, 8, 0), (256, 8, 1, 16, 3), (16, 4, 1, 4, 2)]
DEPTH_WIDTH_GRID = [(80, 2048), (40, 2880), (20, 4096), (32, 4096), (12, 8192), (126, 16384), (30, 576)]
NORM_GRID = [(p, layers, g) for p in cm.PLACEMENTS for layers in (6, 32, 80) for g in (0.5, 1.0, 2.0)]
MTP_ACCEPT = [0.6, 0.85, 0.9]


def build() -> tuple[str, str]:
    base_id = "llama-3-8b"
    base = spec_of(load(base_id))
    variants = cm.attention_variants(base)

    mtp_ids = sorted(p.stem for p in (ROOT / "data/models").glob("*.yaml")
                     if (load(p.stem).get("arch") or {}) and strip(load(p.stem)["arch"]).get("mtp_layers"))
    loop_id, ced_id, t5_id = "ouro-2.6b", "deepseek-v4.1-flash", "t5-11b"
    ced_spec, t5_spec = spec_of(load(ced_id)), spec_of(load(t5_id))

    data = {
        "generator": "scripts/make_chapter_fixtures.py",
        "attention": {"base": base_id, "variants": variants},
        "moe": {"base": base_id, "spec": base},
        "depth_width": {"vocab": 128256},
        "rope": [rope_entry(m) for m in ROPE_MODELS],
        "context": CONTEXT_MODELS,
        "mtp": mtp_ids,
        "loops": {"id": loop_id, "spec": spec_of(load(loop_id))},
        "ced": {"id": ced_id, "spec": ced_spec, "t5": t5_id, "t5_spec": t5_spec},
    }

    fx: dict = {"generator": "scripts/make_chapter_fixtures.py", "reference": "reference/chapter_model.py"}
    fx["attention"] = [{"id": v["id"], "out": costs(v["spec"])} for v in variants]
    fx["moe"] = [{"args": list(a), "spec": cm.moe_spec(base, *a), "params": am.params(cm.moe_spec(base, *a)).__dict__,
                  "decode_bytes": am.decode_bytes(cm.moe_spec(base, *a), 8192.0, BF16, BF16)}
                 for a in MOE_GRID]
    fx["depth_width"] = [{"args": list(a), "spec": cm.depth_width_spec(*a), "out": costs(cm.depth_width_spec(*a))}
                         for a in DEPTH_WIDTH_GRID]
    # pow() may differ in the last bit between libms (CI's glibc vs a laptop's), so the stored
    # wavelengths are rounded to 15 significant digits; the port compares them to 1e-12
    fx["rope"] = [{"id": r["id"], "wavelengths": [float(f"{w:.15g}") for w in
                                                  cm.rope_wavelengths(r["head_dim"], r["fraction"], r["theta"])],
                   "long_pairs": [{"context": c, "v": cm.rope_long_pairs(r["head_dim"], r["fraction"], r["theta"], c)}
                                  for c in CONTEXTS]}
                  for r in data["rope"]]
    fx["rotary_dims"] = [{"args": [h, f], "v": cm.rotary_dims(h, f)} for h in (64, 80, 128, 256) for f in (1.0, 0.5, 0.25, 0.3)]
    fx["norms"] = [{"args": [p, n, g], "out": cm.norm_profile(p, n, g)} for p, n, g in NORM_GRID]
    fx["norms_gamma"] = cm.norm_profile("output-norm", 32, 3.0, 0.5)
    fx["mtp"] = []
    for mid in mtp_ids:
        s = spec_of(load(mid))
        fx["mtp"].append({"id": mid, "module_active": cm.mtp_module_active(s), "mtp_params": am.params(s).mtp,
                          "speedup": [{"depth": d, "acceptance": a, "out": cm.mtp_speedup(s, 8192.0, d, a, BF16, BF16)}
                                      for d in (1, 2, 3) for a in MTP_ACCEPT]})
    fx["mtp_tokens"] = [{"args": [a, d], "v": cm.mtp_expected_tokens(a, d)} for a in (0.0, 0.5, 0.85, 0.9, 1.0)
                        for d in (0, 1, 2, 4)]
    loop_base = data["loops"]["spec"]
    fx["loops"] = [{"loops": k, "spec": cm.looped_spec(loop_base, k), "out": costs(cm.looped_spec(loop_base, k)),
                    "kv_per_loop": [cm.looped_kv_bytes(cm.looped_spec(loop_base, k), c, BF16, True) for c in CONTEXTS],
                    "kv_shared": [cm.looped_kv_bytes(cm.looped_spec(loop_base, k), c, BF16, False) for c in CONTEXTS]}
                   for k in (1, 2, 3, 4)]
    dec = cm.as_decoder(ced_spec)
    fx["ced"] = {"as_decoder": dec, "ced": costs(ced_spec), "decoder": costs(dec), "t5": costs(t5_spec)}
    dump = lambda x: json.dumps(x, separators=(",", ":")) + "\n"  # noqa: E731
    return dump(data), dump(fx)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    data, fx = build()
    targets = [(ROOT / "src/data/chapters.json", data), (ROOT / "tests/fixtures/chapter_fixtures.json", fx)]
    if a.check:
        stale = [p for p, text in targets if not p.exists() or p.read_text() != text]  # noqa: E501
        for p, text in targets:
            if p in stale:
                old = p.read_text() if p.exists() else ""
                i = next((k for k, (x, y) in enumerate(zip(old, text)) if x != y), min(len(old), len(text)))
                print(f"{p.relative_to(ROOT)} out of date: run python scripts/make_chapter_fixtures.py")
                print(f"  first difference at {i}: committed {old[max(0, i - 60):i + 60]!r}")
                print(f"  {'':22}rebuilt   {text[max(0, i - 60):i + 60]!r}")
        if not stale:
            print("chapter data and fixtures up to date")
        return 1 if stale else 0
    for p, text in targets:
        p.write_text(text)
        print(f"wrote {p.relative_to(ROOT)} ({len(text) // 1024} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
