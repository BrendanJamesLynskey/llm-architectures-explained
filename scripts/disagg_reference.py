"""Write the CED simulation fixtures from Disaggregated_Inference_Sim at the vendored commit.

The encoder-decoder chapter runs the simulator's own JavaScript engine
(`src/lib/disagg/vendor/sim_engine.js`, vendored byte for byte by
`pnpm vendor:sim`) on the Causal Encoder-Decoder (CED) option that brief 11
added. This script runs the Python package at the same commit and writes:

* `tests/unit/fixtures/disagg_ced_parity.json`: engine parity. The ten
  configurations of the simulator's own `test_javascript_port_matches_python_for_ced`
  (replay on prefill and on decode, two-GPU prefill pools, an 8B variant with
  W = 32, colocated, KV compression, a power cap), each with its workload,
  every request's seven timestamps (prefill_done included), every instance's
  energy counters and the link's.
* `tests/unit/fixtures/disagg_ced_results.json`: what results.md sections 16-18
  record, and the analytic ceilings the capacity search starts from. Every
  value, formatted the way `examples/results.py` formats it, is checked to
  appear in its results.md line; the capacity rates are the full-precision
  values of `examples/results_ced.json`, which the TypeScript search must
  reproduce exactly.
* `public/disagg/ced-results.json`: the same recorded values, slimmed for the
  browser (no configurations or ceilings), which the chapter's interactive
  compares its live runs with.
* `public/disagg/workloads/ced-<prompt>-<output>.json`: the four workloads of
  sections 17-18 (1,000 requests, seed 1, cv 0.5), recorded as each request's
  unit-rate exponential draw `-log(1 - u)` plus its lengths. Python's
  `expovariate(rate)` is exactly `-log(1 - u) / rate`, so the browser rebuilds
  the arrivals at *any* rate bit for bit (`t += e / rate`); this script checks
  that for every rate the search recorded. (A JavaScript port of the random
  number generator would not do: V8's Math.log differs from glibc's in the
  last bit for some inputs.)

Run with the simulator's virtualenv, the simulator checked out at the commit
recorded in `src/lib/disagg/vendor/VENDORED.json`:

    ../Disaggregated_Inference_Sim/.venv/bin/python scripts/disagg_reference.py [path/to/sim]
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "Disaggregated_Inference_Sim"
sys.path.insert(0, str(SIM / "src"))

from disagg_sim.hardware import A100_SXM, H100_SXM, KV_PRESETS, LINKS, MODELS, CostModel, KVTransit  # noqa: E402
from disagg_sim.metrics import summarise  # noqa: E402
from disagg_sim.search import Workload, analytic_capacity  # noqa: E402
from disagg_sim.sim import SimConfig, simulate  # noqa: E402
from disagg_sim.workload import LengthDist, poisson_workload  # noqa: E402

VENDORED = json.loads((ROOT / "src/lib/disagg/vendor/VENDORED.json").read_text())
FIXTURES = ROOT / "tests/unit/fixtures"
WORKLOADS = ROOT / "public/disagg/workloads"

# examples/results.py, sections 16-18
CED_WORKLOADS = [(2048, 512), (4096, 256), (8192, 128), (16384, 64)]
CED_CLUSTER, CED_N, CED_SEED = 6, 1000, 1
CED_SMALL_PREFILL = [(2, 5), (4, 4), (6, 3), (8, 2)]
VARIANTS = {"decoder-only": dict(model="llama3-70b"),
            "CED, replay on prefill": dict(model="llama3-70b-ced"),
            "CED, replay on decode": dict(model="llama3-70b-ced", cedReplayOn="decode")}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(SIM), *args], capture_output=True, text=True, check=True).stdout.strip()


def py_config(c: dict) -> SimConfig:
    """The Python SimConfig a JS config means: the mapping of the simulator's tests/test_ced.py, plus
    pool sizes, the TTFT SLO and the per-pool device counts."""
    model = MODELS[c["model"]]
    kw = {k2: c[k1] for k1, k2 in (("cedEncoderLayers", "ced_encoder_layers"), ("cedReplay", "ced_replay"),
                                    ("cedReplayOn", "ced_replay_on"), ("lmHead", "prefill_lm_head")) if k1 in c}
    if kw:
        model = replace(model, **kw)
    tr = None
    if "kvCompress" in c:
        tr = KVTransit(KV_PRESETS[c["kvCompress"]], where=c.get("kvCompressAt", "transit"))
    dev = {"h100": H100_SXM, "a100": A100_SXM}
    return SimConfig(model=model, device=H100_SXM, devices_per_instance=c.get("devicesPerInstance", 4),
                     mode=c.get("mode", "disagg"), n_prefill=c.get("nPrefill", 1), n_decode=c.get("nDecode", 1),
                     n_colocated=2, link=LINKS[c.get("link", "ib-ndr")], power_cap_w=c.get("powerCap"),
                     dvfs=c.get("dvfs", False), prefill_devices_per_instance=c.get("prefillDevicesPerInstance"),
                     decode_device=dev.get(c.get("decodeDevice")), kv_transit=tr,
                     ttft_slo=c.get("ttftSlo", 1.0), tpot_slo=c.get("tpotSlo", 0.025))


def js_config(c: dict) -> dict:
    out = {"device": "h100", "devicesPerInstance": 4, "mode": "disagg", "nPrefill": 1, "nDecode": 1,
           "nColocated": 2, "link": "ib-ndr", "ttftSlo": 1.0, "tpotSlo": 0.025}
    out.update(c)
    return out


# ───────────────────────────────────────────────────────────── 1. parity ──
# The simulator's own JS-parity cases (tests/test_ced.py JS_CASES), on its workload.
PARITY = {
    "replay on prefill": dict(model="llama3-70b-ced"),
    "replay on decode": dict(model="llama3-70b-ced", cedReplayOn="decode"),
    "replay on decode, 2-GPU prefill pool, 2P2D": dict(model="llama3-70b-ced", cedReplayOn="decode",
                                                       prefillDevicesPerInstance=2, nPrefill=2, nDecode=2),
    "8B, W=32, LM head last": dict(model="llama3-8b-ced", devicesPerInstance=1, cedReplay=32, lmHead="last"),
    "8B, encoder 8 layers, replay on decode, A100 decode": dict(model="llama3-8b", cedEncoderLayers=8,
                                                               cedReplayOn="decode", devicesPerInstance=1,
                                                               decodeDevice="a100"),
    "colocated, replay on prefill": dict(model="llama3-70b-ced", mode="colocated"),
    "replay on decode, fp8 at the GPU, 25 GbE": dict(model="llama3-70b-ced", cedReplayOn="decode", link="eth-25g",
                                                     kvCompress="fp8", kvCompressAt="endpoint"),
    "replay on decode, fp4 in transit": dict(model="llama3-70b-ced", cedReplayOn="decode", link="eth-25g",
                                             kvCompress="fp4-block"),
    "replay on decode, power cap + DVFS": dict(model="llama3-70b-ced", cedReplayOn="decode", powerCap=350.0, dvfs=True),
    "decoder-only baseline (unchanged path)": dict(model="llama3-70b"),
}


def parity() -> list[dict]:
    cases = []
    for name, c in PARITY.items():
        reqs = poisson_workload(2.5, 150, LengthDist(4096, 0.3), LengthDist(48, 1.0), seed=17)
        for r in reqs[::7]:
            r.output_len = 1
        rows = [[r.arrival, r.prompt_len, r.output_len] for r in reqs]
        res = simulate(py_config(c), reqs)
        m = summarise(res)
        cases.append({
            "name": name, "cfg": js_config(c), "rows": rows,
            "stamps": [[r.prefill_start, r.first_token, r.prefill_done, r.kv_start, r.kv_ready, r.decode_start,
                        r.finish] for r in reqs],
            "inst": [[i.compute_j, i.memory_j, i.busy, i.peak_power, i.steps, i.batch_sum] for i in res.instances],
            "link": [res.link.energy, res.link.bytes, res.link.wait, res.link.transit_j],
            "total": m["energy"]["total_J"],
            "slo": m["throughput"]["slo_attainment"],
            "capped": any(i.power_bound_time > 0 for i in res.instances),
        })
    return cases


# ─────────────────────────────────────────────────────────── 2. workloads ──
def unit_draws(p: int, o: int) -> list[list]:
    """[e, prompt, output] per request: e = -log(1 - u), the draw expovariate divides by the rate."""
    rng, pd, od = random.Random(CED_SEED), LengthDist(p, 0.5), LengthDist(o, 0.5)
    rows = []
    for _ in range(CED_N):
        e = -math.log(1.0 - rng.random())
        rows.append([e, pd.sample(rng), od.sample(rng)])
    return rows


def check_rebuild(rows: list[list], p: int, o: int, rate: float) -> None:
    """The browser's rebuild (t += e / rate) must equal poisson_workload at this rate, bit for bit."""
    wl = poisson_workload(rate, CED_N, LengthDist(p, 0.5), LengthDist(o, 0.5), seed=CED_SEED)
    t = 0.0
    for (e, pl, ol), r in zip(rows, wl):
        t += e / rate
        if (t, pl, ol) != (r.arrival, r.prompt_len, r.output_len):
            raise SystemExit(f"workload {p}:{o} at rate {rate!r}: rebuild differs at request {r.rid}")


# ───────────────────────────────────────────────────── 3. results.md 16-18 ──
def results_line(md: list[str], section: str, prefix: str, after: str | None = None) -> str:
    start = next(i for i, l in enumerate(md) if l.startswith(f"## {section}. "))
    end = next((i for i in range(start + 1, len(md)) if md[i].startswith("## ")), len(md))
    if after is not None:
        start = next(i for i in range(start, end) if md[i].startswith(after))
    hits = [l for l in md[start:end] if l.startswith(prefix)]
    if not hits:
        raise SystemExit(f"results.md §{section}: no line starts with {prefix!r}")
    return hits[0]


def need(text: str, line: str, what: str) -> dict:
    if text not in line:
        raise SystemExit(f"{what}: {text!r} not in results.md line {line!r}")
    return {"text": text}


def ms(x):  # examples/results.py ms()
    return f"{1e3 * x:,.1f} ms" if x < 1 else f"{1e3 * x:,.0f} ms"


def section16(md: list[str], ced: dict) -> dict:
    pa = ced["params"]
    b = lambda x: f"{x / 1e9:.2f}B"
    lines = {
        "decoder_only_token": results_line(md, "16", "| Decoder-only, any token"),
        "ced_prompt_token": results_line(md, "16", "| CED, prompt token"),
        "ced_decode_token": results_line(md, "16", "| CED, decode token"),
        "ced_kv_proj": results_line(md, "16", "| ... of which the decoder's K/V projections"),
    }
    params = {k: {"v": pa[k], "line": lines[k], **need(b(pa[k]), lines[k], k)} for k in lines}
    steps = {}
    for s, row in ced["steps"].items():
        line = results_line(md, "16", f"| {int(s):,} |")
        cells = {}
        for k in ("base", "ced", "enc"):
            cells[k] = {"flops": row[k]["flops"], "time": row[k]["time"], "bound": row[k]["bound"],
                        "pflop": need(f"{row[k]['flops'] / 1e15:.3f}", line, f"{s} {k} flops")["text"],
                        "ms": need(ms(row[k]["time"]), line, f"{s} {k} time")["text"]}
        for k in ("ced", "enc"):
            cells[k]["ratio"] = need(f"{row[k]['time'] / row['base']['time']:.3f}", line, f"{s} {k} ratio")["text"]
        steps[s] = {"line": line, "cells": cells}
    room = {}
    for n, row in ced["room"].items():
        line = results_line(md, "16", f"| {n}x H100 |")
        room[n] = {"line": line, **row}
        for k in ("base", "enc"):
            need(f"{row[k]['weights'] / 1e9:.1f} GB", line, f"room {n} {k}")
            need("does not fit" if row[k]["kv_tokens"] is None else f"{row[k]['kv_tokens']:,}", line, f"room {n} {k}")
    return {"params": params, "steps": steps, "room": room}


def section17(md: list[str], ced: dict, draws: dict) -> list[dict]:
    out = []
    for p, o in CED_WORKLOADS:
        key = f"{p}:{o}"
        sp = ced["split"][key]
        slo = sp["ttft_slo"]
        base_slo = 5.0 * CostModel(MODELS["llama3-70b"], H100_SXM, 4).prefill([p]).time
        assert slo == base_slo, (key, slo, base_slo)
        rows = {}
        for name, c in VARIANTS.items():
            if name == "decoder-only":
                line = results_line(md, "17", f"| {p} : {o} | {ms(slo)} | decoder-only |")
            else:
                line = results_line(md, "17", f"| {p} : {o} |  | {name} |")
            cells = {}
            for np_ in range(1, CED_CLUSTER):
                rate = sp["rates"][name][str(np_)]
                check_rebuild(draws[(p, o)], p, o, rate)
                cfg = js_config({**c, "nPrefill": np_, "nDecode": CED_CLUSTER - np_, "ttftSlo": slo})
                wl = Workload(LengthDist(p, 0.5), LengthDist(o, 0.5), n=CED_N, seed=CED_SEED)
                ceiling = analytic_capacity(py_config(cfg), wl)["ceiling"]
                cells[str(np_)] = {"rate": rate, "text": need(f"{rate:.2f}", line, f"{key} {name} {np_}")["text"],
                                   "ceiling": ceiling, "cfg": cfg}
            best = max(range(1, CED_CLUSTER), key=lambda k: sp["rates"][name][str(k)])
            need(f"| {best}P{CED_CLUSTER - best}D | {sp['rates'][name][str(best)]:.2f} |", line, f"{key} {name} best")
            rows[name] = {"line": line, "cells": cells, "best": best}
        out.append({"workload": key, "prompt": p, "output": o, "ttft_slo": slo, "slo_text": ms(slo), "rows": rows})
    return out


def section18(md: list[str], ced: dict, draws: dict) -> dict:
    sm = ced["small_prefill"]
    p, o = (int(x) for x in sm["workload"].split(":"))
    slo = sm["ttft_slo"]
    rows = {}
    for name, c in VARIANTS.items():
        line = results_line(md, "18", f"| {name} |")
        cells = {}
        for np_, nd in CED_SMALL_PREFILL:
            k = f"{np_}x2+{nd}x4"
            rate = sm["rates"][name][k]
            check_rebuild(draws[(p, o)], p, o, rate)
            cfg = js_config({**c, "nPrefill": np_, "nDecode": nd, "prefillDevicesPerInstance": 2, "ttftSlo": slo})
            wl = Workload(LengthDist(p, 0.5), LengthDist(o, 0.5), n=CED_N, seed=CED_SEED)
            cells[k] = {"rate": rate, "text": need(f"{rate:.2f}", line, f"18 {name} {k}")["text"],
                        "ceiling": analytic_capacity(py_config(cfg), wl)["ceiling"], "cfg": cfg}
        rows[name] = {"line": line, "cells": cells}
    return {"workload": sm["workload"], "prompt": p, "output": o, "ttft_slo": slo, "rows": rows,
            "router_note": results_line(md, "18", "Prefill instances on 2 H100s")}


def main() -> None:
    commit = git("rev-parse", "HEAD")
    if commit != VENDORED["commit"]:
        raise SystemExit(f"simulator is at {commit[:7]}, the vendored engine at {VENDORED['commit'][:7]}:"
                         f" check out {VENDORED['commit'][:7]} (or re-vendor) first")
    if git("status", "--porcelain", "--", "src", "web", "examples/results.md", "examples/results_ced.json"):
        raise SystemExit("simulator checkout has local changes in src/, web/ or examples/")
    header = {"simulator": VENDORED["repository"], "commit": commit}

    WORKLOADS.mkdir(parents=True, exist_ok=True)
    draws = {}
    for p, o in CED_WORKLOADS:
        draws[(p, o)] = unit_draws(p, o)
        (WORKLOADS / f"ced-{p}-{o}.json").write_text(json.dumps(
            {**header, "n": CED_N, "prompt": [p, 0.5], "output": [o, 0.5], "seed": CED_SEED,
             "columns": ["unit exponential -log(1-u)", "prompt tokens", "output tokens"],
             "rows": draws[(p, o)]}, separators=(",", ":")) + "\n")

    (FIXTURES / "disagg_ced_parity.json").write_text(json.dumps({**header, "cases": parity()}) + "\n")
    md = (SIM / "examples/results.md").read_text().splitlines()
    ced = json.loads((SIM / "examples/results_ced.json").read_text())
    out = {**header, "s16": section16(md, ced), "s17": section17(md, ced, draws), "s18": section18(md, ced, draws),
           "quotes": {"s16_intro": results_line(md, "16", "DeepSeek-V4.1-Flash (arXiv:2609.19969"),
                      "s16_paper": results_line(md, "16", "The paper's own figures"),
                      "s17_intro": results_line(md, "17", "Highest Poisson rate")}}
    (FIXTURES / "disagg_ced_results.json").write_text(json.dumps(out, indent=1) + "\n")
    slim = lambda rows: {k: {"line": r["line"], **({"best": r["best"]} if "best" in r else {}),
                             "cells": {c: {"rate": v["rate"], "text": v["text"]} for c, v in r["cells"].items()}}
                         for k, r in rows.items()}
    browser = {**header,
               "s16": {"steps": {s_: {k: {kk: c[kk] for kk in ("flops", "time", "pflop", "ms")}
                                      for k, c in row["cells"].items()} for s_, row in out["s16"]["steps"].items()}},
               "s17": [{k: w[k] for k in ("workload", "prompt", "output", "ttft_slo", "slo_text")} | {"rows": slim(w["rows"])}
                       for w in out["s17"]],
               "s18": {k: out["s18"][k] for k in ("workload", "prompt", "output", "ttft_slo")} | {"rows": slim(out["s18"]["rows"])}}
    (ROOT / "public/disagg/ced-results.json").write_text(json.dumps(browser, separators=(",", ":")) + "\n")
    print(f"wrote CED fixtures and {len(CED_WORKLOADS)} workloads at {commit[:7]}")


if __name__ == "__main__":
    main()
