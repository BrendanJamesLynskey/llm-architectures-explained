"""
The states behind the brief 41 animations (the reference for
``src/lib/chapters/visuals.ts``; written to tests/fixtures/visual_fixtures.json
by scripts/make_chapter_fixtures.py, checked by tests/unit/chapters/visuals.test.ts):

* ``rope_bank``: rotary position embedding as a bank of phasors. Pair i of a
  head rotates by m·ω_i at position m, ω_i = θ^(−2i/d). With the query and
  key equal in every pair (unit vectors, the best case), the score of a query
  at m against a key at n is the mean over pairs of cos((m − n)·ω_i): a sum
  of cosines, so it depends only on m − n. "query" moves the query with the
  key fixed at 0; "both" moves both, a fixed distance apart.
* ``head_sharing``: MHA, GQA, MQA and MLA on Llama 3 8B's body: query heads,
  the K/V units they share, and the KV bytes per token (the cost model's).
* ``moe_routing``: tokens routed to their top-k of E experts by sigmoid
  affinity (of the token-centroid dot product over √d), the way DeepSeek-V3 routes (arXiv:2412.19437, section 2.1.2),
  with its auxiliary-loss-free balancing: a bias per expert, added to the
  affinities only to choose the experts, nudged by γ after every step (down
  for an expert over the mean load, up for one under it). Token features and
  expert centroids come from a seeded generator (mulberry32 + Box–Muller, the
  site's), with the tokens clustered near a few experts so the load starts
  unbalanced.
"""
from __future__ import annotations

import math
from typing import Any

import arch_model as am

# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

ROPE_POSITIONS = 96
ROPE_SHOWN = 8
ROPE_GAP = 8  # "both": the query is this many tokens after the key


def rope_omegas(head_dim: int, theta: float) -> list[float]:
    d = head_dim
    return [theta ** ((-2.0 * i) / d) for i in range(d // 2)]


def rope_shown(pairs: int, shown: int = ROPE_SHOWN) -> list[int]:
    return [round(j * (pairs - 1) / (shown - 1)) for j in range(shown)]


def rope_bank(head_dim: int, theta: float, mode: str, positions: int = ROPE_POSITIONS) -> list[dict[str, Any]]:
    om = rope_omegas(head_dim, theta)
    show = rope_shown(len(om))
    out = []
    for t in range(positions):
        m, n = (t, 0) if mode == "query" else (t + ROPE_GAP, t)
        total = 0.0
        pairs = []
        for i, w in enumerate(om):
            aq = m * w
            ak = n * w
            # the dot product of the two rotated unit vectors (what RoPE computes)
            c = math.cos(aq) * math.cos(ak) + math.sin(aq) * math.sin(ak)
            total += c
            if i in show:
                pairs.append({"i": i, "omega": w, "wavelength": 2.0 * math.pi / w,
                              "angQ": aq, "angK": ak, "contrib": c})
        out.append({"t": t, "m": m, "n": n, "score": total / len(om), "pairs": pairs})
    return out


# ---------------------------------------------------------------------------
# Head sharing
# ---------------------------------------------------------------------------

SHARING = ("mha", "gqa", "mqa", "mla")


def head_sharing(variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for v in variants:
        if v["id"] not in SHARING:
            continue
        s = v["spec"]
        mx = next(iter(s["mixers"].values()))
        layers = sum(r["n"] for r in s["layout"])
        kv = am.kv_cache(s, 1.0, 2.0)
        if mx["type"] == "mla":
            units, width = 1, mx["kv_lora_rank"] + mx["qk_rope"]
        else:
            units, width = mx["kv_heads"], 2 * mx["head_dim"]
        out.append({"id": v["id"], "label": v["label"], "heads": mx["heads"], "units": units,
                    "unitElems": width, "layers": layers,
                    "bytesPerToken": kv["bytes_per_token_unbounded"],
                    "bytesPerTokenLayer": kv["bytes_per_token_unbounded"] / layers})
    return out


# ---------------------------------------------------------------------------
# MoE routing
# ---------------------------------------------------------------------------

U32 = 0xFFFFFFFF


def mulberry32(seed: int):
    state = seed & U32

    def rng() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & U32
        t = state
        t = ((t ^ (t >> 15)) * (t | 1)) & U32
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & U32)) & U32
        return ((t ^ (t >> 14)) & U32) / 4294967296

    return rng


def normal(rng):
    def sample() -> float:
        u1 = max(rng(), 1e-12)
        u2 = rng()
        return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)

    return sample


MOE = {"experts": 8, "top_k": 2, "tokens": 64, "dim": 8, "steps": 32, "gamma": 0.02,
       "hot": [0, 1, 2], "spread": 0.6, "seed": 7}


def moe_routing(gamma: float, cfg: dict[str, Any] = MOE) -> list[dict[str, Any]]:
    E, K, T, D = cfg["experts"], cfg["top_k"], cfg["tokens"], cfg["dim"]
    g = normal(mulberry32(cfg["seed"]))
    cent = [[g() for _ in range(D)] for _ in range(E)]
    hot = cfg["hot"]
    bias = [0.0] * E
    mean = T * K / E
    out = []
    for step in range(cfg["steps"]):
        tok = normal(mulberry32(cfg["seed"] * 1000 + step + 1))
        loads = [0] * E
        routes = []
        for t in range(T):
            # each token sits near one of the hot experts' centroids
            c = cent[hot[t % len(hot)]]
            u = [c[j] + cfg["spread"] * tok() for j in range(D)]
            s = []
            for e in range(E):
                x = 0.0
                for j in range(D):
                    x += u[j] * cent[e][j]
                s.append(1.0 / (1.0 + math.exp(-x / math.sqrt(D))))
            order = sorted(range(E), key=lambda e: (-(s[e] + bias[e]), e))
            pick = order[:K]
            tot = 0.0
            for e in pick:
                tot += s[e]
            routes.append({"experts": pick, "gates": [s[e] / tot for e in pick]})
            for e in pick:
                loads[e] += 1
        imbalance = max(loads) / mean
        out.append({"step": step, "routes": routes, "loads": loads, "bias": list(bias),
                    "imbalance": imbalance})
        for e in range(E):
            if loads[e] > mean:
                bias[e] -= gamma
            elif loads[e] < mean:
                bias[e] += gamma
    return out


def build(chapters: dict[str, Any]) -> dict[str, Any]:
    rope = {r["id"]: r for r in chapters["rope"]}
    banks = {}
    for mid in ("mistral-7b", "qwen3-8b"):
        r = rope[mid]
        for mode in ("query", "both"):
            banks[f"{mid}|{mode}"] = {"head_dim": r["head_dim"], "theta": r["theta"],
                                      "states": rope_bank(r["head_dim"], r["theta"], mode)}
    return {"note": "Brief 41 animation states from reference/visuals.py.",
            "rope": {"positions": ROPE_POSITIONS, "shown": ROPE_SHOWN, "gap": ROPE_GAP, "banks": banks},
            "sharing": head_sharing(chapters["attention"]["variants"]),
            "moe": {"cfg": MOE, "on": moe_routing(MOE["gamma"]), "off": moe_routing(0.0)}}
