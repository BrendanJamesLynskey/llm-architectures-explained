"""Architecture cost model: the Python reference.

Given one model's normalised architecture (the ``arch`` block of a
``data/models/<id>.yaml`` file, with provenance stripped), this module
computes:

* parameter counts, total and active per token, split by component;
* KV-cache (and recurrent-state) bytes per token and in total, for every
  mixer type: MHA/GQA/MQA, sliding window, MLA, DeepSeek sparse attention
  (DSA), compressed attention (DeepSeek V4 CSA/HCA), linear attention
  (Gated DeltaNet, KDA, Lightning), Mamba, short convolutions, RWKV, mLSTM;
* FLOPs per token for prefill and for decode, including the Causal
  Encoder-Decoder (CED), where prefill runs only the encoder half plus a
  bounded replay of the last ``W`` tokens through the decoder half;
* memory-traffic bytes per decode step (weights plus the KV read).

Every quantity is a closed form. ``src/lib/arch/costModel.ts`` is a line by
line TypeScript port; ``scripts/make_fixtures.py`` writes fixtures that the
port must reproduce exactly. To make "exactly" possible, every value is a
Python ``float`` and the operation order below is mirrored in the port. Do
not "simplify" an expression here without changing the port the same way.

Conventions (documented on the site's /about page as well):

* A matmul of an ``m``-vector by an ``m x n`` matrix costs ``2 m n`` FLOPs.
* Attention FLOPs use the expanded (non-absorbed) form for every type.
* Normalisation, activation, softmax, rotary and routing arithmetic are not
  counted (they are a few per cent at most).
* Per-token FLOPs at context position ``t`` attend to ``t`` cached tokens
  (the current token included).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Spec = dict[str, Any]

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _f(x: Any) -> float:
    return float(x)


def expand_layout(spec: Spec, key: str = "layout") -> list[dict[str, Any]]:
    """Expand the run-length ``layout`` into one dict per layer."""
    out: list[dict[str, Any]] = []
    for run in spec.get(key) or []:
        n = int(run["n"])
        for _ in range(n):
            out.append(run)
    return out


def sum_min(n: float, w: float) -> float:
    """sum_{t=1..n} min(t, w), as a float (w <= 0 means no limit)."""
    if w <= 0 or n <= w:
        return n * (n + 1.0) / 2.0
    return w * (w + 1.0) / 2.0 + (n - w) * w


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def mixer_params(m: dict[str, Any], d: float, kv_shared: bool = False) -> float:
    """Weights of one mixer (token-mixing block) of model width ``d``.

    ``kv_shared``: the layer reuses another layer's keys and values, so it
    has no key or value projections of its own (Gemma 4's shared-KV layers).
    """
    t = m["type"]
    if t == "attn" and kv_shared:
        h = _f(m["heads"])
        hd = _f(m["head_dim"])
        vd = _f(m.get("v_head_dim") or m["head_dim"])
        return d * h * hd + h * vd * d
    if t == "attn":
        h = _f(m["heads"])
        kv = _f(m["kv_heads"])
        hd = _f(m["head_dim"])
        vd = _f(m.get("v_head_dim") or m["head_dim"])
        q = d * h * hd
        if m.get("gate") == "elementwise":
            q = q + d * h * vd
        elif m.get("gate") == "headwise":
            q = q + d * h
        k = d * kv * hd
        v = 0.0 if m.get("k_eq_v") else d * kv * vd
        o = h * vd * d
        p = q + k + v + o
        if m.get("bias"):
            p = p + h * hd + kv * hd + kv * vd
        if m.get("qk_norm"):
            p = p + 2.0 * hd
        ix = m.get("indexer")
        if ix:
            ih = _f(ix["heads"])
            idd = _f(ix["head_dim"])
            p = p + d * ih * idd + d * idd + d * ih
        return p
    if t == "mla":
        h = _f(m["heads"])
        qk = _f(m["qk_nope"]) + _f(m["qk_rope"])
        vd = _f(m["v_head_dim"])
        r_kv = _f(m["kv_lora_rank"])
        rope = _f(m["qk_rope"])
        if m.get("q_lora_rank"):
            r_q = _f(m["q_lora_rank"])
            q = d * r_q + r_q + r_q * h * qk
        else:
            q = d * h * qk
        kv = d * (r_kv + rope) + r_kv + r_kv * h * (_f(m["qk_nope"]) + vd)
        o = h * vd * d
        p = q + kv + o
        if m.get("gate") == "elementwise":
            p = p + d * h * vd
        ix = m.get("indexer")
        if ix:
            ih = _f(ix["heads"])
            idd = _f(ix["head_dim"])
            q_in = _f(m["q_lora_rank"]) if m.get("q_lora_rank") else d
            p = p + q_in * ih * idd + d * idd + d * ih
        return p
    if t == "csa":
        # DeepSeek V4: one shared latent KV head (head_dim wide), low-rank
        # query, grouped low-rank output, plus the compressor and indexer.
        h = _f(m["heads"])
        hd = _f(m["head_dim"])
        r_q = _f(m["q_lora_rank"])
        r_o = _f(m["o_lora_rank"])
        g = _f(m["o_groups"])
        q = d * r_q + r_q * h * hd
        kv = d * hd
        # grouped low-rank output: each of g groups projects its heads to
        # r_o, then the g * r_o concatenation projects to d
        o = h * hd * r_o + g * r_o * d
        p = q + kv + o
        ix = m.get("indexer")
        if ix:
            ih = _f(ix["heads"])
            idd = _f(ix["head_dim"])
            p = p + r_q * ih * idd + d * idd + d * ih
        return p
    if t in ("deltanet", "kda"):
        kh = _f(m["k_heads"])
        vh = _f(m["v_heads"])
        kd = _f(m["k_head_dim"])
        vdd = _f(m["v_head_dim"])
        ck = _f(m.get("conv_kernel") or 0)
        qk_dim = kh * kd
        v_dim = vh * vdd
        proj = d * (2.0 * qk_dim + v_dim)
        if t == "deltanet":
            proj = proj + d * v_dim + d * 2.0 * vh  # z gate; beta and alpha
        else:
            proj = proj + d * v_dim + d * vh + 2.0 * d * kd + 2.0 * kd * qk_dim
        conv = (2.0 * qk_dim + v_dim) * ck
        out = v_dim * d
        return proj + conv + out + 2.0 * vh
    if t == "linear":
        # Lightning / plain linear attention: q, k, v, output gate, output.
        h = _f(m["heads"])
        hd = _f(m["head_dim"])
        return 5.0 * d * h * hd
    if t == "mamba2":
        nh = _f(m["heads"])
        hd = _f(m["head_dim"])
        n = _f(m["state"])
        g = _f(m["groups"])
        ck = _f(m["conv_kernel"])
        di = nh * hd
        in_proj = d * (2.0 * di + 2.0 * g * n + nh)
        conv = (di + 2.0 * g * n) * ck
        return in_proj + conv + 3.0 * nh + di + di * d
    if t == "mamba1":
        di = _f(m["d_inner"])
        n = _f(m["state"])
        ck = _f(m["conv_kernel"])
        r = _f(m["dt_rank"])
        return d * 2.0 * di + di * ck + di * (r + 2.0 * n) + r * di + di * n + di + di * d
    if t == "conv":
        ck = _f(m["kernel"])
        return d * 3.0 * d + d * ck + d * d
    if t == "rwkv":
        # time mixing: receptance, key, value, output (RWKV-4) plus a gate
        # (RWKV-5/6), each d x d
        return _f(m["mats"]) * d * d
    if t == "mlstm":
        qk = _f(m["qk_dim"])
        vd = _f(m["v_dim"])
        h = _f(m["heads"])
        return d * qk * 2.0 + d * vd + d * vd + 2.0 * d * h + vd * d
    if t == "none":
        return 0.0
    raise ValueError(f"unknown mixer type {t}")


def ffn_params(f: dict[str, Any], d: float) -> tuple[float, float]:
    """(total, active) weights of one FFN block."""
    t = f["type"]
    if t == "none":
        return 0.0, 0.0
    if t == "dense":
        mats = 3.0 if f.get("gated", True) else 2.0
        p = mats * d * _f(f["d_ff"])
        if f.get("receptance"):
            p = p + d * d
        if f.get("bias"):
            p = p + _f(f["d_ff"]) + d
        return p, p
    if t == "moe":
        mats = 3.0 if f.get("gated", True) else 2.0
        e = _f(f["experts"])
        k = _f(f["active"])
        de = _f(f["d_expert"])
        width = _f(f["latent"]) if f.get("latent") else d
        per_expert = mats * width * de
        router = d * e
        shared = 0.0
        if f.get("shared"):
            ds = _f(f.get("d_shared") or f["d_expert"])
            shared = _f(f["shared"]) * mats * d * ds
        latent = 2.0 * d * width if f.get("latent") else 0.0
        parallel = 0.0
        if f.get("dense_parallel_d_ff"):
            parallel = mats * d * _f(f["dense_parallel_d_ff"])
        total = e * per_expert + router + shared + latent + parallel
        active = k * per_expert + router + shared + latent + parallel
        return total, active
    raise ValueError(f"unknown ffn type {t}")


@dataclass
class Params:
    embedding: float
    norms: float
    lm_head: float
    extra_embedding: float
    mixers: float
    ffn_total: float
    ffn_active: float
    cross_attention: float
    mtp: float
    total: float
    active: float
    # FLOPs-relevant weights touched per decoded token (matmuls only:
    # active minus the input-embedding lookup and the extra tables)
    matmul_active: float
    # Totals without the input embedding, the output head and the extra
    # lookup tables (labs often quote these as "non-embedding" counts)
    non_embedding_total: float
    non_embedding_active: float


def layer_params(spec: Spec, run: dict[str, Any]) -> tuple[float, float, float]:
    """(mixer, ffn total, ffn active) for one layer."""
    d = _f(spec["d_model"])
    mix = (
        mixer_params(spec["mixers"][run["mixer"]], d, bool(run.get("kv_shared")))
        if run["mixer"] != "none"
        else 0.0
    )
    if run.get("ffn", "none") == "none":
        ft, fa = 0.0, 0.0
    else:
        ft, fa = ffn_params(spec["ffns"][run["ffn"]], d)
    return mix, ft, fa


def params(spec: Spec) -> Params:
    d = _f(spec["d_model"])
    vocab = _f(spec["vocab"])
    emb = vocab * d
    head = 0.0 if spec.get("tied_embeddings") else vocab * d
    extra = _f(spec.get("extra_embedding_params") or 0)
    norms = _f(spec.get("norms_per_layer") or 2)
    mixers = 0.0
    ft = 0.0
    fa = 0.0
    layers = expand_layout(spec)
    norm_w = 0.0
    for run in layers:
        m, t, a = layer_params(spec, run)
        mixers = mixers + m
        norm_w = norm_w + norms * d
        ft = ft + t
        fa = fa + a
    cross = 0.0
    if spec.get("kind") == "encoder-decoder":
        for run in expand_layout(spec, "encoder_layout"):
            m, t, a = layer_params(spec, run)
            mixers = mixers + m
            norm_w = norm_w + norms * d
            ft = ft + t
            fa = fa + a
        cm = spec["mixers"][spec.get("cross_mixer", "full")]
        for _ in layers:
            cross = cross + mixer_params(cm, d)
            norm_w = norm_w + d
    mtp = 0.0
    n_mtp = _f(spec.get("mtp_layers") or 0)
    if n_mtp > 0:
        run = spec.get("mtp_layer") or spec["layout"][-1]
        m, t, _a = layer_params(spec, run)
        mtp = n_mtp * (m + t + 2.0 * d * d + 4.0 * d)
    total = emb + head + extra + mixers + norm_w + ft + cross
    active = emb + head + extra + mixers + norm_w + fa + cross
    # norm weights scale activations; they are not matmuls
    matmul_active = head + mixers + fa + cross
    if spec.get("tied_embeddings"):
        matmul_active = matmul_active + emb
    ne_total = mixers + norm_w + ft + cross
    ne_active = mixers + norm_w + fa + cross
    return Params(emb, norm_w, head, extra, mixers, ft, fa, cross, mtp, total, active, matmul_active, ne_total, ne_active)


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------


def kv_elems_per_token(spec: Spec, run: dict[str, Any]) -> float:
    """Cache elements one layer adds per token (0 for state-only mixers)."""
    if run.get("kv_shared"):
        return 0.0
    name = run["mixer"]
    if name == "none":
        return 0.0
    m = spec["mixers"][name]
    t = m["type"]
    if t == "attn":
        kv = _f(m["kv_heads"])
        hd = _f(m["head_dim"])
        vd = _f(m.get("v_head_dim") or m["head_dim"])
        e = kv * hd if m.get("k_eq_v") else kv * (hd + vd)
        ix = m.get("indexer")
        if ix and run.get("indexer", True):
            e = e + _f(ix["head_dim"])
        return e
    if t == "mla":
        e = _f(m["kv_lora_rank"]) + _f(m["qk_rope"])
        ix = m.get("indexer")
        if ix and run.get("indexer", True):
            e = e + _f(ix["head_dim"])
        return e
    if t == "csa":
        ratio = _f(run.get("ratio") or 0)
        if ratio <= 0 or run.get("kv_source") is False:
            return 0.0
        e = _f(m["head_dim"]) / ratio
        ix = m.get("indexer")
        if ix and run.get("indexer"):
            e = e + _f(ix["head_dim"]) / ratio
        return e
    return 0.0


def window_of(spec: Spec, run: dict[str, Any]) -> float:
    """Tokens a layer keeps (0 = all of them). Compressed attention keeps
    every compressed entry; its sliding window is a separate side cache
    (counted in state_elems)."""
    if run["mixer"] == "none":
        return 0.0
    m = spec["mixers"][run["mixer"]]
    if m["type"] == "csa":
        return 0.0
    return _f(m.get("window") or 0)


def state_elems(spec: Spec, run: dict[str, Any]) -> float:
    """Fixed-size per-sequence state of one layer (recurrent state, conv
    taps, or a sliding-window side cache for compressed attention)."""
    name = run["mixer"]
    if name == "none":
        return 0.0
    m = spec["mixers"][name]
    t = m["type"]
    if t in ("deltanet", "kda"):
        vh = _f(m["v_heads"])
        kd = _f(m["k_head_dim"])
        vdd = _f(m["v_head_dim"])
        ck = _f(m.get("conv_kernel") or 0)
        conv = (2.0 * _f(m["k_heads"]) * kd + vh * vdd) * (ck - 1.0) if ck > 0 else 0.0
        return vh * kd * vdd + conv
    if t == "linear":
        h = _f(m["heads"])
        hd = _f(m["head_dim"])
        return h * hd * hd
    if t == "mamba2":
        nh = _f(m["heads"])
        hd = _f(m["head_dim"])
        n = _f(m["state"])
        g = _f(m["groups"])
        ck = _f(m["conv_kernel"])
        return nh * hd * n + (nh * hd + 2.0 * g * n) * (ck - 1.0)
    if t == "mamba1":
        di = _f(m["d_inner"])
        return di * _f(m["state"]) + di * (_f(m["conv_kernel"]) - 1.0)
    if t == "conv":
        d = _f(spec["d_model"])
        return d * (_f(m["kernel"]) - 1.0)
    if t == "rwkv":
        d = _f(spec["d_model"])
        hd = _f(m["head_dim"])
        return d * hd + 2.0 * d
    if t == "mlstm":
        h = _f(m["heads"])
        qk = _f(m["qk_dim"]) / h
        vd = _f(m["v_dim"]) / h
        return h * (qk * vd + qk + 1.0)
    if t == "csa":
        w = _f(m.get("window") or 0)
        return w * _f(m["head_dim"])
    return 0.0


def kv_cache(spec: Spec, context: float, bytes_per_elem: float, state_bytes: float = 2.0) -> dict[str, float]:
    """Cache bytes for one sequence of ``context`` tokens."""
    growing = 0.0
    windowed = 0.0
    state = 0.0
    per_token_unbounded = 0.0
    layers = expand_layout(spec)
    for run in layers:
        e = kv_elems_per_token(spec, run)
        w = window_of(spec, run)
        if e > 0:
            if w > 0:
                windowed = windowed + e * min(context, w)
            else:
                growing = growing + e * context
                per_token_unbounded = per_token_unbounded + e
        state = state + state_elems(spec, run)
    if spec.get("kind") == "encoder-decoder":
        # Cross-attention keys and values over the encoder output: one set
        # per decoder layer, for ``context`` input tokens.
        cm = spec["mixers"][spec.get("cross_mixer", "full")]
        e = _f(cm["kv_heads"]) * (_f(cm["head_dim"]) + _f(cm.get("v_head_dim") or cm["head_dim"]))
        growing = growing + e * context * _f(len(layers))
        per_token_unbounded = per_token_unbounded + e * _f(len(layers))
    total = (growing + windowed) * bytes_per_elem + state * state_bytes
    return {
        "kv_bytes": (growing + windowed) * bytes_per_elem,
        "state_bytes": state * state_bytes,
        "total_bytes": total,
        "bytes_per_token_unbounded": per_token_unbounded * bytes_per_elem,
    }


# ---------------------------------------------------------------------------
# FLOPs
# ---------------------------------------------------------------------------


def attended(spec: Spec, run: dict[str, Any], t: float) -> float:
    """Cache entries one layer's attention reads for the token at position t."""
    name = run["mixer"]
    if name == "none":
        return 0.0
    m = spec["mixers"][name]
    if m["type"] == "attn":
        ix = m.get("indexer")
        if ix:
            return min(t, _f(ix["topk"]))
        w = _f(m.get("window") or 0)
        return min(t, w) if w > 0 else t
    if m["type"] == "mla":
        ix = m.get("indexer")
        if ix:
            return min(t, _f(ix["topk"]))
        return t
    if m["type"] == "csa":
        ratio = _f(run.get("ratio") or 0)
        w = _f(m.get("window") or 0)
        e = min(t, w)
        if ratio > 0:
            c = t / ratio
            ix = m.get("indexer")
            if ix and run.get("indexer"):
                c = min(c, _f(ix["topk"]))
            e = e + c
        return e
    return 0.0


def attn_flops_per_entry(spec: Spec, run: dict[str, Any]) -> float:
    """FLOPs per attended entry (scores plus weighted sum)."""
    name = run["mixer"]
    if name == "none":
        return 0.0
    m = spec["mixers"][name]
    if m["type"] == "attn":
        h = _f(m["heads"])
        return 2.0 * h * (_f(m["head_dim"]) + _f(m.get("v_head_dim") or m["head_dim"]))
    if m["type"] == "mla":
        h = _f(m["heads"])
        return 2.0 * h * (_f(m["qk_nope"]) + _f(m["qk_rope"]) + _f(m["v_head_dim"]))
    if m["type"] == "csa":
        h = _f(m["heads"])
        return 2.0 * h * (2.0 * _f(m["head_dim"]))
    return 0.0


def indexer_flops_per_entry(spec: Spec, run: dict[str, Any]) -> float:
    name = run["mixer"]
    if name == "none":
        return 0.0
    m = spec["mixers"][name]
    ix = m.get("indexer")
    if not ix:
        return 0.0
    if m["type"] == "csa" and not run.get("indexer"):
        return 0.0
    return 2.0 * _f(ix["heads"]) * _f(ix["head_dim"])


def recurrent_flops(spec: Spec, run: dict[str, Any]) -> float:
    """State-update FLOPs per token for recurrent mixers."""
    name = run["mixer"]
    if name == "none":
        return 0.0
    m = spec["mixers"][name]
    t = m["type"]
    if t in ("deltanet", "kda"):
        return 6.0 * _f(m["v_heads"]) * _f(m["k_head_dim"]) * _f(m["v_head_dim"])
    if t == "linear":
        h = _f(m["heads"])
        hd = _f(m["head_dim"])
        return 4.0 * h * hd * hd
    if t == "mamba2":
        return 6.0 * _f(m["heads"]) * _f(m["head_dim"]) * _f(m["state"])
    if t == "mamba1":
        return 6.0 * _f(m["d_inner"]) * _f(m["state"])
    if t == "rwkv":
        return 4.0 * _f(spec["d_model"]) * _f(m["head_dim"])
    if t == "mlstm":
        h = _f(m["heads"])
        return 4.0 * h * (_f(m["qk_dim"]) / h) * (_f(m["v_dim"]) / h)
    return 0.0


def _sum_attended(spec: Spec, run: dict[str, Any], n: float) -> float:
    """sum over t = 1..n of attended(t), in closed form."""
    name = run["mixer"]
    if name == "none":
        return 0.0
    m = spec["mixers"][name]
    if m["type"] == "attn":
        ix = m.get("indexer")
        if ix:
            return sum_min(n, _f(ix["topk"]))
        return sum_min(n, _f(m.get("window") or 0))
    if m["type"] == "mla":
        ix = m.get("indexer")
        if ix:
            return sum_min(n, _f(ix["topk"]))
        return n * (n + 1.0) / 2.0
    if m["type"] == "csa":
        w = _f(m.get("window") or 0)
        s = sum_min(n, w)
        ratio = _f(run.get("ratio") or 0)
        if ratio > 0:
            ix = m.get("indexer")
            if ix and run.get("indexer"):
                # sum min(t / ratio, k) = sum min(t, k ratio) / ratio
                s = s + sum_min(n, _f(ix["topk"]) * ratio) / ratio
            else:
                s = s + n * (n + 1.0) / 2.0 / ratio
        return s
    return 0.0


def layer_matmul(spec: Spec, run: dict[str, Any]) -> float:
    mix, _t, fa = layer_params(spec, run)
    return mix + fa


def decode_flops(spec: Spec, context: float) -> float:
    """FLOPs to generate one token with ``context`` tokens in the cache."""
    p = params(spec)
    f = 2.0 * p.matmul_active
    loops = _f(spec.get("loops") or 1)
    attn = 0.0
    for run in expand_layout(spec):
        attn = attn + attended(spec, run, context) * attn_flops_per_entry(spec, run)
        attn = attn + context * indexer_flops_per_entry(spec, run)
        attn = attn + recurrent_flops(spec, run)
    if spec.get("kind") == "encoder-decoder":
        cm = spec["mixers"][spec.get("cross_mixer", "full")]
        per = 2.0 * _f(cm["heads"]) * (_f(cm["head_dim"]) + _f(cm.get("v_head_dim") or cm["head_dim"]))
        attn = attn + per * context * _f(len(expand_layout(spec)))
    if loops > 1:
        # A looped model reruns its layer stack: everything but the
        # embeddings and the output head.
        stack = 2.0 * (p.matmul_active - p.lm_head - (p.embedding if spec.get("tied_embeddings") else 0.0))
        return f + (loops - 1.0) * stack + loops * attn
    return f + attn


def _layers_flops_prefill(spec: Spec, runs: list[dict[str, Any]], n: float) -> float:
    total = 0.0
    for run in runs:
        total = total + 2.0 * layer_matmul(spec, run) * n
        total = total + _sum_attended(spec, run, n) * attn_flops_per_entry(spec, run)
        total = total + n * (n + 1.0) / 2.0 * indexer_flops_per_entry(spec, run)
        total = total + recurrent_flops(spec, run) * n
    return total


def prefill_flops(spec: Spec, n: float) -> float:
    """FLOPs to prefill an ``n``-token prompt (logits for the last token).

    For a CED model only the encoder layers run over the whole prompt; the
    decoder layers project their K/V from the encoder output for every
    token and run in full for the last ``ced_window`` tokens (the paper's
    Decoder SWA Bounded Replay)."""
    p = params(spec)
    runs = expand_layout(spec)
    head = 2.0 * (p.lm_head if not spec.get("tied_embeddings") else p.embedding)
    loops = _f(spec.get("loops") or 1)
    if spec.get("kind") == "ced":
        k = int(spec["ced_encoder_layers"])
        enc = runs[:k]
        dec = runs[k:]
        w = min(n, _f(spec.get("ced_window") or 0))
        total = _layers_flops_prefill(spec, enc, n)
        d = _f(spec["d_model"])
        kv_proj = 0.0
        for run in dec:
            kv_proj = kv_proj + kv_elems_per_token(spec, run) * d
        total = total + 2.0 * kv_proj * n
        total = total + _layers_flops_prefill(spec, dec, w)
        return total + head
    total = _layers_flops_prefill(spec, runs, n)
    if spec.get("kind") == "encoder-decoder":
        total = total + _layers_flops_prefill(spec, expand_layout(spec, "encoder_layout"), n)
    return loops * total + head


def decode_bytes(spec: Spec, context: float, weight_bytes: float, kv_bytes: float, state_bytes: float = 2.0) -> dict[str, float]:
    """Bytes read from memory for one decode step at batch size 1."""
    p = params(spec)
    w = (p.active - p.embedding - p.extra_embedding) * weight_bytes
    if spec.get("tied_embeddings"):
        w = w + p.embedding * weight_bytes
    kv = 0.0
    state = 0.0
    for run in expand_layout(spec):
        e = kv_elems_per_token(spec, run)
        if e > 0:
            kv = kv + attended_stored(spec, run, context) * e
        state = state + state_elems(spec, run)
    if spec.get("kind") == "encoder-decoder":
        cm = spec["mixers"][spec.get("cross_mixer", "full")]
        e = _f(cm["kv_heads"]) * (_f(cm["head_dim"]) + _f(cm.get("v_head_dim") or cm["head_dim"]))
        kv = kv + e * context * _f(len(expand_layout(spec)))
    total = w + kv * kv_bytes + state * state_bytes * 2.0
    return {"weights": w, "kv": kv * kv_bytes, "state": state * state_bytes * 2.0, "total": total}


def attended_stored(spec: Spec, run: dict[str, Any], t: float) -> float:
    """Stored entries a decode step reads in one layer (sparse selection
    reads only the selected entries; the indexer keys are read in full and
    counted through the element size)."""
    m = spec["mixers"][run["mixer"]]
    if m["type"] == "csa":
        ratio = _f(run.get("ratio") or 0)
        return t / ratio if ratio > 0 else 0.0
    w = _f(m.get("window") or 0)
    return min(t, w) if w > 0 else t


def summary(spec: Spec, contexts: list[float] | None = None) -> dict[str, Any]:
    contexts = contexts or [1024.0, 8192.0, 32768.0, 131072.0]
    p = params(spec)
    out: dict[str, Any] = {"params": p.__dict__}
    out["kv"] = {str(int(c)): kv_cache(spec, c, 2.0) for c in contexts}
    out["decode_flops"] = {str(int(c)): decode_flops(spec, c) for c in contexts}
    out["prefill_flops"] = {str(int(c)): prefill_flops(spec, c) for c in contexts}
    out["decode_bytes"] = {str(int(c)): decode_bytes(spec, c, 2.0, 2.0) for c in contexts}
    return out
