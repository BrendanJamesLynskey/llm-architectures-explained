"""Chapter reference: the spec builders and closed forms behind the concept chapters.

The chapters under ``content/chapters/`` hold one interactive each. The ones
that vary an architecture build a spec here (from a real model's sourced
shape) and cost it with ``arch_model``; the others use a closed form defined
here. ``src/lib/chapters/*.ts`` is a line by line TypeScript port, and
``scripts/make_chapter_fixtures.py`` writes fixtures the port must
reproduce: exactly, except where a closed form calls ``pow`` (RoPE
wavelengths), which libms may round differently in the last bit, so those
compare to 1e-12 relative.

Everything here is a model of the architecture, not a measurement:

* the attention variants keep Llama 3 8B's widths and swap only the token
  mixer, so the differences are the mixer's alone;
* the normalisation profile is the textbook variance argument (each
  sub-block's output independent of the stream, with a fixed gain), the
  idealisation of Xiong et al. (arXiv:2002.04745), not a trained model;
* the multi-token-prediction speed-up assumes every drafted token is
  accepted independently with the same probability and that decode is
  memory-bound at batch 1.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import arch_model as am

Spec = dict[str, Any]

# ---------------------------------------------------------------------------
# Spec builders
# ---------------------------------------------------------------------------


def _decoder(base: Spec, mixers: dict[str, Any], layout: list[dict[str, Any]]) -> Spec:
    s = copy.deepcopy(base)
    s["mixers"] = copy.deepcopy(mixers)
    s["layout"] = copy.deepcopy(layout)
    return s


def _pattern(a: str, na: int, b: str, nb: int, layers: int, ffn: str) -> list[dict[str, Any]]:
    """Repeat (na x a, nb x b) until ``layers`` layers; the last repeat may be cut short."""
    out: list[dict[str, Any]] = []
    left = layers
    while left > 0:
        k = min(na, left)
        out.append({"mixer": a, "ffn": ffn, "n": k})
        left -= k
        if left <= 0:
            break
        k = min(nb, left)
        out.append({"mixer": b, "ffn": ffn, "n": k})
        left -= k
    return out


def attention_variants(base: Spec) -> list[dict[str, Any]]:
    """The token mixers of chapter 1 on one base shape (its width, depth, FFN and vocabulary).

    ``base`` must be a single-mixer GQA decoder (Llama 3 8B: 32 layers, 32 query heads of 128, 8 KV heads).
    """
    full = base["mixers"]["full"]
    h, hd = full["heads"], full["head_dim"]
    layers = sum(r["n"] for r in base["layout"])
    ffn = base["layout"][0]["ffn"]
    gqa = dict(full)
    mha = {**full, "kv_heads": h}
    mqa = {**full, "kv_heads": 1}
    # DeepSeek-V3's latent sizes (kv_lora_rank 512, 64 rotary dims), on this width without a query latent
    mla = {"type": "mla", "heads": h, "kv_lora_rank": 512, "qk_nope": hd, "qk_rope": 64, "v_head_dim": hd}
    # Gemma 3's pattern: five sliding-window layers (1,024 tokens) to one global layer
    swa = {**full, "window": 1024}
    # DeepSeek-V3.2's sparse attention: an indexer picks 2,048 cached tokens per query
    dsa = {**full, "indexer": {"heads": 64, "head_dim": 128, "topk": 2048}}
    # Qwen3-Next's Gated DeltaNet sizes, three linear layers to one full-attention layer
    delta = {"type": "deltanet", "k_heads": 16, "v_heads": 32, "k_head_dim": 128, "v_head_dim": 128, "conv_kernel": 4}
    # Mamba-2 (state 128, as Nemotron-H and Nemotron 3 use), seven to one, as Jamba's ratio
    mamba = {"type": "mamba2", "heads": 2 * base["d_model"] // 64, "head_dim": 64, "state": 128, "groups": 8,
             "conv_kernel": 4}
    one = [{"mixer": "full", "ffn": ffn, "n": layers}]
    return [
        {"id": "mha", "label": "MHA", "spec": _decoder(base, {"full": mha}, one)},
        {"id": "gqa", "label": "GQA (8 KV heads)", "spec": _decoder(base, {"full": gqa}, one)},
        {"id": "mqa", "label": "MQA", "spec": _decoder(base, {"full": mqa}, one)},
        {"id": "mla", "label": "MLA", "spec": _decoder(base, {"mla": mla}, [{"mixer": "mla", "ffn": ffn, "n": layers}])},
        {"id": "swa", "label": "Sliding window 5:1", "spec": _decoder(
            base, {"full": gqa, "sliding": swa}, _pattern("sliding", 5, "full", 1, layers, ffn))},
        {"id": "dsa", "label": "Sparse (indexer, top 2,048)", "spec": _decoder(base, {"full": dsa}, one)},
        {"id": "deltanet", "label": "Gated DeltaNet 3:1", "spec": _decoder(
            base, {"full": gqa, "linear": delta}, _pattern("linear", 3, "full", 1, layers, ffn))},
        {"id": "mamba", "label": "Mamba-2 7:1", "spec": _decoder(
            base, {"full": gqa, "mamba": mamba}, _pattern("mamba", 7, "full", 1, layers, ffn))},
    ]


def moe_spec(base: Spec, experts: int, active: int, shared: int, granularity: int, dense_prefix: int) -> Spec:
    """Replace a dense model's FFN with a mixture of experts.

    Each expert is ``d_ff / granularity`` wide (granularity 1 = Mixtral's full-width experts; DeepSeekMoE's
    fine-grained experts split them further), ``shared`` experts of the same width always run, and the first
    ``dense_prefix`` layers keep the dense FFN. ``experts == 1`` returns the dense model.
    """
    s = copy.deepcopy(base)
    dense = s["ffns"]["dense"]
    layers = sum(r["n"] for r in s["layout"])
    mixer = s["layout"][0]["mixer"]
    if experts <= 1:
        return s
    d_expert = dense["d_ff"] // granularity
    moe = {"type": "moe", "experts": experts, "active": active, "d_expert": d_expert, "gated": dense.get("gated", True)}
    if shared > 0:
        moe["shared"] = shared
        moe["d_shared"] = d_expert
    s["ffns"] = {"dense": dense, "moe": moe}
    layout = []
    if dense_prefix > 0:
        layout.append({"mixer": mixer, "ffn": "dense", "n": dense_prefix})
    layout.append({"mixer": mixer, "ffn": "moe", "n": layers - dense_prefix})
    s["layout"] = layout
    return s


def depth_width_spec(layers: int, d_model: int, vocab: int = 128256) -> Spec:
    """A Llama-style decoder of a given depth and width: 128-wide heads, 8 KV heads (fewer if fewer heads),
    a gated FFN 3.5x as wide as the model (rounded down to a multiple of 256), untied embeddings."""
    heads = d_model // 128
    return {
        "kind": "decoder",
        "d_model": d_model,
        "vocab": vocab,
        "tied_embeddings": False,
        "mixers": {"full": {"type": "attn", "heads": heads, "kv_heads": min(8, heads), "head_dim": 128}},
        "ffns": {"dense": {"type": "dense", "d_ff": (7 * d_model // 2) // 256 * 256, "gated": True}},
        "layout": [{"mixer": "full", "ffn": "dense", "n": layers}],
    }


def looped_spec(base: Spec, loops: int) -> Spec:
    s = copy.deepcopy(base)
    if loops > 1:
        s["loops"] = loops
    else:
        s.pop("loops", None)
    return s


def as_decoder(spec: Spec) -> Spec:
    """The same stack run as an ordinary decoder: every prompt token through every layer."""
    s = copy.deepcopy(spec)
    s["kind"] = "decoder"
    s.pop("ced_encoder_layers", None)
    s.pop("ced_window", None)
    return s


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------


def rotary_dims(head_dim: int, fraction: float) -> int:
    """Rotated dimensions of a head: the fraction of head_dim, rounded down to an even number."""
    return int(head_dim * fraction) // 2 * 2


def rope_wavelengths(head_dim: int, fraction: float, theta: float) -> list[float]:
    """Wavelength, in tokens, of each rotated pair i: 2 pi / theta^(-2i/d_rot) = 2 pi theta^(2i/d_rot)."""
    d = rotary_dims(head_dim, fraction)
    return [2.0 * math.pi * theta ** (2.0 * i / d) for i in range(d // 2)]


def rope_long_pairs(head_dim: int, fraction: float, theta: float, context: float) -> int:
    """Pairs whose wavelength exceeds the context: they turn less than once across it."""
    return sum(1 for w in rope_wavelengths(head_dim, fraction, theta) if w > context)


PLACEMENTS = ("post-ln", "pre-norm", "output-norm")


def norm_profile(placement: str, layers: int, gain: float, gamma: float = 1.0) -> dict[str, list[float] | float]:
    """Residual-stream RMS after each sub-block (two per layer), each sub-block's update relative to the
    stream it is added to, and the embedding's share of the final stream (amplitude), in the variance
    idealisation: unit-variance embedding, each sub-block's output independent of the stream.

    * ``post-ln`` (the original Transformer): x <- LN(x + f(x)); f's output has RMS ``gain`` times its input's.
    * ``pre-norm`` (GPT-2 onwards): x <- x + f(LN(x)); f sees a unit-RMS input, so adds RMS ``gain``.
    * ``output-norm`` (sandwich norm, and OLMo 2's norm after the sub-block): x <- x + gamma * N(f(.)); the
      update's RMS is the norm's scale ``gamma``, whatever ``gain`` is.
    """
    if placement not in PLACEMENTS:
        raise ValueError(f"unknown placement {placement}")
    var = 1.0
    emb = 1.0
    rms: list[float] = []
    update: list[float] = []
    for _ in range(2 * layers):
        if placement == "post-ln":
            update.append(gain / math.sqrt(var))
            total = var + gain * gain
            emb = emb / math.sqrt(total)
            var = 1.0
        else:
            add = gain if placement == "pre-norm" else gamma
            update.append(add / math.sqrt(var))
            var = var + add * add
            emb = 1.0 / math.sqrt(var)
        rms.append(math.sqrt(var))
    return {"rms": rms, "update": update, "embedding_share": emb}


def mtp_expected_tokens(acceptance: float, depth: int) -> float:
    """Tokens emitted per decode step with ``depth`` drafted tokens, each accepted with probability
    ``acceptance`` given the ones before it: 1 + a + a^2 + ... + a^depth."""
    total = 0.0
    term = 1.0
    for _ in range(depth + 1):
        total = total + term
        term = term * acceptance
    return total


def mtp_module_active(spec: Spec) -> float:
    """Weights one MTP module touches per token: one block (its mixer and active FFN) plus the projection
    that merges the previous depth's state with the next token's embedding (2 d^2) and its norms (4 d).
    The same per-module convention as ``arch_model.params`` (which counts the module's total FFN)."""
    d = am._f(spec["d_model"])
    run = spec.get("mtp_layer") or spec["layout"][-1]
    m, _t, a = am.layer_params(spec, run)
    return m + a + 2.0 * d * d + 4.0 * d


def mtp_speedup(spec: Spec, context: float, depth: int, acceptance: float, weight_bytes: float, kv_bytes: float) -> dict[str, float]:
    """Memory-bound decode at batch 1: a step reads the model's active weights and its KV once, plus each
    MTP module's weights (their KV is ignored); it emits ``mtp_expected_tokens`` tokens."""
    main = am.decode_bytes(spec, context, weight_bytes, kv_bytes)["total"]
    extra = depth * mtp_module_active(spec) * weight_bytes
    tokens = mtp_expected_tokens(acceptance, depth)
    step = main + extra
    return {"tokens_per_step": tokens, "bytes_per_step": step, "bytes_per_token": step / tokens,
            "speedup": tokens * main / step}


def looped_kv_bytes(spec: Spec, context: float, kv_bytes: float, per_loop_cache: bool) -> float:
    """KV of a looped model: one cache per pass through the stack (``per_loop_cache``), or one shared cache
    (Ouro's decode-time reuse, arXiv:2510.25741 section 5.4.2), as ``arch_model.kv_cache`` counts it."""
    kv = am.kv_cache(spec, context, kv_bytes)["kv_bytes"]
    loops = am._f(spec.get("loops") or 1)
    return loops * kv if per_loop_cache else kv
