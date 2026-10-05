"""Unit tests for every closed form in reference/arch_model.py, plus the
checks of the modelled parameter counts against what the authors state and
against the weights each open model actually publishes."""

from __future__ import annotations

import math

import pytest

import arch_model as am
from helpers import by_id, models, snapshot, spec

# --- tiny hand-made specs ---------------------------------------------------


def gqa(d=64, h=4, kv=2, hd=16, layers=2, vocab=100, d_ff=128, window=0, tied=False, **mix):
    m = {"type": "attn", "heads": h, "kv_heads": kv, "head_dim": hd, **mix}
    if window:
        m["window"] = window
    return {
        "kind": "decoder", "d_model": d, "vocab": vocab, "tied_embeddings": tied, "norms_per_layer": 2,
        "mixers": {"full": m}, "ffns": {"dense": {"type": "dense", "d_ff": d_ff, "gated": True}},
        "layout": [{"mixer": "full", "ffn": "dense", "n": layers}],
    }


def test_sum_min_closed_form_matches_loop():
    for n in (1, 5, 17, 100):
        for w in (0, 1, 4, 17, 300):
            loop = sum(min(t, w) if w > 0 else t for t in range(1, n + 1))
            assert am.sum_min(float(n), float(w)) == loop


def test_attention_params_by_hand():
    d, h, kv, hd = 64.0, 4.0, 2.0, 16.0
    m = {"type": "attn", "heads": 4, "kv_heads": 2, "head_dim": 16}
    assert am.mixer_params(m, d) == d * h * hd + 2 * d * kv * hd + h * hd * d
    assert am.mixer_params({**m, "bias": True}, d) == am.mixer_params(m, d) + h * hd + 2 * kv * hd
    assert am.mixer_params({**m, "qk_norm": True}, d) == am.mixer_params(m, d) + 2 * hd
    assert am.mixer_params({**m, "gate": "elementwise"}, d) == am.mixer_params(m, d) + d * h * hd
    assert am.mixer_params({**m, "gate": "headwise"}, d) == am.mixer_params(m, d) + d * h
    assert am.mixer_params({**m, "k_eq_v": True}, d) == am.mixer_params(m, d) - d * kv * hd
    assert am.mixer_params(m, d, kv_shared=True) == d * h * hd + h * hd * d


def test_mla_params_by_hand():
    d = 100.0
    m = {"type": "mla", "heads": 2, "q_lora_rank": 8, "kv_lora_rank": 6, "qk_nope": 4, "qk_rope": 2, "v_head_dim": 3}
    q = d * 8 + 8 + 8 * 2 * 6
    kv = d * (6 + 2) + 6 + 6 * 2 * (4 + 3)
    o = 2 * 3 * d
    assert am.mixer_params(m, d) == q + kv + o
    no_q = {**m, "q_lora_rank": None}
    assert am.mixer_params(no_q, d) == d * 2 * 6 + kv + o
    ix = {**m, "indexer": {"heads": 3, "head_dim": 5, "topk": 7}}
    assert am.mixer_params(ix, d) == q + kv + o + 8 * 3 * 5 + d * 5 + d * 3


def test_csa_params_by_hand():
    d = 50.0
    m = {"type": "csa", "heads": 4, "head_dim": 8, "q_lora_rank": 6, "o_lora_rank": 5, "o_groups": 2, "window": 3}
    assert am.mixer_params(m, d) == d * 6 + 6 * 4 * 8 + d * 8 + 4 * 8 * 5 + 2 * 5 * d


def test_recurrent_mixer_params_by_hand():
    d = 32.0
    dn = {"type": "deltanet", "k_heads": 2, "v_heads": 4, "k_head_dim": 3, "v_head_dim": 5, "conv_kernel": 4}
    qk, v = 6.0, 20.0
    exp = d * (2 * qk + v) + d * v + d * 2 * 4 + (2 * qk + v) * 4 + v * d + 2 * 4
    assert am.mixer_params(dn, d) == exp
    mb = {"type": "mamba2", "heads": 4, "head_dim": 8, "state": 16, "groups": 1, "conv_kernel": 4}
    di = 32.0
    assert am.mixer_params(mb, d) == d * (2 * di + 2 * 16 + 4) + (di + 2 * 16) * 4 + 3 * 4 + di + di * d
    assert am.mixer_params({"type": "conv", "kernel": 3}, d) == 3 * d * d + 3 * d + d * d
    assert am.mixer_params({"type": "rwkv", "mats": 4, "head_dim": 32}, d) == 4 * d * d
    assert am.mixer_params({"type": "linear", "heads": 2, "head_dim": 4}, d) == 5 * d * 2 * 4
    assert am.mixer_params({"type": "none"}, d) == 0.0
    with pytest.raises(ValueError):
        am.mixer_params({"type": "bogus"}, d)


def test_ffn_params_by_hand():
    d = 10.0
    assert am.ffn_params({"type": "dense", "d_ff": 7, "gated": True}, d) == (3 * d * 7, 3 * d * 7)
    assert am.ffn_params({"type": "dense", "d_ff": 7, "gated": False, "bias": True}, d) == (2 * d * 7 + 7 + d,) * 2
    t, a = am.ffn_params({"type": "moe", "experts": 8, "active": 2, "d_expert": 5, "shared": 1, "d_shared": 6}, d)
    assert t == 8 * 3 * d * 5 + d * 8 + 3 * d * 6
    assert a == 2 * 3 * d * 5 + d * 8 + 3 * d * 6
    t, a = am.ffn_params({"type": "moe", "experts": 4, "active": 1, "d_expert": 5, "latent": 3, "gated": False}, d)
    assert t == 4 * 2 * 3 * 5 + d * 4 + 2 * d * 3
    assert a == 1 * 2 * 3 * 5 + d * 4 + 2 * d * 3
    assert am.ffn_params({"type": "none"}, d) == (0.0, 0.0)


def test_whole_model_params_by_hand():
    s = gqa()
    d, v = 64.0, 100.0
    attn = d * 4 * 16 + 2 * d * 2 * 16 + 4 * 16 * d
    ffn = 3 * d * 128
    p = am.params(s)
    assert p.total == v * d * 2 + 2 * (attn + 2 * d + ffn)
    assert p.norms == 2 * 2 * d
    assert p.active == p.total
    assert p.non_embedding_total == 2 * (attn + 2 * d + ffn)
    assert p.matmul_active == p.total - v * d - p.norms
    tied = am.params(gqa(tied=True))
    assert tied.total == p.total - v * d
    assert tied.matmul_active == tied.total - tied.norms


def test_chunked_closed_forms_match_loop():
    for c in (1, 4, 17):
        for n in (1, 3, 4, 5, 16, 17, 18, 100):
            loop = [(t - 1) % c + 1 for t in range(1, n + 1)]
            assert am.in_chunk(float(n), float(c)) == loop[-1]
            assert am.sum_chunked(float(n), float(c)) == sum(loop)


def test_chunked_attention_by_hand():
    """A chunked layer reserves one chunk of cache, and the query at position
    t reads only the tokens of its own chunk up to itself."""
    s = gqa(layers=1, chunk=4)
    per_entry = 2 * 4 * (16 + 16)
    assert am.kv_cache(s, 10.0, 2.0)["kv_bytes"] == 2 * 32 * 4 * 2
    assert am.kv_cache(s, 3.0, 2.0)["kv_bytes"] == 2 * 32 * 3 * 2
    assert am.kv_cache(s, 10.0, 2.0)["bytes_per_token_unbounded"] == 0
    base = 2 * am.params(s).matmul_active
    assert am.decode_flops(s, 8.0) == base + 4 * per_entry
    assert am.decode_flops(s, 9.0) == base + 1 * per_entry
    assert am.decode_bytes(s, 10.0, 0.0, 1.0)["kv"] == 2 * 32 * 2
    layer = am.layer_matmul(s, s["layout"][0])
    assert am.prefill_flops(s, 10.0) == 2 * layer * 10 + (10 + 10 + 3) * per_entry + 2 * am.params(s).lm_head


def test_kv_cache_gqa_and_window():
    s = gqa(kv=2, hd=16, layers=3)
    kv = am.kv_cache(s, 100.0, 2.0)
    assert kv["kv_bytes"] == 3 * 2 * (16 + 16) * 100 * 2
    assert kv["bytes_per_token_unbounded"] == 3 * 2 * 32 * 2
    w = am.kv_cache(gqa(kv=2, hd=16, layers=3, window=10), 100.0, 2.0)
    assert w["kv_bytes"] == 3 * 2 * 32 * 10 * 2
    assert w["bytes_per_token_unbounded"] == 0


def test_kv_cache_mla_dsa_and_csa():
    s = gqa()
    s["mixers"] = {"mla": {"type": "mla", "heads": 4, "q_lora_rank": None, "kv_lora_rank": 512, "qk_nope": 128, "qk_rope": 64, "v_head_dim": 128}}
    s["layout"] = [{"mixer": "mla", "ffn": "dense", "n": 2}]
    assert am.kv_cache(s, 10.0, 1.0)["kv_bytes"] == 2 * (512 + 64) * 10
    s["mixers"]["mla"]["indexer"] = {"heads": 2, "head_dim": 128, "topk": 4}
    assert am.kv_cache(s, 10.0, 1.0)["kv_bytes"] == 2 * (512 + 64 + 128) * 10
    s["layout"] = [{"mixer": "mla", "ffn": "dense", "n": 1, "indexer": False}, {"mixer": "mla", "ffn": "dense", "n": 1}]
    assert am.kv_cache(s, 10.0, 1.0)["kv_bytes"] == (576 + 704) * 10
    c = gqa()
    c["mixers"] = {"csa": {"type": "csa", "heads": 4, "head_dim": 512, "q_lora_rank": 8, "o_lora_rank": 8, "o_groups": 2, "window": 128,
                           "indexer": {"heads": 2, "head_dim": 128, "topk": 16}}}
    c["layout"] = [{"mixer": "csa", "ffn": "dense", "n": 1, "ratio": 4, "indexer": True},
                   {"mixer": "csa", "ffn": "dense", "n": 1, "ratio": 128, "indexer": False},
                   {"mixer": "csa", "ffn": "dense", "n": 1, "ratio": 2, "kv_source": False},
                   {"mixer": "csa", "ffn": "dense", "n": 1, "ratio": 0}]
    kv = am.kv_cache(c, 1024.0, 1.0)
    assert kv["kv_bytes"] == (512 / 4 + 128 / 4 + 512 / 128) * 1024
    assert kv["state_bytes"] == 4 * 128 * 512 * 2.0


def test_state_sizes():
    s = gqa()
    s["mixers"] = {"dn": {"type": "deltanet", "k_heads": 2, "v_heads": 4, "k_head_dim": 3, "v_head_dim": 5, "conv_kernel": 4},
                   "mb": {"type": "mamba2", "heads": 4, "head_dim": 8, "state": 16, "groups": 1, "conv_kernel": 4}}
    s["layout"] = [{"mixer": "dn", "ffn": "dense", "n": 1}, {"mixer": "mb", "ffn": "dense", "n": 1}]
    kv = am.kv_cache(s, 1e6, 2.0, 4.0)
    assert kv["kv_bytes"] == 0
    dn = 4 * 3 * 5 + (2 * 2 * 3 + 4 * 5) * 3
    mb = 4 * 8 * 16 + (32 + 32) * 3
    assert kv["state_bytes"] == (dn + mb) * 4.0


def test_decode_and_prefill_flops_by_hand():
    s = gqa(layers=1)
    p = am.params(s)
    per_entry = 2 * 4 * (16 + 16)
    assert am.decode_flops(s, 10.0) == 2 * p.matmul_active + 10 * per_entry
    layer = am.layer_matmul(s, s["layout"][0])
    head = p.lm_head
    assert am.prefill_flops(s, 10.0) == 2 * layer * 10 + 55 * per_entry + 2 * head
    w = gqa(layers=1, window=4)
    assert am.decode_flops(w, 10.0) == 2 * am.params(w).matmul_active + 4 * per_entry


def test_ced_prefill_runs_encoder_plus_replay():
    s = gqa(layers=4)
    s["layout"] = [{"mixer": "full", "ffn": "dense", "n": 4}]
    s["kind"] = "ced"
    s["ced_encoder_layers"] = 2
    s["ced_window"] = 8
    full = am.prefill_flops({**s, "kind": "decoder"}, 100.0)
    ced = am.prefill_flops(s, 100.0)
    runs = am.expand_layout(s)
    enc = am._layers_flops_prefill(s, runs[:2], 100.0)
    replay = am._layers_flops_prefill(s, runs[2:], 8.0)
    kvp = 2 * (2 * 2 * 32 * 64.0) * 100
    assert ced == enc + kvp + replay + 2 * am.params(s).lm_head
    assert ced < full
    assert am.decode_flops(s, 100.0) == am.decode_flops({**s, "kind": "decoder"}, 100.0)


def test_loops_multiply_the_stack():
    s = gqa(layers=2)
    one = am.decode_flops(s, 16.0)
    s4 = {**s, "loops": 4}
    p = am.params(s)
    stack = 2 * (p.matmul_active - p.lm_head)
    attn = one - 2 * p.matmul_active
    assert am.decode_flops(s4, 16.0) == 2 * p.matmul_active + 3 * stack + 4 * attn


def test_decode_bytes_by_hand():
    s = gqa(layers=2, window=0)
    p = am.params(s)
    b = am.decode_bytes(s, 50.0, 2.0, 1.0)
    assert b["weights"] == (p.active - p.embedding) * 2.0
    assert b["kv"] == 2 * 2 * 32 * 50 * 1.0
    assert b["total"] == b["weights"] + b["kv"]


def test_encoder_decoder_counts_cross_attention():
    s = gqa(layers=2)
    s["kind"] = "encoder-decoder"
    s["encoder_layout"] = [{"mixer": "full", "ffn": "dense", "n": 3}]
    p = am.params(s)
    attn = am.mixer_params(s["mixers"]["full"], 64.0)
    assert p.cross_attention == 2 * attn
    assert p.norms == (2 + 3) * 2 * 64.0 + 2 * 64.0
    kv = am.kv_cache(s, 10.0, 1.0)
    assert kv["kv_bytes"] == 2 * 64 * 10 + 2 * 64 * 10


# --- the real models --------------------------------------------------------

WITH_ARCH = [m for m in models() if m["arch"] is not None]


@pytest.mark.parametrize("m", WITH_ARCH, ids=[m["id"] for m in WITH_ARCH])
def test_prefill_equals_sum_of_decode_steps(m):
    """Prefilling n tokens costs the same as decoding them one by one, less
    the output head on all but the last (prefill needs one set of logits)."""
    s = spec(m)
    if s["kind"] in ("ced", "encoder-decoder", "encoder") or s.get("loops"):
        pytest.skip("prefill differs from decode by design")
    n = 64
    p = am.params(s)
    head = p.lm_head if not s.get("tied_embeddings") else p.embedding
    total = sum(am.decode_flops(s, float(t)) for t in range(1, n + 1)) - (n - 1) * 2 * head
    assert math.isclose(am.prefill_flops(s, float(n)), total, rel_tol=1e-12)


# Documented exceptions to the parameter checks (see README, "Parameter checks").
TOTAL_EXCEPTIONS = {
    "chinchilla": "the paper's 70B is not reproduced by its own Table 4 shape (64.95B); difference unexplained",
}
MULTIMODAL_FLOOR = 0.85  # text stack only; vision/audio encoders are not modelled


@pytest.mark.parametrize("m", WITH_ARCH, ids=[m["id"] for m in WITH_ARCH])
def test_total_matches_what_the_authors_state(m):
    stated = m["facts"]["total_params"]["v"]
    if stated is None:
        pytest.skip("no total stated")
    if m["id"] in TOTAL_EXCEPTIONS:
        pytest.skip(TOTAL_EXCEPTIONS[m["id"]])
    p = am.params(spec(m))
    near = lambda x: abs(x / stated - 1) <= 0.05  # noqa: E731  rounding of the stated figure
    if m.get("multimodal"):
        assert MULTIMODAL_FLOOR * stated <= p.total <= 1.05 * stated
    else:
        assert near(p.total) or near(p.non_embedding_total), (p.total, p.non_embedding_total, stated)


@pytest.mark.parametrize("m", WITH_ARCH, ids=[m["id"] for m in WITH_ARCH])
def test_active_matches_what_the_authors_state(m):
    """Labs differ on whether 'active' counts the embedding and output head,
    so the stated figure must fall between the two conventions (+-10%)."""
    stated = m["facts"]["active_params"]["v"]
    if stated is None:
        pytest.skip("no active count stated")
    if m["id"] in ACTIVE_EXCEPTIONS:
        pytest.skip(ACTIVE_EXCEPTIONS[m["id"]])
    p = am.params(spec(m))
    assert 0.9 * p.non_embedding_active <= stated <= 1.1 * p.active, (p.non_embedding_active, p.active, stated)


def llama4_vision_params(vc: dict, text_d: int) -> int:
    """Weights of Llama 4's vision encoder, adapter and projector, from its
    vision_config, following transformers 5.18.0 Llama4VisionModel and
    Llama4MultiModalProjector."""
    h, i, L, ps = vc["hidden_size"], vc["intermediate_size"], vc["num_hidden_layers"], vc["patch_size"]
    layer = 4 * (h * h + h) + (h * i + i) + (i * h + h) + 2 * 2 * h  # q/k/v/o with biases, fc1/fc2, two LayerNorms
    patches = (vc["image_size"] // ps) ** 2 + 1
    embed = vc["num_channels"] * ps * ps * h + h + patches * h  # unfold conv, class embedding, positions
    adapter = i * vc["projector_input_dim"] + vc["projector_output_dim"] * vc["projector_output_dim"]
    projector = vc["vision_output_dim"] * text_d
    return L * layer + embed + 2 * 2 * h + adapter + projector  # + pre and post LayerNorms


def test_llama4_maverick_matches_published_weights_exactly():
    """Multimodal models skip the weights check below; for Llama 4 Maverick
    the vision tower is small and fully specified, so the text stack the cost
    model counts, plus the final norm (which the cost model leaves out), plus
    the vision tower must equal the safetensors count exactly."""
    m = by_id("llama-4-maverick")
    p = am.params(spec(m))
    cfg = snapshot(m)["config"]
    d = cfg["text_config"]["hidden_size"]
    total = int(p.total) + d + llama4_vision_params(cfg["vision_config"], d)
    assert total == m["checks"]["hf_weight_count"]["v"] == 401_583_781_376
    assert int(p.total) == 400_711_843_840
    assert int(p.active) == 17_184_686_080


HF_EXCEPTIONS = {
    "gpt-2-xl": "the checkpoint also stores the causal-mask buffers (attn.bias, 48 x 1024 x 1024)",
    "falcon-7b": "the checkpoint stores lm_head separately although the code ties it",
    "falcon-40b": "the checkpoint stores lm_head separately although the code ties it",
    "bert-large": "the checkpoint holds the pooler and the MLM head transform",
    "antares-1b": "the card's backbone is Granite 4.0 1B; the published weights also hold an untied output head",
    "zaya1-8b": "CCA attention is approximated as GQA",
    "kimi-linear-48b-a3b": "KDA gate projections are approximated",
    "solar-open-2": "KDA gate projections are approximated",
    "longcat-flash-lite": "the n-gram embedding table is approximated",
    "motif-3-beta": "GDLA attention is approximated as MLA",
    "glm-4.5-air": "the multi-token-prediction layer's weights are approximated",
}
ACTIVE_EXCEPTIONS = {
    "longcat-flash-lite": "zero (identity) experts make the number of real experts per token input-dependent",
}


@pytest.mark.parametrize("m", WITH_ARCH, ids=[m["id"] for m in WITH_ARCH])
def test_total_matches_published_weights(m):
    """Against the parameter count of the published safetensors at the
    pinned revision: the tightest check there is. Multi-token-prediction
    layers count as weights there, so either side may include them."""
    hw = m["checks"].get("hf_weight_count")
    if not hw or not hw["v"] or hw["packed"] or m.get("multimodal") or "arch_from" in m:
        pytest.skip("no comparable weight count")
    p = am.params(spec(m))
    ratio = min(abs(p.total / hw["v"] - 1), abs((p.total + p.mtp) / hw["v"] - 1))
    tol = 0.05 if m["id"] in HF_EXCEPTIONS else 0.005
    assert ratio <= tol, (m["id"], p.total, p.mtp, hw["v"])
