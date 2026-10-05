"""Every closed form and builder in reference/chapter_model.py, against an
independent statement of the same thing (the formulas the chapters print)."""

from __future__ import annotations

import math

import pytest

import arch_model as am
import chapter_model as cm

BASE = {
    "kind": "decoder", "d_model": 4096, "vocab": 128256, "tied_embeddings": False,
    "mixers": {"full": {"type": "attn", "heads": 32, "kv_heads": 8, "head_dim": 128}},
    "ffns": {"dense": {"type": "dense", "d_ff": 14336, "gated": True}},
    "layout": [{"mixer": "full", "ffn": "dense", "n": 32}],
}


def test_rotary_dims_round_down_to_even():
    assert cm.rotary_dims(128, 1.0) == 128
    assert cm.rotary_dims(128, 0.25) == 32
    assert cm.rotary_dims(80, 0.3) == 24
    assert cm.rotary_dims(64, 0.5) == 32


def test_rope_wavelengths_match_the_definition():
    w = cm.rope_wavelengths(128, 1.0, 10000.0)
    assert len(w) == 64
    assert w[0] == pytest.approx(2 * math.pi)
    for i, x in enumerate(w):
        assert x == pytest.approx(2 * math.pi / 10000.0 ** (-2 * i / 128), rel=1e-12)
    assert all(b > a for a, b in zip(w, w[1:]))


def test_rope_long_pairs_counts_wavelengths_beyond_the_context():
    w = cm.rope_wavelengths(128, 1.0, 1e6)
    for c in (1024.0, 32768.0, 1048576.0):
        assert cm.rope_long_pairs(128, 1.0, 1e6, c) == sum(x > c for x in w)
    # a larger base leaves more pairs slower than the same context
    assert cm.rope_long_pairs(128, 1.0, 1e6, 32768.0) > cm.rope_long_pairs(128, 1.0, 1e4, 32768.0)


@pytest.mark.parametrize("g", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("layers", [1, 6, 32])
def test_norm_profiles_match_the_closed_forms(g, layers):
    k_total = 2 * layers
    pre = cm.norm_profile("pre-norm", layers, g)
    post = cm.norm_profile("post-ln", layers, g)
    out = cm.norm_profile("output-norm", layers, g, gamma=0.5)
    for k in range(1, k_total + 1):
        assert pre["rms"][k - 1] == pytest.approx(math.sqrt(1 + k * g * g))
        assert pre["update"][k - 1] == pytest.approx(g / math.sqrt(1 + (k - 1) * g * g))
        assert post["rms"][k - 1] == 1.0
        assert post["update"][k - 1] == pytest.approx(g)
        assert out["rms"][k - 1] == pytest.approx(math.sqrt(1 + k * 0.25))
    assert pre["embedding_share"] == pytest.approx((1 + k_total * g * g) ** -0.5)
    assert post["embedding_share"] == pytest.approx((1 + g * g) ** (-k_total / 2))
    assert out["embedding_share"] == pytest.approx((1 + k_total * 0.25) ** -0.5)


def test_output_norm_ignores_the_gain_and_unknown_placements_fail():
    assert cm.norm_profile("output-norm", 8, 0.5) == cm.norm_profile("output-norm", 8, 4.0)
    with pytest.raises(ValueError):
        cm.norm_profile("middle", 8, 1.0)


@pytest.mark.parametrize("a", [0.0, 0.5, 0.85, 0.99])
@pytest.mark.parametrize("depth", [0, 1, 3])
def test_mtp_expected_tokens_is_the_geometric_sum(a, depth):
    closed = depth + 1 if a == 1.0 else (1 - a ** (depth + 1)) / (1 - a)
    assert cm.mtp_expected_tokens(a, depth) == pytest.approx(closed)
    assert cm.mtp_expected_tokens(1.0, depth) == depth + 1


def test_mtp_speedup_is_tokens_over_relative_step_cost():
    s = dict(BASE, mtp_layers=1)
    main = am.decode_bytes(s, 8192.0, 2.0, 2.0)["total"]
    mod = cm.mtp_module_active(s)
    d = 4096.0
    layer = am.layer_params(s, s["layout"][-1])
    assert mod == layer[0] + layer[2] + 2 * d * d + 4 * d
    out = cm.mtp_speedup(s, 8192.0, 2, 0.8, 2.0, 2.0)
    assert out["tokens_per_step"] == pytest.approx(1 + 0.8 + 0.64)
    assert out["speedup"] == pytest.approx(2.44 * main / (main + 2 * mod * 2.0))
    assert out["bytes_per_token"] == pytest.approx(out["bytes_per_step"] / 2.44)


def test_moe_spec_parameter_counts():
    d, dff = 4096.0, 14336.0
    s = cm.moe_spec(BASE, 64, 6, 2, 8, 1)
    de = dff // 8
    p = am.params(s)
    dense = am.params(BASE)
    per_moe_total = 3 * d * (64 * de + 2 * de) + d * 64
    per_moe_active = 3 * d * (6 * de + 2 * de) + d * 64
    assert p.ffn_total == pytest.approx(3 * d * dff + 31 * per_moe_total)
    assert p.ffn_active == pytest.approx(3 * d * dff + 31 * per_moe_active)
    # 6 + 2 experts of an eighth of the width: the same active FFN as dense, plus routers
    assert p.ffn_active - dense.ffn_active == pytest.approx(31 * d * 64)
    assert cm.moe_spec(BASE, 1, 1, 0, 1, 0) == BASE


def test_depth_width_spec_shapes():
    s = cm.depth_width_spec(48, 3072)
    full = s["mixers"]["full"]
    assert (full["heads"], full["kv_heads"], full["head_dim"]) == (24, 8, 128)
    assert s["ffns"]["dense"]["d_ff"] == 10752 and s["ffns"]["dense"]["d_ff"] % 256 == 0
    assert cm.depth_width_spec(30, 512)["mixers"]["full"]["kv_heads"] == 4
    kv = am.kv_cache(s, 1.0, 2.0)["bytes_per_token_unbounded"]
    assert kv == 48 * 2 * 8 * 128 * 2.0
    assert am.kv_cache(cm.depth_width_spec(48, 8192), 1.0, 2.0)["bytes_per_token_unbounded"] == kv


def test_looping_keeps_parameters_and_scales_the_cache_choice():
    one, four = cm.looped_spec(BASE, 1), cm.looped_spec(BASE, 4)
    assert "loops" not in one and four["loops"] == 4
    assert am.params(one).total == am.params(four).total
    kv = am.kv_cache(one, 4096.0, 2.0)["kv_bytes"]
    assert cm.looped_kv_bytes(four, 4096.0, 2.0, True) == 4 * kv
    assert cm.looped_kv_bytes(four, 4096.0, 2.0, False) == kv
    p = am.params(one)
    attn = am.decode_flops(one, 4096.0) - 2 * p.matmul_active
    expected = 2 * p.matmul_active + 3 * 2 * (p.matmul_active - p.lm_head) + 4 * attn
    assert am.decode_flops(four, 4096.0) == pytest.approx(expected)


def test_attention_variants_change_only_the_mixer():
    vs = cm.attention_variants(BASE)
    assert [v["id"] for v in vs] == ["mha", "gqa", "mqa", "mla", "swa", "dsa", "deltanet", "mamba"]
    for v in vs:
        s = v["spec"]
        assert sum(r["n"] for r in s["layout"]) == 32
        assert s["ffns"] == BASE["ffns"] and s["d_model"] == 4096 and s["vocab"] == 128256
    kv = {v["id"]: am.kv_cache(v["spec"], 131072.0, 2.0)["bytes_per_token_unbounded"] for v in vs}
    assert kv["mha"] == 4 * kv["gqa"] == 32 * kv["mqa"]
    assert kv["mla"] == 32 * 576 * 2.0
    swa = next(v for v in vs if v["id"] == "swa")["spec"]
    assert [r["mixer"] for r in swa["layout"]][:2] == ["sliding", "full"]


def test_as_decoder_drops_the_split():
    ced = dict(BASE, kind="ced", ced_encoder_layers=16, ced_window=128)
    dec = cm.as_decoder(ced)
    assert dec["kind"] == "decoder" and "ced_encoder_layers" not in dec and "ced_window" not in dec
    assert am.prefill_flops(ced, 8192.0) < am.prefill_flops(dec, 8192.0)
    assert am.decode_flops(ced, 8192.0) == am.decode_flops(dec, 8192.0)
