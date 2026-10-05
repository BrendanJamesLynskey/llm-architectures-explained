"""Facts that a config.json does not state directly: where the norms sit,
whether queries and keys are normalised, and how positions are encoded.

Each rule names its evidence. ``TF`` refers to the modelling code of the
transformers release pinned in hf_adapter.TF; ``repo`` to the model's own
modelling file at its pinned revision (`auto_map` in its config). The
placement names are:

* ``pre``      : a norm before attention and before the MLP (GPT-2 onwards)
* ``post``     : norms after attention and after the MLP, inside the residual
                 (OLMo 2/3's re-ordered norm)
* ``sandwich`` : both a pre-norm and a post-norm around each sub-block
* ``parallel`` : one norm feeds attention and the MLP, which run in parallel
* ``post-ln``  : the original Transformer's norm after each residual add
"""

from __future__ import annotations

from typing import Any

from hf_adapter import TF, F, code

# model_type -> (placement, norm type, evidence)
NORMS: dict[str, tuple[str, str, str]] = {
    "gpt2": ("pre", "LayerNorm", f"{TF} gpt2: ln_1 before attention, ln_2 before the MLP"),
    "bert": ("post-ln", "LayerNorm", f"{TF} bert: LayerNorm after each residual add"),
    "t5": ("pre", "RMSNorm", f"{TF} t5: layer_norm before each sub-layer (T5LayerNorm, no bias or mean)"),
    "switch_transformers": ("pre", "RMSNorm", f"{TF} switch_transformers: T5-style pre-norm"),
    "bloom": ("pre", "LayerNorm", f"{TF} bloom: input_layernorm and post_attention_layernorm (pre-MLP), plus an embedding LayerNorm"),
    "opt": ("pre", "LayerNorm", f"{TF} opt: do_layer_norm_before: true"),
    "gptj": ("parallel", "LayerNorm", f"{TF} gptj: one ln_1 feeds attention and the MLP in parallel"),
    "gpt_neox": ("parallel", "LayerNorm", f"{TF} gpt_neox: use_parallel_residual, separate input and post-attention norms"),
    "falcon": ("parallel", "LayerNorm", "repo modeling_falcon.py: parallel attention and MLP"),
    "cohere2": ("parallel", "LayerNorm", f"{TF} cohere2: a single input_layernorm feeds attention and the MLP"),
    "cohere2_moe": ("parallel", "LayerNorm", f"{TF} cohere2_moe: a single input_layernorm feeds attention and the MoE"),
    "palm": ("parallel", "LayerNorm", "PaLM paper §2: parallel layers"),
    "olmo2": ("post", "RMSNorm", f"{TF} olmo2: post_attention_layernorm and post_feedforward_layernorm only"),
    "olmo3": ("post", "RMSNorm", f"{TF} olmo3: post_attention_layernorm and post_feedforward_layernorm only"),
    "gemma3_text": ("sandwich", "RMSNorm", "google-deepmind/gemma _gemma.py: use_post_attn_norm and use_post_ffw_norm"),
    "gemma4_text": ("sandwich", "RMSNorm", f"{TF} gemma4: pre and post norms around attention and MLP"),
    "gemma4_unified_text": ("sandwich", "RMSNorm", f"{TF} gemma4: pre and post norms around attention and MLP"),
    "afmoe": ("sandwich", "RMSNorm", f"{TF} afmoe: pre_mlp_layernorm and post_mlp_layernorm around the MLP"),
    "muse_glimmer_text": ("sandwich", "RMSNorm", f"{TF} muse_glimmer: pre_feedforward and post_feedforward norms"),
    "ouro": ("sandwich", "RMSNorm", "repo modeling_ouro.py: input_layernorm(_2) and post_attention_layernorm(_2)"),
    "nemotron_h": ("pre", "RMSNorm", f"{TF} nemotron_h: one norm before each mixer or MLP block"),
    "mamba": ("pre", "RMSNorm", f"{TF} mamba: one norm before each block"),
    "mamba2": ("pre", "RMSNorm", f"{TF} mamba2: one norm before each block"),
    "mamba2_ssm": ("pre", "RMSNorm", "mamba_ssm: one norm before each block"),
    "rwkv": ("pre", "LayerNorm", f"{TF} rwkv: ln1 before time mixing, ln2 before channel mixing"),
    "rwkv6": ("pre", "LayerNorm", "repo modeling_rwkv6.py: ln1 and ln2 before each mixing block"),
    "xlstm": ("pre", "RMSNorm", f"{TF} xlstm: norm_mlstm and norm_ffn before each block"),
    "lfm2": ("pre", "RMSNorm", f"{TF} lfm2: operator_norm and ffn_norm"),
    "lfm2_moe": ("pre", "RMSNorm", f"{TF} lfm2_moe: operator_norm and ffn_norm"),
    "glam": ("pre", "LayerNorm", "GLaM paper: standard Transformer layers"),
    "chinchilla": ("pre", "RMSNorm", "Gopher paper §3: RMSNorm instead of LayerNorm"),
}
PRE_RMS = ("pre", "RMSNorm")

# model_types whose decoder layer was checked to hold exactly a norm before
# attention and a norm before the MLP (input_layernorm and
# post_attention_layernorm, or equivalent), in transformers 5.18.0 or in the
# repository's own modelling file at its pinned revision.
PRE_CHECKED = {
    "llama", "mistral", "qwen2", "qwen3", "phi3", "smollm3", "granite", "qwen3_moe", "qwen2_moe", "deepseek",
    "deepseek_v2", "deepseek_v3", "kimi_k2", "deepseek_v32", "deepseek_v4", "glm4_moe", "glm_moe_dsa", "gpt_oss",
    "qwen3_next", "qwen3_5_text", "qwen3_5_moe_text", "minimax_m2", "minimax_text_01", "minimax", "kimi_linear",
    "mimo_v2_flash", "mimo_v2", "naive_n05_flash", "mistral4", "laguna", "zaya", "hy_v3", "hy_v4", "mellum",
    "solar_open2", "mixtral", "llama4_text", "olmoe", "sarvam_moe", "sarvam_mla", "step3p5", "nanbeige", "Motif", "bailing_hybrid",
    "glm5_next_text", "granitemoehybrid", "jamba", "afmoe",
}

# model_types whose attention normalises queries and keys (QK-norm)
QK_NORM: dict[str, str] = {
    "qwen3": f"{TF} qwen3: q_norm and k_norm",
    "qwen3_moe": f"{TF} qwen3_moe: q_norm and k_norm",
    "qwen3_next": f"{TF} qwen3_next: q_norm and k_norm",
    "qwen3_5_text": f"{TF} qwen3_5: q_norm and k_norm",
    "qwen3_5_moe_text": f"{TF} qwen3_5_moe: q_norm and k_norm",
    "qwen4_exp_text": f"{TF} qwen4_exp: norm_query and norm_key",
    "olmo2": f"{TF} olmo2: q_norm and k_norm",
    "olmo3": f"{TF} olmo3: q_norm and k_norm",
    "olmoe": f"{TF} olmoe: q_norm and k_norm",
    "gemma4_text": f"{TF} gemma4: q_norm and k_norm",
    "gemma4_unified_text": f"{TF} gemma4: q_norm and k_norm",
    "afmoe": f"{TF} afmoe: q_norm and k_norm",
    "mellum": f"{TF} mellum: q_norm and k_norm",
    "laguna": "repo modeling_laguna.py: q/k norms",
    "lfm2": f"{TF} lfm2: q_layernorm and k_layernorm",
    "lfm2_moe": f"{TF} lfm2_moe: q_layernorm and k_layernorm",
    "nanbeige": "repo modeling_nanbeige.py: q_norm and k_norm",
    "step3p5": "config use_qk_norm: true",
    "sarvam_moe": "config use_qk_norm: true",
}
QK_FROM_CONFIG = ("use_qk_norm", "qk_norm", "add_qk_norm")


def norm_facts(mt: str, cfg: dict[str, Any] | None, is_mla: bool, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if mt in NORMS:
        pl, typ, ev = NORMS[mt]
    elif mt == "transformer":
        pl, typ, ev = "post-ln", "LayerNorm", "Attention Is All You Need §3.1: LayerNorm(x + Sublayer(x))"
    elif mt in PRE_CHECKED:
        pl, typ = PRE_RMS
        ev = f"{TF} / repo modelling code for {mt}: input_layernorm before attention, post_attention_layernorm before the MLP"
    else:
        pl = typ = ev = None
    if pl is None:
        out["norm_placement"] = F(None, "not-disclosed", note="not checked in the modelling code")
        out["norm_type"] = F(None, "not-disclosed", note="not checked in the modelling code")
    else:
        out["norm_placement"] = code(pl, ev)
        out["norm_type"] = code(typ, ev)
    qk = None
    c = cfg or {}
    for k in QK_FROM_CONFIG:
        if k in c and isinstance(c[k], bool):
            qk = F(c[k], "config", "hf", prefix + k)
            break
    if qk is None and mt in QK_NORM:
        qk = code(True, QK_NORM[mt])
    if qk is None and mt == "gemma3_text":
        qk = F(bool(c.get("use_qk_norm")), "disclosed", "gemma-github", "_gemma.py: use_qk_norm")
    if qk is None:
        qk = code(False, "no q/k normalisation in the attention block" + (" (MLA normalises its latent vectors, which is not QK-norm)" if is_mla else ""))
    out["qk_norm"] = qk
    return out


def position_facts(mt: str, cfg: dict[str, Any] | None, arch: dict[str, Any] | None, prefix: str = "") -> dict[str, Any]:
    """Positional encoding: {scheme, rope_fraction, nope}."""
    c = cfg or {}
    tc = c.get("text_config") if isinstance(c.get("text_config"), dict) else c

    def pf(v: dict[str, Any], st: str, ref: str, src: str = "code") -> dict[str, Any]:
        if st == "config":
            return F(v, st, "hf", ", ".join(prefix + r.strip() if not r.strip().startswith(("qk_", "rotary_dim /")) else r.strip() for r in [ref]))
        return F(v, st, src, ref)

    if mt in ("gpt2", "opt", "bert"):
        return {"position": pf({"scheme": "learned", "rope_fraction": None, "nope": None}, "code", f"{TF} {mt}: learned absolute position embeddings")}
    if mt == "bloom":
        return {"position": pf({"scheme": "alibi", "rope_fraction": None, "nope": None}, "code", f"{TF} bloom: ALiBi attention biases")}
    if mt in ("t5", "switch_transformers"):
        return {"position": pf({"scheme": "relative-bias", "rope_fraction": None, "nope": None}, "code", f"{TF} t5: bucketed relative position bias")}
    if mt in ("mamba", "mamba2", "mamba2_ssm", "rwkv", "rwkv6", "xlstm"):
        return {"position": pf({"scheme": "none", "rope_fraction": None, "nope": "all layers: the recurrence carries order"}, "code", f"{TF} {mt}: no positional encoding")}
    if mt == "jamba":
        return {"position": pf({"scheme": "none", "rope_fraction": None, "nope": "attention layers have no positional encoding; the Mamba layers carry order"}, "paper", "Jamba paper (arXiv 2403.19887) §2: no explicit positional information", "paper")}
    frac = 1.0
    ref = "rotary on the full head (default)"
    st = "code"
    for k in ("partial_rotary_factor", "rotary_pct"):
        if isinstance(tc.get(k), (int, float)):
            frac = float(tc[k])
            ref, st = k, "config"
            break
    rp = tc.get("rope_parameters")
    if isinstance(rp, dict) and isinstance(rp.get("partial_rotary_factor"), (int, float)):
        frac, ref, st = float(rp["partial_rotary_factor"]), "rope_parameters.partial_rotary_factor", "config"
    if isinstance(rp, dict) and isinstance(rp.get("full_attention"), dict) and "partial_rotary_factor" in rp["full_attention"]:
        frac, ref, st = float(rp["full_attention"]["partial_rotary_factor"]), "rope_parameters.full_attention.partial_rotary_factor (global layers)", "config"
    if isinstance(tc.get("rotary_dim"), int) and isinstance(tc.get("head_dim"), int) and tc["head_dim"]:
        frac, ref, st = tc["rotary_dim"] / tc["head_dim"], "rotary_dim / head_dim", "config"
    if arch and "csa" in (arch.get("mixers") or {}) and isinstance(tc.get("qk_rope_head_dim"), int):
        frac, ref, st = tc["qk_rope_head_dim"] / tc["head_dim"], "qk_rope_head_dim / head_dim", "config"
    if arch and "mla" in (arch.get("mixers") or {}):
        m = arch["mixers"]["mla"]
        nope, rope = m["qk_nope"]["v"], m["qk_rope"]["v"]
        if nope + rope:
            frac, ref, st = rope / (nope + rope), "qk_rope_head_dim / (qk_nope_head_dim + qk_rope_head_dim): decoupled RoPE", "config"
    scheme = "rope"
    if isinstance(rp, dict) and rp.get("mrope_section"):
        scheme = "mrope"
    nope = None
    if tc.get("no_rope_layer_interval"):
        nope = f"every {tc['no_rope_layer_interval']}th layer has no RoPE"
        ref, st = "no_rope_layer_interval", "config"
    if isinstance(tc.get("layer_rope_theta"), list) and 0 in tc["layer_rope_theta"]:
        nope = "global (full-attention) layers have no RoPE"
        ref, st = "layer_rope_theta (0 on global layers)", "config"
    if mt == "llama4_text" and not tc.get("no_rope_layer_interval"):
        nope = "every 4th layer has no RoPE (global attention); the others use RoPE within their attention chunk"
        ref, st = f"{TF} llama4: no_rope_layer_interval default 4; RoPE layers are chunked_attention", "code"
    if mt in ("cohere2", "cohere2_moe"):
        nope = "full-attention layers have no RoPE; sliding-window layers use it"
        ref, st = f"{TF} {mt}: RoPE applied only when the layer has a sliding window", "code"
    if tc.get("mla_use_nope"):
        nope = "MLA layers use no RoPE; the linear-attention layers carry order"
        ref, st = "mla_use_nope", "config"
    if tc.get("use_rope") is False:
        nope = "attention layers use no RoPE; the linear-attention layers carry order"
        ref, st = "use_rope: false", "config"
    if mt == "glm5_next_text" and tc.get("qk_rope_head_dim") == 0:
        frac = 0.0
    return {"position": pf({"scheme": scheme, "rope_fraction": round(frac, 4), "nope": nope}, st, ref)}
