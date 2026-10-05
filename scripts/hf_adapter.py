"""Read a Hugging Face ``config.json`` into this site's normalised architecture.

Every leaf it returns is a *field*: ``{"v": value, "st": status, "src":
source id, "ref": where in the source}``. Values read straight from the
config carry ``st: config`` and ``ref: <key path>``; the provenance test
(``tests/python/test_provenance.py``) re-reads each of those keys from the
pinned snapshot in ``data/hf/`` and checks the value. Values that need a
rule from the model's code (a default the config leaves out, or a layer
pattern) carry ``st: code`` and a ``ref`` naming the file and the rule.

One handler per ``model_type`` family. The handlers are deliberately
explicit rather than clever: each one says which keys it reads.
"""

from __future__ import annotations

import math
from typing import Any, Callable

TF = "transformers 5.18.0"  # the library version whose modelling code the `code` refs cite


class Missing(KeyError):
    pass


def F(v: Any, st: str, src: str | None = None, ref: str | None = None, note: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"v": v, "st": st}
    if src:
        out["src"] = src
    if ref:
        out["ref"] = ref
    if note:
        out["note"] = note
    return out


class Cfg:
    """A config with a key prefix (``text_config.`` for multimodal wrappers)."""

    def __init__(self, raw: dict[str, Any]):
        self.raw = raw
        self.prefix = ""
        self.c = raw
        for k in ("text_config", "llm_config", "language_config"):
            if isinstance(raw.get(k), dict) and ("hidden_size" in raw[k] or "num_hidden_layers" in raw[k]):
                self.c = raw[k]
                self.prefix = k + "."
                break

    def has(self, key: str) -> bool:
        return key in self.c and self.c[key] is not None

    def get(self, key: str, default: Any = None) -> Any:
        v = self.c.get(key)
        return default if v is None else v

    def f(self, key: str, *alts: str) -> dict[str, Any]:
        for k in (key, *alts):
            if self.has(k):
                return F(self.c[k], "config", "hf", self.prefix + k)
        raise Missing(f"{key} (or {alts}) not in config")

    def fopt(self, key: str, *alts: str) -> dict[str, Any] | None:
        try:
            return self.f(key, *alts)
        except Missing:
            return None

    def model_type(self) -> str:
        return str(self.c.get("model_type") or self.raw.get("model_type"))


def code(v: Any, ref: str, note: str | None = None) -> dict[str, Any]:
    return F(v, "code", "code", ref, note)


def rle(seq: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in seq:
        if out and {k: v for k, v in out[-1].items() if k != "n"} == item:
            out[-1]["n"] += 1
        else:
            out.append({**item, "n": 1})
    return out


def layout_field(per_layer: list[dict[str, Any]], st: str, ref: str, src: str = "hf") -> dict[str, Any]:
    return F(rle(per_layer), st, src if st in ("config", "code") else src, ref)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def attn_mixer(c: Cfg, *, heads: str = "num_attention_heads", kv: str = "num_key_value_heads",
               hd: str = "head_dim", window: Any = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    m: dict[str, Any] = {"type": F("attn", "code", "code", "attention block")}
    m["heads"] = c.f(heads)
    m["kv_heads"] = c.f(kv) if c.has(kv) else F(c.get(heads), "code", "code", f"{TF}: {kv} defaults to {heads}")
    if c.has(hd):
        m["head_dim"] = c.f(hd)
    else:
        d = c.get("hidden_size", c.get("n_embd"))
        m["head_dim"] = F(d // c.get(heads), "code", "code", f"{TF}: head_dim = hidden_size / {heads}")
    if window is not None:
        m["window"] = window
    if extra:
        m.update(extra)
    return m


def dense_ffn(c: Cfg, key: str = "intermediate_size", gated: bool = True, bias: bool = False) -> dict[str, Any]:
    out = {"type": F("dense", "code", "code", "MLP block"), "d_ff": c.f(key),
           "gated": code(gated, "MLP: gated (SwiGLU/GeGLU)" if gated else "MLP: two matrices, no gate")}
    if bias:
        out["bias"] = code(True, "MLP has biases")
    return out


def moe_ffn(c: Cfg, *, experts: tuple[str, ...], active: tuple[str, ...], d_expert: tuple[str, ...],
            shared: tuple[str, ...] | None = None, d_shared: tuple[str, ...] | None = None,
            gated: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {"type": F("moe", "code", "code", "MoE block")}
    out["experts"] = c.f(*experts)
    out["active"] = c.f(*active)
    out["d_expert"] = c.f(*d_expert)
    out["gated"] = code(gated, "experts are gated MLPs" if gated else "experts are two-matrix MLPs")
    if shared:
        sf = c.fopt(*shared)
        if sf is not None and sf["v"]:
            out["shared"] = sf
            if d_shared:
                ds = c.fopt(*d_shared)
                if ds is not None:
                    out["d_shared"] = ds
    return out


def base(c: Cfg) -> dict[str, Any]:
    a: dict[str, Any] = {}
    a["d_model"] = c.f("hidden_size", "n_embd", "d_model", "dim")
    a["vocab"] = c.f("vocab_size")
    if c.has("tie_word_embeddings"):
        a["tied_embeddings"] = c.f("tie_word_embeddings")
    else:
        a["tied_embeddings"] = code(False, f"{TF}: PretrainedConfig.tie_word_embeddings default", note="not in config")
    return a


def n_layers(c: Cfg) -> int:
    for k in ("num_hidden_layers", "n_layer", "num_layers", "n_layers", "num_blocks"):
        if c.has(k):
            return int(c.get(k))
    raise Missing("num_hidden_layers")


LT_MAP = {
    "full_attention": "full",
    "sliding_attention": "sliding",
    "chunked_attention": "full",
    "linear_attention": "linear",
    "deepseek_sparse_attention": "mla",
    "conv": "conv",
    "hybrid": "full",
    "attention": "full",
    "mamba": "mamba",
}


def from_layer_types(c: Cfg, ffn_for: Callable[[int], str], mapping: dict[str, str] | None = None) -> dict[str, Any]:
    mp = {**LT_MAP, **(mapping or {})}
    lts = c.get("layer_types")
    per = [{"mixer": mp[str(t)], "ffn": ffn_for(i)} for i, t in enumerate(lts)]
    return layout_field(per, "config", c.prefix + "layer_types")


def uniform(c: Cfg, mixer: str, ffn_for: Callable[[int], str], ref: str | None = None) -> dict[str, Any]:
    n = n_layers(c)
    per = [{"mixer": mixer, "ffn": ffn_for(i)} for i in range(n)]
    return layout_field(per, "config", ref or c.prefix + "num_hidden_layers")


def dense_prefix(k: int) -> Callable[[int], str]:
    return lambda i: "dense" if i < k else "moe"


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def llama_like(c: Cfg) -> dict[str, Any]:
    a = base(c)
    window = None
    if c.get("use_sliding_window") and c.has("sliding_window"):
        window = c.f("sliding_window")
    if c.model_type() == "mistral" and c.has("sliding_window"):
        window = c.f("sliding_window")
    extra = {}
    if c.get("attention_bias") or c.model_type() in ("qwen2",):
        extra["bias"] = code(True, f"{TF} {c.model_type()}: q/k/v projections have biases")
    if c.model_type() in ("qwen3", "olmo2", "olmo3", "ouro"):
        extra["qk_norm"] = code(True, f"{TF} {c.model_type()}: q_norm and k_norm")
    full = attn_mixer(c, window=window, extra=extra)
    a["mixers"] = {"full": full}
    a["ffns"] = {"dense": dense_ffn(c)}
    if c.has("layer_types") and "sliding_attention" in c.get("layer_types"):
        sl = attn_mixer(c, window=c.f("sliding_window"), extra=extra)
        a["mixers"]["sliding"] = sl
        a["layout"] = from_layer_types(c, lambda i: "dense")
    else:
        a["layout"] = uniform(c, "full", lambda i: "dense")
    return a


def gpt2(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["tied_embeddings"] = code(True, f"{TF} gpt2: lm_head tied to wte")
    d = c.get("n_embd")
    a["mixers"] = {"full": attn_mixer(c, heads="n_head", kv="n_head", extra={"bias": code(True, "c_attn/c_proj have biases")})}
    a["ffns"] = {"dense": {"type": F("dense", "code", "code", "MLP"),
                           "d_ff": code(4 * d, f"{TF} gpt2: n_inner defaults to 4 * n_embd"),
                           "gated": code(False, "GELU MLP, two matrices"), "bias": code(True, "MLP biases")}}
    a["layout"] = uniform(c, "full", lambda i: "dense", c.prefix + "n_layer")
    a["extra_embedding_params"] = code(c.get("n_positions") * d, "learned position embeddings: n_positions x n_embd")
    return a


def qwen_moe(c: Cfg) -> dict[str, Any]:
    a = base(c)
    extra = {"qk_norm": code(True, f"{TF} qwen3_moe: q_norm and k_norm")} if c.model_type() == "qwen3_moe" else {"bias": code(True, "qwen2_moe: qkv biases")}
    a["mixers"] = {"full": attn_mixer(c, extra=extra)}
    a["ffns"] = {"moe": moe_ffn(c, experts=("num_experts",), active=("num_experts_per_tok",), d_expert=("moe_intermediate_size",),
                               shared=("shared_expert_intermediate_size",), d_shared=("shared_expert_intermediate_size",))}
    if "shared" in a["ffns"]["moe"]:
        a["ffns"]["moe"]["shared"] = code(1, "one shared expert of shared_expert_intermediate_size")
    only = set(c.get("mlp_only_layers") or [])
    if only:
        a["ffns"]["dense"] = dense_ffn(c)
    a["layout"] = uniform(c, "full", lambda i: "dense" if i in only else "moe")
    return a


def mla_mixer(c: Cfg, *, indexer: bool = False, heads: str = "num_attention_heads") -> dict[str, Any]:
    m: dict[str, Any] = {"type": code("mla", "multi-head latent attention")}
    m["heads"] = c.f(heads)
    m["q_lora_rank"] = c.f("q_lora_rank") if c.has("q_lora_rank") else code(None, "q_lora_rank: null (full-rank query)")
    m["kv_lora_rank"] = c.f("kv_lora_rank")
    m["qk_nope"] = c.f("qk_nope_head_dim")
    m["qk_rope"] = c.f("qk_rope_head_dim")
    m["v_head_dim"] = c.f("v_head_dim")
    if indexer:
        m["indexer"] = {"heads": c.f("index_n_heads"), "head_dim": c.f("index_head_dim"), "topk": c.f("index_topk")}
    return m


def deepseek(c: Cfg) -> dict[str, Any]:
    a = base(c)
    mt = c.model_type()
    a["mixers"] = {"mla": mla_mixer(c, indexer=(mt == "deepseek_v32"))}
    a["ffns"] = {"dense": dense_ffn(c), "moe": moe_ffn(c, experts=("n_routed_experts",), active=("num_experts_per_tok",),
                                                     d_expert=("moe_intermediate_size",), shared=("n_shared_experts",),
                                                     d_shared=("moe_intermediate_size",))}
    k = int(c.get("first_k_dense_replace", 0))
    a["layout"] = layout_field([{"mixer": "mla", "ffn": "dense" if i < k else "moe"} for i in range(n_layers(c))],
                               "config", c.prefix + "num_hidden_layers, first_k_dense_replace")
    if c.get("num_nextn_predict_layers"):
        a["mtp_layers"] = c.f("num_nextn_predict_layers")
    return a


def deepseek_moe_v1(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["mixers"] = {"full": attn_mixer(c)}
    a["ffns"] = {"dense": dense_ffn(c), "moe": moe_ffn(c, experts=("n_routed_experts",), active=("num_experts_per_tok",),
                                                     d_expert=("moe_intermediate_size",), shared=("n_shared_experts",),
                                                     d_shared=("moe_intermediate_size",))}
    k = int(c.get("first_k_dense_replace", 0))
    a["layout"] = layout_field([{"mixer": "full", "ffn": "dense" if i < k else "moe"} for i in range(n_layers(c))],
                               "config", c.prefix + "num_hidden_layers, first_k_dense_replace")
    return a


def glm4_moe(c: Cfg) -> dict[str, Any]:
    a = base(c)
    extra: dict[str, Any] = {"bias": code(True, "attention_bias: true")} if c.get("attention_bias") else {}
    if c.get("use_qk_norm"):
        extra["qk_norm"] = c.f("use_qk_norm")
    a["mixers"] = {"full": attn_mixer(c, extra=extra)}
    a["ffns"] = {"dense": dense_ffn(c), "moe": moe_ffn(c, experts=("n_routed_experts",), active=("num_experts_per_tok",),
                                                     d_expert=("moe_intermediate_size",), shared=("n_shared_experts",),
                                                     d_shared=("moe_intermediate_size",))}
    k = int(c.get("first_k_dense_replace", 0))
    a["layout"] = layout_field([{"mixer": "full", "ffn": "dense" if i < k else "moe"} for i in range(n_layers(c))],
                               "config", c.prefix + "num_hidden_layers, first_k_dense_replace")
    if c.get("num_nextn_predict_layers"):
        a["mtp_layers"] = c.f("num_nextn_predict_layers")
    return a


def glm_dsa(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["mixers"] = {"mla": mla_mixer(c, indexer=True)}
    a["ffns"] = {"dense": dense_ffn(c), "moe": moe_ffn(c, experts=("n_routed_experts",), active=("num_experts_per_tok",),
                                                     d_expert=("moe_intermediate_size",), shared=("n_shared_experts",),
                                                     d_shared=("moe_intermediate_size",))}
    k = int(c.get("first_k_dense_replace", 0))
    it = c.get("indexer_types")
    per = []
    for i in range(n_layers(c)):
        run: dict[str, Any] = {"mixer": "mla", "ffn": "dense" if i < k else "moe"}
        if it is not None:
            run["indexer"] = it[i] == "full"
        per.append(run)
    ref = "num_hidden_layers, first_k_dense_replace" + (", indexer_types" if it is not None else "")
    a["layout"] = layout_field(per, "config", c.prefix + ref)
    if c.get("num_nextn_predict_layers"):
        a["mtp_layers"] = c.f("num_nextn_predict_layers")
    return a


def gpt_oss(c: Cfg) -> dict[str, Any]:
    a = base(c)
    bias = {"bias": code(True, "attention_bias: true")}
    a["mixers"] = {"full": attn_mixer(c, extra=bias), "sliding": attn_mixer(c, window=c.f("sliding_window"), extra=bias)}
    a["ffns"] = {"moe": moe_ffn(c, experts=("num_local_experts",), active=("num_experts_per_tok", "experts_per_token"),
                                d_expert=("intermediate_size",))}
    a["layout"] = from_layer_types(c, lambda i: "moe")
    return a


def grok(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["mixers"] = {"full": attn_mixer(c)}
    moe = moe_ffn(c, experts=("num_local_experts",), active=("num_experts_per_tok",), d_expert=("moe_intermediate_size",))
    if c.get("residual_moe"):
        moe["dense_parallel_d_ff"] = c.f("intermediate_size")
    a["ffns"] = {"moe": moe}
    a["layout"] = uniform(c, "full", lambda i: "moe")
    return a


def qwen3_next(c: Cfg) -> dict[str, Any]:
    a = base(c)
    gate = {"gate": code("elementwise", "attn_output_gate: q_proj also produces a sigmoid output gate")}
    if c.get("attn_output_gate") is False:
        gate = {}
    gate["qk_norm"] = code(True, f"{TF} {c.model_type()}: q_norm and k_norm")
    a["mixers"] = {
        "full": attn_mixer(c, extra=gate),
        "linear": {"type": code("deltanet", "Gated DeltaNet linear attention"), "k_heads": c.f("linear_num_key_heads"),
                   "v_heads": c.f("linear_num_value_heads"), "k_head_dim": c.f("linear_key_head_dim"),
                   "v_head_dim": c.f("linear_value_head_dim"), "conv_kernel": c.f("linear_conv_kernel_dim")},
    }
    if c.has("num_experts"):
        a["ffns"] = {"moe": moe_ffn(c, experts=("num_experts",), active=("num_experts_per_tok",), d_expert=("moe_intermediate_size",),
                                    shared=("shared_expert_intermediate_size",), d_shared=("shared_expert_intermediate_size",))}
        if "shared" in a["ffns"]["moe"]:
            a["ffns"]["moe"]["shared"] = code(1, "one shared expert of shared_expert_intermediate_size")
        ffn = "moe"
    else:
        a["ffns"] = {"dense": dense_ffn(c)}
        ffn = "dense"
    if c.has("layer_types"):
        a["layout"] = from_layer_types(c, lambda i: ffn)
    else:
        k = int(c.get("full_attention_interval"))
        per = [{"mixer": "full" if (i + 1) % k == 0 else "linear", "ffn": ffn} for i in range(n_layers(c))]
        a["layout"] = layout_field(per, "config", c.prefix + "full_attention_interval")
    mtp = c.fopt("mtp_num_hidden_layers", "num_nextn_predict_layers")
    if mtp and mtp["v"]:
        a["mtp_layers"] = mtp
        a["mtp_layer"] = code({"mixer": "full", "ffn": ffn, "n": 1}, "MTP layer: a full-attention block")
    return a


def minimax_m2(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["mixers"] = {"full": attn_mixer(c, extra={"qk_norm": c.f("use_qk_norm")})}
    a["ffns"] = {"moe": moe_ffn(c, experts=("num_local_experts",), active=("num_experts_per_tok",), d_expert=("intermediate_size",))}
    lst = c.get("attn_type_list")
    per = [{"mixer": "full" if x == 1 else "linear", "ffn": "moe"} for x in lst]
    a["layout"] = layout_field(per, "config", c.prefix + "attn_type_list")
    if c.get("use_mtp"):
        a["mtp_layers"] = c.f("num_mtp_modules")
    return a


def minimax_text01(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["mixers"] = {"full": attn_mixer(c),
                   "linear": {"type": code("linear", "Lightning attention (linear)"), "heads": c.f("num_attention_heads"),
                              "head_dim": c.f("head_dim")}}
    a["ffns"] = {"moe": moe_ffn(c, experts=("num_local_experts",), active=("num_experts_per_tok",), d_expert=("intermediate_size",))}
    lst = c.get("attn_type_list")
    per = [{"mixer": "full" if x == 1 else "linear", "ffn": "moe"} for x in lst]
    a["layout"] = layout_field(per, "config", c.prefix + "attn_type_list")
    return a


def kimi_linear(c: Cfg) -> dict[str, Any]:
    a = base(c)
    la = c.get("linear_attn_config")
    mla = mla_mixer(c)
    if c.get("mla_use_output_gate"):
        mla["gate"] = code("elementwise", "mla_use_output_gate: true")
    a["mixers"] = {
        "mla": mla,
        "kda": {"type": code("kda", "Kimi Delta Attention"), "k_heads": F(la["num_heads"], "config", "hf", c.prefix + "linear_attn_config.num_heads"),
                "v_heads": F(la["num_heads"], "config", "hf", c.prefix + "linear_attn_config.num_heads"),
                "k_head_dim": F(la["head_dim"], "config", "hf", c.prefix + "linear_attn_config.head_dim"),
                "v_head_dim": F(la["head_dim"], "config", "hf", c.prefix + "linear_attn_config.head_dim"),
                "conv_kernel": F(la.get("short_conv_kernel_size", 4), "config", "hf", c.prefix + "linear_attn_config.short_conv_kernel_size")},
    }
    moe = moe_ffn(c, experts=("num_experts",), active=("num_experts_per_token", "num_experts_per_tok"),
                  d_expert=("moe_intermediate_size",), shared=("num_shared_experts",), d_shared=("moe_intermediate_size",))
    if c.has("routed_expert_hidden_size"):
        moe["latent"] = c.f("routed_expert_hidden_size")
    a["ffns"] = {"dense": dense_ffn(c), "moe": moe}
    k = int(c.get("first_k_dense_replace", 0))
    full = set(la["full_attn_layers"])
    per = [{"mixer": "mla" if (i + 1) in full else "kda", "ffn": "dense" if i < k else "moe"} for i in range(n_layers(c))]
    a["layout"] = layout_field(per, "config", c.prefix + "linear_attn_config.full_attn_layers (1-based), first_k_dense_replace")
    return a


def nemotron_h(c: Cfg) -> dict[str, Any]:
    a = base(c)
    d = c.get("hidden_size")
    a["mixers"] = {
        "full": attn_mixer(c),
        "mamba": {"type": code("mamba2", "Mamba-2 mixer"), "heads": c.f("mamba_num_heads"), "head_dim": c.f("mamba_head_dim"),
                  "state": c.f("ssm_state_size"), "groups": c.f("n_groups"), "conv_kernel": c.f("conv_kernel")},
    }
    ffns: dict[str, Any] = {}
    if c.has("n_routed_experts"):
        moe = moe_ffn(c, experts=("n_routed_experts",), active=("num_experts_per_tok",), d_expert=("moe_intermediate_size",),
                      shared=("n_shared_experts",), d_shared=("moe_shared_expert_intermediate_size",), gated=False)
        if c.get("moe_latent_size"):
            moe["latent"] = c.f("moe_latent_size")
        ffns["moe"] = moe
    ffns["dense"] = dense_ffn(c, gated=False)
    a["ffns"] = ffns
    per: list[dict[str, Any]] = []
    if c.has("hybrid_override_pattern"):
        pat = c.get("hybrid_override_pattern")
        for ch in pat:
            per.append({"M": {"mixer": "mamba", "ffn": "none"}, "*": {"mixer": "full", "ffn": "none"},
                        "-": {"mixer": "none", "ffn": "dense"}, "E": {"mixer": "none", "ffn": "moe"}}[ch])
        ref = "hybrid_override_pattern"
    else:
        for t in c.get("layers_block_type"):
            per.append({"mamba": {"mixer": "mamba", "ffn": "none"}, "attention": {"mixer": "full", "ffn": "none"},
                        "mlp": {"mixer": "none", "ffn": "dense"}, "moe": {"mixer": "none", "ffn": "moe"}}[t])
        ref = "layers_block_type"
    a["layout"] = layout_field(per, "config", c.prefix + ref)
    a["norms_per_layer"] = code(1, f"{TF} nemotron_h: one pre-norm per block (each block is a mixer or an MLP)")
    if c.get("num_nextn_predict_layers"):
        a["mtp_layers"] = c.f("num_nextn_predict_layers")
        a["mtp_layer"] = code({"mixer": "full", "ffn": "moe" if "moe" in ffns else "dense", "n": 1}, "MTP: attention + MoE blocks")
    _ = d
    return a


def mimo(c: Cfg) -> dict[str, Any]:
    a = base(c)
    full = attn_mixer(c, extra={"v_head_dim": c.f("v_head_dim")})
    sw = attn_mixer(c, heads="swa_num_attention_heads", kv="swa_num_key_value_heads", hd="swa_head_dim",
                    window=c.f("sliding_window"), extra={"v_head_dim": c.f("swa_v_head_dim")})
    a["mixers"] = {"full": full, "sliding": sw}
    if c.has("index_head_dim"):
        full["indexer_note"] = code("DSA indexer on full layers", "enable_dsa: true")
    a["ffns"] = {"dense": dense_ffn(c), "moe": moe_ffn(c, experts=("n_routed_experts",), active=("num_experts_per_tok",),
                                                     d_expert=("moe_intermediate_size",))}
    pat = c.get("hybrid_layer_pattern")
    mf = c.get("moe_layer_freq")
    per = [{"mixer": "full" if p == 0 else "sliding", "ffn": "moe" if mf[i] else "dense"} for i, p in enumerate(pat)]
    a["layout"] = layout_field(per, "config", c.prefix + "hybrid_layer_pattern (0 = full, 1 = sliding), moe_layer_freq")
    if c.get("num_nextn_predict_layers"):
        a["mtp_layers"] = c.f("num_nextn_predict_layers")
        a["mtp_layer"] = code({"mixer": "sliding", "ffn": "dense", "n": 1}, "MTP block: sliding-window attention + dense MLP")
    return a


def afmoe(c: Cfg) -> dict[str, Any]:
    a = base(c)
    qk = {"qk_norm": code(True, f"{TF} afmoe: q_norm and k_norm"), "gate": code("elementwise", f"{TF} afmoe: gated attention output")}
    a["mixers"] = {"full": attn_mixer(c, extra=qk), "sliding": attn_mixer(c, window=c.f("sliding_window"), extra=qk)}
    a["ffns"] = {"dense": dense_ffn(c), "moe": moe_ffn(c, experts=("num_experts",), active=("num_experts_per_tok",),
                                                     d_expert=("moe_intermediate_size",), shared=("num_shared_experts",),
                                                     d_shared=("moe_intermediate_size",))}
    k = int(c.get("num_dense_layers"))
    lt = c.get("layer_types")
    per = [{"mixer": LT_MAP[t], "ffn": "dense" if i < k else "moe"} for i, t in enumerate(lt)]
    a["layout"] = layout_field(per, "config", c.prefix + "layer_types, num_dense_layers")
    a["norms_per_layer"] = code(4, f"{TF} afmoe: pre and post norms around attention and MLP")
    return a


def gemma4(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["tied_embeddings"] = code(True, f"{TF} gemma4: tie_word_embeddings default true")
    qk = {"qk_norm": code(True, f"{TF} gemma4: q_norm and k_norm")}
    gk = c.get("num_global_key_value_heads") or c.get("num_key_value_heads")
    full: dict[str, Any] = {"type": code("attn", "attention"), "heads": c.f("num_attention_heads"),
                            "kv_heads": c.f("num_global_key_value_heads") if c.has("num_global_key_value_heads") else c.f("num_key_value_heads"),
                            "head_dim": c.f("global_head_dim"), **qk}
    if c.get("attention_k_eq_v"):
        full["k_eq_v"] = c.f("attention_k_eq_v")
    _ = gk
    sl = attn_mixer(c, window=c.f("sliding_window"), extra=qk)
    a["mixers"] = {"full": full, "sliding": sl}
    ffns: dict[str, Any] = {"dense": dense_ffn(c)}
    if c.get("enable_moe_block"):
        moe = moe_ffn(c, experts=("num_experts",), active=("top_k_experts",), d_expert=("moe_intermediate_size", "expert_intermediate_size"))
        moe["dense_parallel_d_ff"] = c.f("intermediate_size")
        ffns["moe"] = moe
    a["ffns"] = ffns
    ffn = "moe" if c.get("enable_moe_block") else "dense"
    lt = c.get("layer_types")
    n = len(lt)
    shared_from = n - int(c.get("num_kv_shared_layers") or 0)
    per = []
    for i, t in enumerate(lt):
        run: dict[str, Any] = {"mixer": LT_MAP[t], "ffn": ffn}
        if i >= shared_from:
            run["kv_shared"] = True
            if c.get("use_double_wide_mlp"):
                run["ffn"] = "dense_wide"
        per.append(run)
    if c.get("use_double_wide_mlp") and shared_from < n:
        ffns["dense_wide"] = {"type": code("dense", "MLP"), "d_ff": code(2 * c.get("intermediate_size"), f"{TF} gemma4: use_double_wide_mlp doubles intermediate_size on KV-shared layers"),
                              "gated": code(True, "GeGLU")}
    a["layout"] = layout_field(per, "config", c.prefix + "layer_types, num_kv_shared_layers")
    a["norms_per_layer"] = code(4, f"{TF} gemma4: sandwich norms (pre and post around attention and MLP)")
    if c.get("hidden_size_per_layer_input"):
        v = c.get("vocab_size_per_layer_input") * n * c.get("hidden_size_per_layer_input")
        a["extra_embedding_params"] = code(v, f"{TF} gemma4: per-layer embeddings, vocab_size_per_layer_input x num_hidden_layers x hidden_size_per_layer_input")
    return a


def step3p5(c: Cfg) -> dict[str, Any]:
    a = base(c)
    gate = {"gate": code("headwise", "use_head_wise_attn_gate: true"), "qk_norm": c.f("use_qk_norm")}
    other = c.get("attention_other_setting")
    a["mixers"] = {
        "full": attn_mixer(c, kv="num_attention_groups", extra=gate),
        "sliding": {"type": code("attn", "attention"), "heads": F(other["num_attention_heads"], "config", "hf", "attention_other_setting.num_attention_heads"),
                    "kv_heads": F(other["num_attention_groups"], "config", "hf", "attention_other_setting.num_attention_groups"),
                    "head_dim": F(other["head_dim"], "config", "hf", "attention_other_setting.head_dim"),
                    "window": c.f("sliding_window"), **gate},
    }
    moe_layers = {int(x) for x in str(c.get("moe_layers_enum")).split(",")}
    moe = moe_ffn(c, experts=("moe_num_experts",), active=("moe_top_k",), d_expert=("moe_intermediate_size",))
    moe["shared"] = code(1, "share_expert_dim > 0: one shared expert")
    moe["d_shared"] = c.f("share_expert_dim")
    a["ffns"] = {"dense": dense_ffn(c), "moe": moe}
    lt = c.get("layer_types")
    per = [{"mixer": LT_MAP[t], "ffn": "moe" if i in moe_layers else "dense"} for i, t in enumerate(lt)]
    a["layout"] = layout_field(per, "config", "layer_types, moe_layers_enum")
    if c.get("num_nextn_predict_layers"):
        a["mtp_layers"] = c.f("num_nextn_predict_layers")
        a["mtp_layer"] = code({"mixer": "sliding", "ffn": "dense", "n": 1}, "MTP block: sliding-window attention + dense MLP")
    return a


def bailing_hybrid(c: Cfg) -> dict[str, Any]:
    a = base(c)
    mla = mla_mixer(c)
    g = int(c.get("layer_group_size"))
    n = n_layers(c)
    mixers: dict[str, Any] = {"mla": mla}
    if "v3" in str(c.get("architectures", c.raw.get("architectures"))).lower() or c.has("kda_lower_bound"):
        mixers["linear"] = {"type": code("kda", "Kimi Delta Attention (BailingMoeV3KimiDeltaAttention)"), "k_heads": c.f("num_attention_heads"),
                            "v_heads": c.f("num_attention_heads"), "k_head_dim": c.f("head_dim"), "v_head_dim": c.f("head_dim"),
                            "conv_kernel": c.f("short_conv_kernel_size")}
        ref = "modeling_bailing_moe_v3.py: softmax attention when (layer_idx + 1) % layer_group_size == 0 or in the tail"
    else:
        mixers["linear"] = {"type": code("linear", "Lightning-style linear attention (BailingMoeV2_5LinearAttention)"),
                            "heads": c.f("num_kv_heads_for_linear_attn", "num_attention_heads"), "head_dim": c.f("head_dim")}
        ref = "modeling_bailing_moe_v2_5.py: softmax attention when (layer_idx + 1) % layer_group_size == 0 or in the tail"
    a["mixers"] = mixers
    k = int(c.get("first_k_dense_replace", 0))
    a["ffns"] = {"dense": dense_ffn(c), "moe": moe_ffn(c, experts=("num_experts",), active=("num_experts_per_tok",),
                                                     d_expert=("moe_intermediate_size",), shared=("num_shared_experts",),
                                                     d_shared=("moe_shared_expert_intermediate_size",))}
    tail = n // g * g
    per = [{"mixer": "mla" if ((i + 1) % g == 0 or i >= tail) else "linear", "ffn": "dense" if i < k else "moe"} for i in range(n)]
    a["layout"] = code(rle(per), ref)
    if c.get("num_nextn_predict_layers"):
        a["mtp_layers"] = c.f("num_nextn_predict_layers")
        a["mtp_layer"] = code({"mixer": "mla", "ffn": "moe", "n": 1}, "MTP layer: MLA + MoE")
    return a


def sarvam(c: Cfg) -> dict[str, Any]:
    a = base(c)
    if c.model_type() == "sarvam_mla":
        a["mixers"] = {"mla": mla_mixer(c)}
        mx = "mla"
    else:
        a["mixers"] = {"full": attn_mixer(c, extra={"qk_norm": c.f("use_qk_norm")})}
        mx = "full"
    a["ffns"] = {"dense": dense_ffn(c), "moe": moe_ffn(c, experts=("num_experts",), active=("num_experts_per_tok",),
                                                     d_expert=("moe_intermediate_size",), shared=("num_shared_experts",),
                                                     d_shared=("moe_shared_expert_intermediate_size", "moe_intermediate_size"))}
    k = int(c.get("first_k_dense_replace", 0))
    per = [{"mixer": mx, "ffn": "dense" if i < k else "moe"} for i in range(n_layers(c))]
    a["layout"] = layout_field(per, "config", "num_hidden_layers, first_k_dense_replace")
    return a


def xlstm(c: Cfg) -> dict[str, Any]:
    a: dict[str, Any] = {"d_model": c.f("embedding_dim"), "vocab": c.f("vocab_size"),
                         "tied_embeddings": code(False, f"{TF} xlstm: separate lm_head")}
    d = c.get("embedding_dim")
    up = c.get("mlstm_round_up_to_multiple_of")
    qk = math.ceil(d * c.get("qk_dim_factor") / up) * up
    v = math.ceil(d * c.get("v_dim_factor") / up) * up
    ff_up = c.get("ffn_round_up_to_multiple_of")
    dff = math.ceil(d * c.get("ffn_proj_factor") / ff_up) * ff_up
    a["mixers"] = {"mlstm": {"type": code("mlstm", "mLSTM cell"), "heads": c.f("num_heads"),
                             "qk_dim": code(qk, "qk_dim = round_up(embedding_dim * qk_dim_factor, mlstm_round_up_to_multiple_of)"),
                             "v_dim": code(v, "v_dim = round_up(embedding_dim * v_dim_factor, mlstm_round_up_to_multiple_of)")}}
    a["ffns"] = {"dense": {"type": code("dense", "MLP"), "d_ff": code(dff, "d_ff = round_up(embedding_dim * ffn_proj_factor, ffn_round_up_to_multiple_of)"),
                           "gated": code(True, "SwiGLU")}}
    a["layout"] = layout_field([{"mixer": "mlstm", "ffn": "dense"}] * int(c.get("num_blocks")), "config", "num_blocks")
    return a


def longcat(c: Cfg) -> dict[str, Any]:
    a = base(c)
    mla = mla_mixer(c)
    a["mixers"] = {"mla": mla}
    moe = moe_ffn(c, experts=("n_routed_experts",), active=("moe_topk",), d_expert=("expert_ffn_hidden_size",))
    a["ffns"] = {"dense": dense_ffn(c, key="ffn_hidden_size"), "moe": moe}
    n = int(c.get("num_layers"))
    # Each LongCat-Flash layer holds two attention blocks, two dense MLPs and
    # one MoE block (shortcut-connected MoE).
    per = []
    for _ in range(n):
        per += [{"mixer": "mla", "ffn": "dense"}, {"mixer": "mla", "ffn": "dense"}, {"mixer": "none", "ffn": "moe"}]
    a["layout"] = code(rle(per), f"{TF} longcat_flash: each of num_layers layers has 2 MLA blocks, 2 dense MLPs and 1 MoE block")
    v = c.get("vocab_size") * c.get("ngram_vocab_size_ratio") * c.get("hidden_size")
    a["extra_embedding_params"] = code(v, "n-gram embedding table: vocab_size x ngram_vocab_size_ratio x hidden_size (approximate)")
    return a


def mistral_params(c: Cfg) -> dict[str, Any]:
    """Mistral's native params.json (Mistral Large 3)."""
    a: dict[str, Any] = {"d_model": c.f("dim"), "vocab": c.f("vocab_size"), "tied_embeddings": c.f("tied_embeddings")}
    m: dict[str, Any] = {"type": code("mla", "multi-head latent attention"), "heads": c.f("n_heads"), "q_lora_rank": c.f("q_lora_rank"),
                         "kv_lora_rank": c.f("kv_lora_rank"), "qk_nope": c.f("qk_nope_head_dim"), "qk_rope": c.f("qk_rope_head_dim"),
                         "v_head_dim": c.f("v_head_dim")}
    a["mixers"] = {"mla": m}
    moe = c.get("moe")
    a["ffns"] = {"dense": {"type": code("dense", "MLP"), "d_ff": c.f("hidden_dim"), "gated": code(True, "SwiGLU")},
                 "moe": {"type": code("moe", "MoE"), "experts": F(moe["num_experts"], "config", "hf", "moe.num_experts"),
                         "active": F(moe["num_experts_per_tok"], "config", "hf", "moe.num_experts_per_tok"),
                         "d_expert": F(moe["expert_hidden_dim"], "config", "hf", "moe.expert_hidden_dim"),
                         "shared": F(moe["num_shared_experts"], "config", "hf", "moe.num_shared_experts"),
                         "d_shared": F(moe["expert_hidden_dim"], "config", "hf", "moe.expert_hidden_dim"),
                         "gated": code(True, "SwiGLU experts")}}
    k = moe["first_k_dense_replace"]
    per = [{"mixer": "mla", "ffn": "dense" if i < k else "moe"} for i in range(int(c.get("n_layers")))]
    a["layout"] = layout_field(per, "config", "n_layers, moe.first_k_dense_replace")
    return a


def mistral4(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["mixers"] = {"mla": mla_mixer(c)}
    a["ffns"] = {"dense": dense_ffn(c), "moe": moe_ffn(c, experts=("n_routed_experts",), active=("num_experts_per_tok",),
                                                     d_expert=("moe_intermediate_size",), shared=("n_shared_experts",),
                                                     d_shared=("moe_intermediate_size",))}
    k = int(c.get("first_k_dense_replace", 0))
    per = [{"mixer": "mla", "ffn": "dense" if i < k else "moe"} for i in range(n_layers(c))]
    a["layout"] = layout_field(per, "config", c.prefix + "num_hidden_layers, first_k_dense_replace")
    return a


def laguna(c: Cfg) -> dict[str, Any]:
    a = base(c)
    hpl = c.get("num_attention_heads_per_layer")
    lt = c.get("layer_types")
    gate = {"gate": code("headwise", "gating: per-head output gate"), "qk_norm": code(True, "modeling_laguna.py: q/k norms")}
    full_h = {hpl[i] for i, t in enumerate(lt) if t == "full_attention"}
    sw_h = {hpl[i] for i, t in enumerate(lt) if t == "sliding_attention"}
    assert len(full_h) == 1 and len(sw_h) == 1
    full = attn_mixer(c, extra=gate)
    full["heads"] = F(full_h.pop(), "config", "hf", "num_attention_heads_per_layer (full layers)")
    sl = attn_mixer(c, window=c.f("sliding_window"), extra=gate)
    sl["heads"] = F(sw_h.pop(), "config", "hf", "num_attention_heads_per_layer (sliding layers)")
    a["mixers"] = {"full": full, "sliding": sl}
    moe = moe_ffn(c, experts=("num_experts",), active=("num_experts_per_tok",), d_expert=("moe_intermediate_size",),
                  shared=("shared_expert_intermediate_size",), d_shared=("shared_expert_intermediate_size",))
    moe["shared"] = code(1, "one shared expert of shared_expert_intermediate_size")
    a["ffns"] = {"dense": dense_ffn(c), "moe": moe}
    mlt = c.get("mlp_layer_types")
    per = [{"mixer": LT_MAP[t], "ffn": "dense" if mlt[i] == "dense" else "moe"} for i, t in enumerate(lt)]
    a["layout"] = layout_field(per, "config", "layer_types, mlp_layer_types")
    return a


def zaya(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["mixers"] = {"full": attn_mixer(c)}
    a["mixers"]["full"]["type"] = code("attn", "CCA (compressed convolutional attention) approximated as GQA attention", note="approximation")
    a["ffns"] = {"moe": moe_ffn(c, experts=("num_experts",), active=("num_experts_per_tok",), d_expert=("moe_intermediate_size",))}
    a["layout"] = uniform(c, "full", lambda i: "moe")
    return a


def hy_v3(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["mixers"] = {"full": attn_mixer(c, extra={"qk_norm": c.f("qk_norm")})}
    a["ffns"] = {"dense": dense_ffn(c), "moe": moe_ffn(c, experts=("num_experts",), active=("num_experts_per_tok",),
                                                     d_expert=("moe_intermediate_size",), shared=("num_shared_experts",),
                                                     d_shared=("moe_intermediate_size",))}
    k = int(c.get("first_k_dense_replace", 0))
    a["layout"] = layout_field([{"mixer": "full", "ffn": "dense" if i < k else "moe"} for i in range(n_layers(c))],
                               "config", "num_hidden_layers, first_k_dense_replace")
    if c.get("num_nextn_predict_layers"):
        a["mtp_layers"] = c.f("num_nextn_predict_layers")
    return a


def hy_v4(c: Cfg) -> dict[str, Any]:
    a = base(c)
    mla = mla_mixer(c, indexer=True)
    mla["gate"] = code("elementwise", "gated_mla: true, gating_type: elementwise")
    a["mixers"] = {"mla": mla}
    a["ffns"] = {"dense": dense_ffn(c), "moe": moe_ffn(c, experts=("n_routed_experts",), active=("num_experts_per_tok",),
                                                     d_expert=("moe_intermediate_size",), shared=("n_shared_experts",),
                                                     d_shared=("moe_intermediate_size",))}
    mlt = c.get("mlp_layer_types")
    it = c.get("indexer_types")
    per = [{"mixer": "mla", "ffn": "dense" if mlt[i] == "dense" else "moe", "indexer": it[i] == "full"} for i in range(n_layers(c))]
    a["layout"] = layout_field(per, "config", "layer_types, mlp_layer_types, indexer_types")
    if c.get("num_nextn_predict_layers"):
        a["mtp_layers"] = c.f("num_nextn_predict_layers")
    return a


def cohere2_moe(c: Cfg) -> dict[str, Any]:
    a = base(c)
    if c.has("use_embedding_sharing"):
        a["tied_embeddings"] = c.f("use_embedding_sharing")
    else:
        a["tied_embeddings"] = code(True, f"{TF} cohere2_moe: tie_word_embeddings default true")
    a["mixers"] = {"full": attn_mixer(c), "sliding": attn_mixer(c, window=c.f("sliding_window"))}
    moe = moe_ffn(c, experts=("num_experts",), active=("num_experts_per_tok",), d_expert=("intermediate_size",),
                  shared=("num_shared_experts",), d_shared=("intermediate_size",))
    a["ffns"] = {"dense": dense_ffn(c, key="prefix_dense_intermediate_size"), "moe": moe}
    k = int(c.get("first_k_dense_replace", 0))
    lt = c.get("layer_types")
    per = [{"mixer": LT_MAP[t], "ffn": "dense" if i < k else "moe"} for i, t in enumerate(lt)]
    a["layout"] = layout_field(per, "config", c.prefix + "layer_types, first_k_dense_replace")
    a["norms_per_layer"] = code(1, f"{TF} cohere2_moe: one LayerNorm per parallel block")
    return a


def lfm2(c: Cfg) -> dict[str, Any]:
    a = base(c)
    if c.has("tie_embedding"):
        a["tied_embeddings"] = c.f("tie_embedding")
    else:
        a["tied_embeddings"] = code(True, f"{TF} lfm2_moe: tie_word_embeddings default true")
    a["mixers"] = {"full": attn_mixer(c, extra={"qk_norm": code(True, f"{TF} lfm2: q_layernorm and k_layernorm")}),
                   "conv": {"type": code("conv", "gated short convolution"), "kernel": c.f("conv_L_cache")}}
    if c.model_type() == "lfm2":
        ff = c.get("block_ff_dim")
        if c.get("block_auto_adjust_ff_dim"):
            ff = int(2 * ff / 3)
            ff = int(c.get("block_ffn_dim_multiplier") * ff)
            mo = c.get("block_multiple_of")
            ff = mo * ((ff + mo - 1) // mo)
        a["ffns"] = {"dense": {"type": code("dense", "MLP"), "d_ff": code(ff, f"{TF} lfm2: block_auto_adjust_ff_dim: int(2/3 * block_ff_dim * multiplier) rounded up to block_multiple_of"),
                               "gated": code(True, "SwiGLU")}}
        fn = lambda i: "dense"  # noqa: E731
    else:
        a["ffns"] = {"dense": dense_ffn(c), "moe": moe_ffn(c, experts=("num_experts",), active=("num_experts_per_tok",), d_expert=("moe_intermediate_size",))}
        k = int(c.get("num_dense_layers", 0))
        fn = dense_prefix(k)
    a["layout"] = from_layer_types(c, fn)
    return a


def mellum(c: Cfg) -> dict[str, Any]:
    a = base(c)
    qk = {"qk_norm": code(True, f"{TF} mellum: q_norm and k_norm")}
    a["mixers"] = {"full": attn_mixer(c, extra=qk), "sliding": attn_mixer(c, window=c.f("sliding_window"), extra=qk)}
    a["ffns"] = {"dense": dense_ffn(c), "moe": moe_ffn(c, experts=("num_experts",), active=("num_experts_per_tok",), d_expert=("moe_intermediate_size",))}
    mlt = c.get("mlp_layer_types")
    lt = c.get("layer_types")
    per = [{"mixer": LT_MAP[t], "ffn": "moe" if mlt[i] == "sparse" else "dense"} for i, t in enumerate(lt)]
    a["layout"] = layout_field(per, "config", "layer_types, mlp_layer_types")
    return a


def minimax_m3(c: Cfg) -> dict[str, Any]:
    a = base(c)
    sac = c.get("sparse_attention_config") or {}
    full = attn_mixer(c, extra={"qk_norm": c.f("use_qk_norm")})
    a["mixers"] = {"full": full}
    if sac.get("use_sparse_attention"):
        sp = attn_mixer(c, extra={"qk_norm": c.f("use_qk_norm")})
        P = c.prefix + "sparse_attention_config."
        topk = (sac["sparse_topk_blocks"] + sac.get("sparse_local_block", 0)) * sac["sparse_block_size"]
        sp["indexer"] = {"heads": F(sac["sparse_num_index_heads"], "config", "hf", P + "sparse_num_index_heads"),
                         "head_dim": F(sac["sparse_index_dim"], "config", "hf", P + "sparse_index_dim"),
                         "topk": code(topk, "(sparse_topk_blocks + sparse_local_block) x sparse_block_size tokens read per query "
                                            f"({TF} minimax_m3_vl lightning indexer)")}
        a["mixers"]["sparse"] = sp
    moe = moe_ffn(c, experts=("num_local_experts",), active=("num_experts_per_tok",), d_expert=("intermediate_size",),
                  shared=("n_shared_experts",), d_shared=("shared_intermediate_size",))
    a["ffns"] = {"dense": dense_ffn(c, key="dense_intermediate_size"), "moe": moe}
    mf = c.get("moe_layer_freq")
    sf = sac.get("sparse_attention_freq") if sac.get("use_sparse_attention") else None
    per = [{"mixer": "sparse" if sf and sf[i] else "full", "ffn": "moe" if mf[i] else "dense"} for i in range(len(mf))]
    a["layout"] = layout_field(per, "config", c.prefix + "moe_layer_freq, sparse_attention_config.sparse_attention_freq")
    if c.get("num_mtp_modules"):
        a["mtp_layers"] = c.f("num_mtp_modules")
    return a


def inkling(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["mixers"] = {"full": attn_mixer(c),
                   "sliding": attn_mixer(c, heads="swa_num_attention_heads", kv="swa_num_key_value_heads", hd="swa_head_dim",
                                         window=c.f("sliding_window_size"))}
    a["ffns"] = {"dense": dense_ffn(c, key="dense_intermediate_size"),
                 "moe": moe_ffn(c, experts=("n_routed_experts",), active=("num_experts_per_tok",), d_expert=("intermediate_size",),
                                shared=("n_shared_experts",), d_shared=("intermediate_size",))}
    local = set(c.get("local_layer_ids"))
    k = int(c.get("dense_mlp_idx"))
    per = [{"mixer": "sliding" if i in local else "full", "ffn": "dense" if i < k else "moe"} for i in range(n_layers(c))]
    a["layout"] = layout_field(per, "config", c.prefix + "local_layer_ids, dense_mlp_idx")
    mc = c.raw.get("mtp_config") or {}
    if mc.get("num_nextn_predict_layers"):
        a["mtp_layers"] = F(mc["num_nextn_predict_layers"], "config", "hf", "mtp_config.num_nextn_predict_layers")
    return a


def motif(c: Cfg) -> dict[str, Any]:
    a = base(c)
    mla: dict[str, Any] = {"type": code("mla", "GDLA: latent attention with q/kv low rank (approximated as MLA)", note="approximation"),
                           "heads": c.f("num_attention_heads"), "q_lora_rank": c.f("q_lora_rank"), "kv_lora_rank": c.f("kv_lora_rank"),
                           "qk_nope": code(c.get("head_dim") - c.get("qk_rope_head_dim"), "head_dim - qk_rope_head_dim"),
                           "qk_rope": c.f("qk_rope_head_dim"), "v_head_dim": c.f("v_head_dim")}
    sw = dict(mla)
    sw["window"] = c.f("sliding_window")
    a["mixers"] = {"mla": mla, "mla_sliding": sw}
    a["ffns"] = {"dense": dense_ffn(c), "moe": moe_ffn(c, experts=("num_experts",), active=("experts_top_k",), d_expert=("moe_intermediate_size",),
                                                     shared=("num_shared_experts",), d_shared=("moe_intermediate_size",))}
    k = int(c.get("n_dense_first_layers"))
    p = int(c.get("sliding_window_period"))
    per = [{"mixer": "mla" if (i + 1) % p == 0 else "mla_sliding", "ffn": "dense" if i < k else "moe"} for i in range(n_layers(c))]
    a["layout"] = code(rle(per), "sliding_window_pattern: interleave, sliding_window_period: every period-th layer global (assumed)", )
    if c.get("num_nextn_predict_layers"):
        a["mtp_layers"] = c.f("num_nextn_predict_layers")
    return a


def solar_open2(c: Cfg) -> dict[str, Any]:
    a = base(c)
    la = c.get("linear_attn_config")
    a["mixers"] = {"full": attn_mixer(c, extra={"gate": code("elementwise", "use_gqa_gate: true")}),
                   "kda": {"type": code("kda", "Kimi Delta Attention"), "k_heads": F(la["num_heads"], "config", "hf", "linear_attn_config.num_heads"),
                           "v_heads": F(la["num_heads"], "config", "hf", "linear_attn_config.num_heads"),
                           "k_head_dim": F(la["head_dim"], "config", "hf", "linear_attn_config.head_dim"),
                           "v_head_dim": F(la["head_dim"], "config", "hf", "linear_attn_config.head_dim"),
                           "conv_kernel": F(la["short_conv_kernel_size"], "config", "hf", "linear_attn_config.short_conv_kernel_size")}}
    a["ffns"] = {"moe": moe_ffn(c, experts=("n_routed_experts",), active=("num_experts_per_tok",), d_expert=("moe_intermediate_size",),
                                shared=("n_shared_experts",), d_shared=("moe_intermediate_size",))}
    g = set(c.get("gqa_layers"))
    per = [{"mixer": "full" if i in g else "kda", "ffn": "moe"} for i in range(n_layers(c))]
    a["layout"] = layout_field(per, "config", "gqa_layers")
    return a


def kolibri(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["mixers"] = {"full": attn_mixer(c), "sliding": attn_mixer(c, window=c.f("sliding_window"))}
    moe = moe_ffn(c, experts=("num_experts",), active=("num_experts_per_tok",), d_expert=("moe_intermediate_size",),
                  shared=("shared_expert_intermediate_size",), d_shared=("shared_expert_intermediate_size",))
    moe["shared"] = code(1, "one shared expert of shared_expert_intermediate_size")
    a["ffns"] = {"moe": moe}
    a["layout"] = from_layer_types(c, lambda i: "moe")
    return a


def muse(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["mixers"] = {"full": attn_mixer(c), "sliding": attn_mixer(c, window=c.f("sliding_window"))}
    a["ffns"] = {"dense": dense_ffn(c)}
    a["layout"] = from_layer_types(c, lambda i: "dense")
    return a


def deepseek_v4(c: Cfg) -> dict[str, Any]:
    a = base(c)
    m: dict[str, Any] = {"type": code("csa", "DeepSeek V4 compressed attention (CSA/HCA) with a sliding window"),
                         "heads": c.f("num_attention_heads"), "head_dim": c.f("head_dim"), "q_lora_rank": c.f("q_lora_rank"),
                         "o_lora_rank": c.f("o_lora_rank"), "o_groups": c.f("o_groups"), "window": c.f("sliding_window"),
                         "indexer": {"heads": c.f("index_n_heads"), "head_dim": c.f("index_head_dim"), "topk": c.f("index_topk")}}
    a["mixers"] = {"csa": m}
    a["ffns"] = {"moe": moe_ffn(c, experts=("n_routed_experts",), active=("num_experts_per_tok",), d_expert=("moe_intermediate_size",),
                                shared=("n_shared_experts",), d_shared=("moe_intermediate_size",))}
    cr = c.get("compress_ratios")
    n = n_layers(c)
    srcs = c.get("kv_source_layer_ids")
    idx_src = c.get("index_source_layer_ids")
    per = []
    for i in range(n):
        r = cr[i]
        run: dict[str, Any] = {"mixer": "csa", "ffn": "moe", "ratio": r}
        if srcs is not None:
            run["kv_source"] = i in srcs
            run["indexer"] = idx_src is not None and i in idx_src
        else:
            run["indexer"] = r == 4
        per.append(run)
    ref = "compress_ratios" + (", kv_source_layer_ids, index_source_layer_ids" if srcs is not None else " (indexer on ratio-4 layers)")
    a["layout"] = layout_field(per, "config", c.prefix + ref)
    if c.get("num_nextn_predict_layers"):
        a["mtp_layers"] = c.f("num_nextn_predict_layers")
    return a


def glm5_next(c: Cfg) -> dict[str, Any]:
    a = base(c)
    la = c.get("linear_attn_config")
    a["mixers"] = {"mla": mla_mixer(c, indexer=True),
                   "kda": {"type": code("kda", "Kimi Delta Attention"), "k_heads": F(la["num_heads"], "config", "hf", c.prefix + "linear_attn_config.num_heads"),
                           "v_heads": F(la["num_heads"], "config", "hf", c.prefix + "linear_attn_config.num_heads"),
                           "k_head_dim": F(la["head_dim"], "config", "hf", c.prefix + "linear_attn_config.head_dim"),
                           "v_head_dim": F(la["head_dim"], "config", "hf", c.prefix + "linear_attn_config.head_dim"),
                           "conv_kernel": F(la["short_conv_kernel_size"], "config", "hf", c.prefix + "linear_attn_config.short_conv_kernel_size")}}
    a["ffns"] = {"dense": dense_ffn(c), "moe": moe_ffn(c, experts=("n_routed_experts",), active=("num_experts_per_tok",),
                                                     d_expert=("moe_intermediate_size",), shared=("n_shared_experts",),
                                                     d_shared=("moe_intermediate_size",))}
    lt = c.get("layer_types")
    mlt = c.get("mlp_layer_types")
    per = [{"mixer": "mla" if t == "deepseek_sparse_attention" else "kda", "ffn": "dense" if mlt[i] == "dense" else "moe"} for i, t in enumerate(lt)]
    a["layout"] = layout_field(per, "config", c.prefix + "layer_types, mlp_layer_types")
    if c.get("num_nextn_predict_layers"):
        a["mtp_layers"] = c.f("num_nextn_predict_layers")
    return a


def qwen4_exp(c: Cfg) -> dict[str, Any]:
    a = qwen3_next(c)
    a["mixers"]["full"]["gate"] = code("elementwise", "output_gate_type: sigmoid")
    return a


def mamba1(c: Cfg) -> dict[str, Any]:
    a = base(c)
    d = c.get("hidden_size")
    a["tied_embeddings"] = c.f("tie_word_embeddings") if c.has("tie_word_embeddings") else code(True, f"{TF} mamba: tied")
    di = c.get("intermediate_size", 2 * d)
    a["mixers"] = {"mamba": {"type": code("mamba1", "Mamba (selective SSM)"), "d_inner": c.f("intermediate_size"),
                             "state": c.f("state_size"), "conv_kernel": c.f("conv_kernel"), "dt_rank": c.f("time_step_rank")}}
    a["ffns"] = {}
    a["layout"] = uniform(c, "mamba", lambda i: "none")
    a["norms_per_layer"] = code(1, "one RMSNorm per Mamba block")
    _ = di
    return a


def mamba2_hf(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["tied_embeddings"] = c.f("tie_word_embeddings") if c.has("tie_word_embeddings") else code(True, "mamba2: tied")
    a["mixers"] = {"mamba": {"type": code("mamba2", "Mamba-2"), "heads": code(c.get("hidden_size") * c.get("expand", 2) // c.get("headdim", 64), "d_inner / headdim"),
                             "head_dim": c.f("headdim"), "state": c.f("d_state"), "groups": c.f("ngroups"), "conv_kernel": c.f("d_conv")}}
    a["ffns"] = {}
    a["layout"] = layout_field([{"mixer": "mamba", "ffn": "none"}] * int(c.get("n_layer")), "config", "n_layer")
    a["norms_per_layer"] = code(1, "one RMSNorm per block")
    return a


def mamba2_ssm(c: Cfg) -> dict[str, Any]:
    a = {"d_model": c.f("d_model"), "vocab": c.f("vocab_size"), "tied_embeddings": c.f("tie_embeddings")}
    ssm = c.get("ssm_cfg") or {}
    d = c.get("d_model")
    hd = ssm.get("headdim", 64)
    a["mixers"] = {"mamba": {"type": code("mamba2", "Mamba-2"), "heads": code(2 * d // hd, "mamba_ssm Mamba2: d_inner = expand(2) * d_model; nheads = d_inner / headdim(64)"),
                             "head_dim": code(hd, "mamba_ssm Mamba2 default headdim 64"), "state": code(ssm.get("d_state", 128), "mamba_ssm Mamba2 default d_state 128"),
                             "groups": code(ssm.get("ngroups", 1), "mamba_ssm Mamba2 default ngroups 1"), "conv_kernel": code(4, "mamba_ssm Mamba2 default d_conv 4")}}
    a["ffns"] = {}
    a["layout"] = layout_field([{"mixer": "mamba", "ffn": "none"}] * int(c.get("n_layer")), "config", "n_layer")
    a["norms_per_layer"] = code(1, "one RMSNorm per block")
    return a


def rwkv(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["tied_embeddings"] = code(False, "RWKV: separate head")
    d = c.get("hidden_size")
    hd = c.get("head_size", 64)
    v6 = c.model_type() == "rwkv6"
    a["mixers"] = {"rwkv": {"type": code("rwkv", "RWKV time mixing"),
                            "head_dim": c.f("head_size") if c.has("head_size") else code(d, "RWKV-4: a single head of width hidden_size"),
                            "mats": code(5 if v6 else 4, "time mixing: receptance, key, value, output" + (", gate (RWKV-6)" if v6 else ""))}}
    if c.has("intermediate_size"):
        dff = c.f("intermediate_size")
    elif v6:
        dff = code(int(d * 3.5) // 32 * 32, "modeling_rwkv6.py: intermediate_size = int(hidden_size * 3.5) // 32 * 32")
    else:
        dff = code(4 * d, f"{TF} rwkv: intermediate_size defaults to 4 * hidden_size")
    a["ffns"] = {"dense": {"type": code("dense", "channel mixing"), "d_ff": dff, "gated": code(False, "channel mixing: key and value matrices"),
                           "receptance": code(True, "channel mixing also has a d x d receptance matrix")}}
    a["layout"] = uniform(c, "rwkv", lambda i: "dense")
    _ = hd
    return a


def falcon(c: Cfg) -> dict[str, Any]:
    a = base(c)
    d = c.get("hidden_size")
    h = c.get("num_attention_heads", c.get("n_head"))
    if c.get("new_decoder_architecture"):
        kv = c.f("num_kv_heads")
    elif c.get("multi_query"):
        kv = code(1, "multi_query: true")
    else:
        kv = F(h, "config", "hf", "num_attention_heads")
    a["mixers"] = {"full": {"type": code("attn", "attention"), "heads": c.f("num_attention_heads", "n_head"), "kv_heads": kv,
                            "head_dim": code(d // h, "hidden_size / num_attention_heads")}}
    a["ffns"] = {"dense": {"type": code("dense", "MLP"), "d_ff": code(4 * d, "modeling_falcon.py: 4 * hidden_size"), "gated": code(False, "GELU MLP")}}
    a["layout"] = uniform(c, "full", lambda i: "dense", "num_hidden_layers, n_layer")
    a["tied_embeddings"] = code(True, "modeling_falcon.py: lm_head tied to word_embeddings")
    a["norms_per_layer"] = code(1, "parallel attention and MLP")
    return a


def bloom(c: Cfg) -> dict[str, Any]:
    a = {"d_model": c.f("n_embed"), "vocab": c.f("vocab_size")}
    d = c.get("n_embed")
    h = c.get("num_attention_heads")
    a["mixers"] = {"full": {"type": code("attn", "attention"), "heads": c.f("num_attention_heads"), "kv_heads": c.f("num_attention_heads"),
                            "head_dim": code(d // h, "n_embed / num_attention_heads"), "bias": code(True, "qkv biases")}}
    a["ffns"] = {"dense": {"type": code("dense", "MLP"), "d_ff": code(4 * d, f"{TF} bloom: 4 * hidden_size"), "gated": code(False, "GELU MLP"), "bias": code(True, "biases")}}
    a["layout"] = layout_field([{"mixer": "full", "ffn": "dense"}] * int(c.get("n_layer")), "config", "n_layer")
    a["tied_embeddings"] = code(True, f"{TF} bloom: tied")
    return a


def neox(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["mixers"] = {"full": attn_mixer(c, kv="num_attention_heads", extra={"bias": code(True, "qkv biases")})}
    a["ffns"] = {"dense": dense_ffn(c, gated=False, bias=True)}
    a["layout"] = uniform(c, "full", lambda i: "dense")
    a["tied_embeddings"] = c.f("tie_word_embeddings")
    return a


def gptj(c: Cfg) -> dict[str, Any]:
    a = base(c)
    d = c.get("n_embd")
    a["mixers"] = {"full": attn_mixer(c, heads="n_head", kv="n_head")}
    a["ffns"] = {"dense": {"type": code("dense", "MLP"), "d_ff": code(4 * d, f"{TF} gptj: n_inner defaults to 4 * n_embd"), "gated": code(False, "GELU MLP"), "bias": code(True, "MLP biases")}}
    a["layout"] = layout_field([{"mixer": "full", "ffn": "dense"}] * int(c.get("n_layer")), "config", "n_layer")
    a["norms_per_layer"] = code(1, "parallel attention and MLP share one LayerNorm")
    return a


def opt(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["mixers"] = {"full": attn_mixer(c, kv="num_attention_heads", extra={"bias": code(True, "biases")})}
    a["ffns"] = {"dense": dense_ffn(c, key="ffn_dim", gated=False, bias=True)}
    a["layout"] = uniform(c, "full", lambda i: "dense")
    a["extra_embedding_params"] = code((c.get("max_position_embeddings") + 2) * c.get("hidden_size"), "learned positions: (max_position_embeddings + 2) x hidden_size")
    return a


def mixtral(c: Cfg) -> dict[str, Any]:
    a = base(c)
    window = c.f("sliding_window") if c.get("sliding_window") else None
    a["mixers"] = {"full": attn_mixer(c, window=window)}
    a["ffns"] = {"moe": moe_ffn(c, experts=("num_local_experts",), active=("num_experts_per_tok",), d_expert=("intermediate_size",))}
    a["layout"] = uniform(c, "full", lambda i: "moe")
    return a


def jamba(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["mixers"] = {"full": attn_mixer(c),
                   "mamba": {"type": code("mamba1", "Mamba (selective SSM)"), "d_inner": code(c.get("mamba_expand") * c.get("hidden_size"), "mamba_expand x hidden_size"),
                             "state": c.f("mamba_d_state"), "conv_kernel": c.f("mamba_d_conv"),
                             "dt_rank": code(math.ceil(c.get("hidden_size") / 16), "mamba_dt_rank: auto = ceil(hidden_size / 16)")}}
    a["ffns"] = {"dense": dense_ffn(c), "moe": moe_ffn(c, experts=("num_experts",), active=("num_experts_per_tok",), d_expert=("intermediate_size",))}
    ap, ao = c.get("attn_layer_period"), c.get("attn_layer_offset")
    ep, eo = c.get("expert_layer_period"), c.get("expert_layer_offset")
    per = [{"mixer": "full" if i % ap == ao else "mamba", "ffn": "moe" if i % ep == eo else "dense"} for i in range(n_layers(c))]
    a["layout"] = layout_field(per, "config", "attn_layer_period/offset, expert_layer_period/offset")
    return a


def t5(c: Cfg) -> dict[str, Any]:
    a: dict[str, Any] = {"d_model": c.f("d_model"), "vocab": c.f("vocab_size"),
                         "tied_embeddings": code(True, f"{TF} t5: tie_word_embeddings default true")}
    a["kind"] = code("encoder-decoder", "T5 encoder-decoder")
    att = {"type": code("attn", "attention"), "heads": c.f("num_heads"), "kv_heads": c.f("num_heads"), "head_dim": c.f("d_kv")}
    a["mixers"] = {"full": att}
    gated = "gated" in str(c.get("feed_forward_proj", ""))
    if c.model_type() == "switch_transformers":
        a["ffns"] = {"dense": {"type": code("dense", "MLP"), "d_ff": c.f("d_ff"), "gated": code(False, "ReLU MLP")},
                     "moe": {"type": code("moe", "Switch MoE"), "experts": c.f("num_experts"), "active": code(1, "Switch routing: top-1"),
                             "d_expert": c.f("d_ff"), "gated": code(False, "ReLU experts")}}
        es, ds = c.get("encoder_sparse_step"), c.get("decoder_sparse_step")
        enc = [{"mixer": "full", "ffn": "moe" if (i % es == es - 1) else "dense"} for i in range(int(c.get("num_layers")))]
        dec = [{"mixer": "full", "ffn": "moe" if (i % ds == ds - 1) else "dense"} for i in range(int(c.get("num_decoder_layers", c.get("num_layers"))))]
        a["layout"] = layout_field(dec, "config", "num_decoder_layers, decoder_sparse_step")
        a["encoder_layout"] = layout_field(enc, "config", "num_layers, encoder_sparse_step")
    else:
        a["ffns"] = {"dense": {"type": code("dense", "MLP"), "d_ff": c.f("d_ff"), "gated": code(gated, "feed_forward_proj")}}
        a["layout"] = layout_field([{"mixer": "full", "ffn": "dense"}] * int(c.get("num_decoder_layers", c.get("num_layers"))), "config", "num_decoder_layers")
        a["encoder_layout"] = layout_field([{"mixer": "full", "ffn": "dense"}] * int(c.get("num_layers")), "config", "num_layers")
    a["norms_per_layer"] = code(2, "pre-norm RMSNorm")
    return a


def bert(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["kind"] = code("encoder", "bidirectional encoder")
    a["tied_embeddings"] = code(True, "MLM head tied to word embeddings")
    a["mixers"] = {"full": attn_mixer(c, kv="num_attention_heads", extra={"bias": code(True, "biases")})}
    a["ffns"] = {"dense": dense_ffn(c, gated=False, bias=True)}
    a["layout"] = uniform(c, "full", lambda i: "dense")
    a["extra_embedding_params"] = code((c.get("max_position_embeddings") + c.get("type_vocab_size")) * c.get("hidden_size"),
                                       "position and token-type embeddings")
    return a


def olmoe(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["mixers"] = {"full": attn_mixer(c, extra={"qk_norm": code(True, f"{TF} olmoe: q_norm and k_norm")})}
    a["ffns"] = {"moe": moe_ffn(c, experts=("num_experts",), active=("num_experts_per_tok",), d_expert=("intermediate_size",))}
    a["layout"] = uniform(c, "full", lambda i: "moe")
    return a


def granite_hybrid(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["mixers"] = {"full": attn_mixer(c)}
    a["ffns"] = {"dense": dense_ffn(c, key="shared_intermediate_size")}
    a["layout"] = from_layer_types(c, lambda i: "dense")
    return a


def gemma3(c: Cfg) -> dict[str, Any]:
    a = base(c)
    qk = {"qk_norm": c.f("use_qk_norm")} if c.get("use_qk_norm") else {}
    a["mixers"] = {"full": attn_mixer(c, extra=qk), "sliding": attn_mixer(c, window=c.f("sliding_window"), extra=qk)}
    a["ffns"] = {"dense": dense_ffn(c)}
    a["layout"] = from_layer_types(c, lambda i: "dense")
    a["norms_per_layer"] = code(4, "Gemma 2/3: sandwich norms (pre and post around attention and MLP)")
    return a


def cohere2(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["mixers"] = {"full": attn_mixer(c), "sliding": attn_mixer(c, window=c.f("sliding_window"))}
    a["ffns"] = {"dense": dense_ffn(c)}
    a["layout"] = from_layer_types(c, lambda i: "dense")
    a["norms_per_layer"] = code(1, "parallel block: one norm feeds attention and MLP")
    return a


def palm(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["mixers"] = {"full": attn_mixer(c)}
    a["ffns"] = {"dense": dense_ffn(c)}
    a["layout"] = uniform(c, "full", lambda i: "dense")
    a["norms_per_layer"] = code(1, "parallel block: one norm feeds attention and MLP")
    return a


def chinchilla(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["mixers"] = {"full": attn_mixer(c)}
    a["ffns"] = {"dense": dense_ffn(c, gated=False)}
    a["layout"] = uniform(c, "full", lambda i: "dense")
    return a


def glam(c: Cfg) -> dict[str, Any]:
    a = base(c)
    a["mixers"] = {"full": attn_mixer(c)}
    moe = moe_ffn(c, experts=("num_local_experts",), active=("num_experts_per_tok",), d_expert=("intermediate_size",), gated=False)
    moe["gated"] = c.f("moe_gated")
    a["ffns"] = {"dense": dense_ffn(c, gated=True), "moe": moe}
    k = int(c.get("moe_every"))
    per = [{"mixer": "full", "ffn": "moe" if i % k == k - 1 else "dense"} for i in range(n_layers(c))]
    a["layout"] = layout_field(per, "config", "num_hidden_layers, moe_every")
    return a


def phi3(c: Cfg) -> dict[str, Any]:
    return llama_like(c)


def granite(c: Cfg) -> dict[str, Any]:
    return llama_like(c)


HANDLERS: dict[str, Callable[[Cfg], dict[str, Any]]] = {
    "gpt2": gpt2,
    "llama": llama_like, "mistral": llama_like, "qwen2": llama_like, "qwen3": llama_like, "olmo2": llama_like,
    "olmo3": llama_like, "phi3": phi3, "smollm3": llama_like, "granite": granite, "ouro": llama_like, "nanbeige": llama_like,
    "qwen3_moe": qwen_moe, "qwen2_moe": qwen_moe,
    "deepseek_v2": deepseek, "deepseek_v3": deepseek, "kimi_k2": deepseek, "deepseek_v32": deepseek, "deepseek": deepseek_moe_v1,
    "glm4_moe": glm4_moe, "glm_moe_dsa": glm_dsa,
    "gpt_oss": gpt_oss, "git": grok,
    "qwen3_next": qwen3_next, "qwen3_5_moe_text": qwen3_next, "qwen3_5_text": qwen3_next, "qwen4_exp_text": qwen4_exp,
    "minimax_m2": minimax_m2, "minimax_text_01": minimax_text01,
    "kimi_linear": kimi_linear,
    "nemotron_h": nemotron_h,
    "mimo_v2_flash": mimo, "mimo_v2": mimo, "naive_n05_flash": mimo,
    "afmoe": afmoe,
    "gemma4_text": gemma4, "gemma4_unified_text": gemma4,
    "step3p5": step3p5,
    "bailing_hybrid": bailing_hybrid,
    "sarvam_moe": sarvam, "sarvam_mla": sarvam,
    "xlstm": xlstm,
    "longcat_flash_ngram": longcat,
    "mistral4": mistral4,
    "laguna": laguna,
    "zaya": zaya,
    "hy_v3": hy_v3, "hy_v4": hy_v4,
    "cohere2_moe": cohere2_moe,
    "lfm2": lfm2, "lfm2_moe": lfm2,
    "mellum": mellum,
    "minimax_m3": minimax_m3,
    "inkling_text": inkling,
    "Motif": motif,
    "solar_open2": solar_open2,
    "kolibri1": kolibri,
    "muse_glimmer_text": muse,
    "deepseek_v4": deepseek_v4, "deepseek_v41_text": deepseek_v4,
    "glm5_next_text": glm5_next,
    "mamba": mamba1, "mamba2": mamba2_hf,
    "rwkv": rwkv, "rwkv6": rwkv,
    "falcon": falcon, "RefinedWeb": falcon, "RefinedWebModel": falcon,
    "bloom": bloom, "gpt_neox": neox, "gptj": gptj, "opt": opt,
    "mixtral": mixtral, "jamba": jamba,
    "t5": t5, "switch_transformers": t5, "bert": bert,
    "minimax": minimax_text01,
    "olmoe": olmoe, "granitemoehybrid": granite_hybrid,
    "gemma3_text": gemma3, "cohere2": cohere2, "palm": palm, "chinchilla": chinchilla, "glam": glam,
}


def model_type_of(raw: dict[str, Any]) -> str:
    c = Cfg(raw)
    mt = c.model_type()
    if mt == "None" and "n_layers" in raw and "dim" in raw:
        return "mistral_params"
    if mt in ("kimi_k3",):
        return "kimi_linear"
    if mt in ("minimax_m3_vl",):
        return "minimax_m3"
    if mt in ("inkling_mm_model",):
        return "inkling_text"
    if mt == "longcat_flash_ngram" or "LongcatFlashNgramForCausalLM" in str(raw.get("architectures")):
        return "longcat_flash_ngram"
    if mt == "None" and "ssm_cfg" in raw:
        return "mamba2_ssm"
    return mt


def adapt(raw: dict[str, Any]) -> dict[str, Any]:
    mt = model_type_of(raw)
    c = Cfg(raw)
    if mt == "mistral_params":
        return mistral_params(c)
    if mt == "mamba2_ssm":
        return mamba2_ssm(c)
    if mt not in HANDLERS:
        raise KeyError(f"no handler for model_type {mt!r}")
    return HANDLERS[mt](c)


def strip(x: Any) -> Any:
    """Field tree -> plain values (what the cost model consumes)."""
    if isinstance(x, dict):
        if "v" in x and "st" in x:
            return strip(x["v"])
        return {k: strip(v) for k, v in x.items()}
    if isinstance(x, list):
        return [strip(i) for i in x]
    return x
