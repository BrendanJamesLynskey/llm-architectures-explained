/**
 * What each model's data says about the axes of variation: attention type,
 * positions, normalisation, mixture-of-experts and the rest. Every feature
 * is derived from the record's fields (never typed in by hand), so the
 * table filters, the timeline and the diagrams always agree with the data.
 */
import type { LayerRun, Mixer, Spec } from "./costModel";
import { expandLayout } from "./costModel";
import type { ModelRecord, Position } from "./types";

/** Field tree -> plain values (what the cost model consumes). */
export function stripFields(x: unknown): unknown {
  if (Array.isArray(x)) return x.map(stripFields);
  if (x && typeof x === "object") {
    const o = x as Record<string, unknown>;
    if ("v" in o && "st" in o) return stripFields(o.v);
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(o)) out[k] = stripFields(v);
    return out;
  }
  return x;
}

export function specOf(m: ModelRecord): Spec | null {
  if (!m.arch) return null;
  return { ...(stripFields(m.arch) as Omit<Spec, "kind">), kind: m.kind };
}

export const FEATURES = [
  {
    id: "mha",
    group: "Attention",
    label: "MHA",
    description: "Multi-head attention: one key/value head per query head.",
  },
  {
    id: "gqa",
    group: "Attention",
    label: "GQA",
    description:
      "Grouped-query attention: several query heads share one key/value head.",
  },
  {
    id: "mqa",
    group: "Attention",
    label: "MQA",
    description:
      "Multi-query attention: all query heads share a single key/value head.",
  },
  {
    id: "mla",
    group: "Attention",
    label: "MLA",
    description:
      "Multi-head latent attention: keys and values cached as one low-rank latent vector.",
  },
  {
    id: "sliding",
    group: "Attention",
    label: "Sliding window",
    description: "Some layers attend only to the most recent W tokens.",
  },
  {
    id: "chunked",
    group: "Attention",
    label: "Chunked attention",
    description:
      "Some layers attend only to earlier tokens in their own fixed-size chunk.",
  },
  {
    id: "sparse",
    group: "Attention",
    label: "Sparse attention",
    description: "An indexer picks which cached tokens each query reads.",
  },
  {
    id: "compressed",
    group: "Attention",
    label: "Compressed KV",
    description:
      "Keys and values compressed along the sequence (DeepSeek V4 CSA/HCA).",
  },
  {
    id: "linear",
    group: "Attention",
    label: "Linear attention",
    description: "A fixed-size recurrent state instead of a growing cache.",
  },
  {
    id: "deltanet",
    group: "Attention",
    label: "DeltaNet / KDA",
    description:
      "Gated delta-rule linear attention (Gated DeltaNet, Kimi Delta Attention).",
  },
  {
    id: "ssm",
    group: "Attention",
    label: "Mamba (SSM)",
    description: "Selective state-space layers.",
  },
  {
    id: "conv",
    group: "Attention",
    label: "Short convolution",
    description: "Gated short-convolution token mixers (LFM2).",
  },
  {
    id: "hybrid",
    group: "Attention",
    label: "Hybrid",
    description: "More than one kind of token mixer in the stack.",
  },
  {
    id: "rope",
    group: "Position",
    label: "RoPE",
    description: "Rotary position embeddings.",
  },
  {
    id: "partial-rope",
    group: "Position",
    label: "Partial RoPE",
    description: "Rotary embeddings on only part of each head.",
  },
  {
    id: "nope",
    group: "Position",
    label: "NoPE layers",
    description: "Some or all attention layers use no positional encoding.",
  },
  {
    id: "alibi",
    group: "Position",
    label: "ALiBi",
    description: "Linear attention biases by distance.",
  },
  {
    id: "learned",
    group: "Position",
    label: "Learned absolute",
    description: "A learned embedding per position.",
  },
  {
    id: "relative-bias",
    group: "Position",
    label: "Relative bias",
    description: "A learned bias per relative distance (T5).",
  },
  {
    id: "pre-norm",
    group: "Normalisation",
    label: "Pre-norm",
    description: "A norm before each sub-block.",
  },
  {
    id: "post-norm",
    group: "Normalisation",
    label: "Post-norm",
    description: "Norms after each sub-block, inside the residual (OLMo 2).",
  },
  {
    id: "sandwich",
    group: "Normalisation",
    label: "Sandwich norm",
    description: "Norms both before and after each sub-block.",
  },
  {
    id: "post-ln",
    group: "Normalisation",
    label: "Post-LN",
    description:
      "The original Transformer: a norm after each residual addition.",
  },
  {
    id: "qk-norm",
    group: "Normalisation",
    label: "QK-norm",
    description: "Queries and keys normalised before the dot product.",
  },
  {
    id: "moe",
    group: "Feed-forward",
    label: "MoE",
    description:
      "Mixture of experts: each token uses a few of many feed-forward experts.",
  },
  {
    id: "shared-expert",
    group: "Feed-forward",
    label: "Shared expert",
    description: "One or more experts that every token uses.",
  },
  {
    id: "dense-prefix",
    group: "Feed-forward",
    label: "Dense first layers",
    description: "The first layers keep a dense feed-forward block.",
  },
  {
    id: "latent-moe",
    group: "Feed-forward",
    label: "Latent MoE",
    description: "Experts work in a smaller latent width.",
  },
  {
    id: "mtp",
    group: "Other",
    label: "MTP",
    description: "Multi-token prediction layers.",
  },
  {
    id: "looped",
    group: "Other",
    label: "Looped",
    description: "The layer stack is run more than once per token.",
  },
  {
    id: "parallel",
    group: "Other",
    label: "Parallel block",
    description: "Attention and the feed-forward block run side by side.",
  },
  {
    id: "kv-sharing",
    group: "Other",
    label: "Cross-layer KV sharing",
    description: "Some layers reuse another layer's keys and values.",
  },
  {
    id: "encoder-decoder",
    group: "Other",
    label: "Encoder-decoder",
    description: "A bidirectional encoder and a decoder with cross-attention.",
  },
  {
    id: "ced",
    group: "Other",
    label: "Causal encoder-decoder",
    description:
      "Prefill runs only the first half of the stack (DeepSeek V4.1).",
  },
] as const;

export type FeatureId = (typeof FEATURES)[number]["id"];

const LINEAR_TYPES = new Set(["linear", "deltanet", "kda", "rwkv", "mlstm"]);

function mixerFamily(m: Mixer): string {
  if (m.type === "attn" || m.type === "mla" || m.type === "csa")
    return "softmax";
  if (m.type === "mamba1" || m.type === "mamba2") return "ssm";
  if (m.type === "conv") return "conv";
  return "linear";
}

function usedMixers(spec: Spec): Mixer[] {
  const names = new Set<string>();
  for (const r of [...spec.layout, ...(spec.encoder_layout ?? [])]) {
    if (r.mixer !== "none") names.add(r.mixer);
  }
  return [...names].sort().map((n) => spec.mixers[n]!);
}

function usedFfns(spec: Spec): LayerRun["ffn"][] {
  return [...new Set(spec.layout.map((r) => r.ffn))];
}

export function features(m: ModelRecord): Set<FeatureId> {
  const out = new Set<FeatureId>();
  const f = m.facts;
  if (m.kind === "encoder-decoder") out.add("encoder-decoder");
  if (m.kind === "ced") out.add("ced");
  if (f.moe?.v || (f.experts?.v ?? 0) > 1) out.add("moe");
  const spec = specOf(m);
  if (spec) {
    const mixers = usedMixers(spec);
    const families = new Set(mixers.map(mixerFamily));
    if (families.size > 1) out.add("hybrid");
    for (const x of mixers) {
      if (x.type === "attn") {
        const kv = x.kv_heads!;
        if (kv === 1) out.add("mqa");
        else if (kv < x.heads!) out.add("gqa");
        else out.add("mha");
        if (x.window) out.add("sliding");
        if (x.chunk) out.add("chunked");
        if (x.indexer) out.add("sparse");
      }
      if (x.type === "mla") {
        out.add("mla");
        if (x.indexer) out.add("sparse");
        if (x.window) out.add("sliding");
      }
      if (x.type === "csa") {
        out.add("compressed");
        out.add("sparse");
        out.add("sliding");
      }
      if (LINEAR_TYPES.has(x.type)) out.add("linear");
      if (x.type === "deltanet" || x.type === "kda") out.add("deltanet");
      if (x.type === "mamba1" || x.type === "mamba2") out.add("ssm");
      if (x.type === "conv") out.add("conv");
    }
    const ffns = usedFfns(spec);
    const moe = Object.entries(spec.ffns).find(([, v]) => v.type === "moe");
    if (moe) {
      out.add("moe");
      const [, mf] = moe;
      if (mf.shared) out.add("shared-expert");
      if (mf.latent) out.add("latent-moe");
      // Dense layers before the MoE block, not interleaved with it (GLaM,
      // Jamba and Llama 4 alternate dense and MoE layers from the start).
      const kinds = expandLayout(spec)
        .filter((r) => r.mixer !== "none")
        .map((r) => spec.ffns[r.ffn]?.type);
      const firstMoe = kinds.indexOf("moe");
      const lastMoe = kinds.lastIndexOf("moe");
      if (
        kinds[0] === "dense" &&
        ffns.includes(moe[0]) &&
        !kinds.slice(firstMoe, lastMoe + 1).includes("dense")
      )
        out.add("dense-prefix");
    }
    if ((spec.mtp_layers ?? 0) > 0) out.add("mtp");
    if ((spec.loops ?? 1) > 1) out.add("looped");
    if (spec.layout.some((r) => r.kv_shared || r.kv_source === false))
      out.add("kv-sharing");
  }
  const pos = f.position?.v as Position | undefined;
  if (pos) {
    if (pos.scheme === "rope" || pos.scheme === "mrope") {
      out.add("rope");
      if (pos.rope_fraction !== null && pos.rope_fraction < 1)
        out.add("partial-rope");
    }
    // NoPE means attention without positions; recurrent-only models have
    // no attention to speak of.
    const hasAttention =
      spec !== null &&
      usedMixers(spec).some((x) => mixerFamily(x) === "softmax");
    if (pos.nope && hasAttention) out.add("nope");
    if (pos.scheme === "alibi") out.add("alibi");
    if (pos.scheme === "learned") out.add("learned");
    if (pos.scheme === "relative-bias") out.add("relative-bias");
  }
  const np = f.norm_placement?.v;
  if (np === "pre") out.add("pre-norm");
  if (np === "post") out.add("post-norm");
  if (np === "sandwich") out.add("sandwich");
  if (np === "post-ln") out.add("post-ln");
  if (np === "parallel") out.add("parallel");
  if (f.qk_norm?.v) out.add("qk-norm");
  return out;
}

/** Number of layers in the decoder stack (blocks, for Nemotron-H style). */
export function layerCount(spec: Spec): number {
  return expandLayout(spec).length;
}

/** "GQA 64q/4kv" style summary of the attention used. */
export function attentionSummary(m: ModelRecord): string {
  const spec = specOf(m);
  if (!spec) return "not disclosed";
  const parts: string[] = [];
  const runs = expandLayout(spec);
  const counts = new Map<string, number>();
  for (const r of runs) {
    if (r.mixer !== "none") counts.set(r.mixer, (counts.get(r.mixer) ?? 0) + 1);
  }
  for (const [name, n] of counts) {
    const x = spec.mixers[name]!;
    let label: string;
    if (x.type === "attn") {
      const kind =
        x.kv_heads === 1 ? "MQA" : x.kv_heads! < x.heads! ? "GQA" : "MHA";
      label = `${kind} ${x.heads}q/${x.kv_heads}kv`;
      if (x.window) label += `, window ${x.window}`;
      if (x.chunk) label += `, chunks of ${x.chunk}`;
      if (x.indexer) label += `, top-${x.indexer.topk}`;
    } else if (x.type === "mla") {
      label = `MLA ${x.heads}h, latent ${x.kv_lora_rank}`;
      if (x.indexer) label += `, top-${x.indexer.topk}`;
      if (x.window) label += `, window ${x.window}`;
    } else if (x.type === "csa") {
      label = `compressed ${x.heads}h, window ${x.window}`;
    } else {
      label = MIXER_NAMES[x.type] ?? x.type;
    }
    parts.push(counts.size > 1 ? `${n}× ${label}` : label);
  }
  return parts.join(" + ");
}

export const MIXER_NAMES: Record<string, string> = {
  attn: "Attention",
  mla: "MLA",
  csa: "Compressed attention",
  deltanet: "Gated DeltaNet",
  kda: "Kimi Delta Attention",
  linear: "Linear attention",
  mamba1: "Mamba",
  mamba2: "Mamba-2",
  conv: "Short convolution",
  rwkv: "RWKV",
  mlstm: "mLSTM",
};

export function released(m: ModelRecord): number | null {
  const v = m.facts.released.v;
  if (!v) return null;
  const [y, mo] = v.split("-");
  return Number(y) + (Number(mo ?? "1") - 1) / 12;
}
