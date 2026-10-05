/**
 * The spec builders and closed forms behind the concept chapters: a line by
 * line TypeScript port of reference/chapter_model.py.
 * tests/unit/chapters/model.test.ts checks every value against the fixtures
 * scripts/make_chapter_fixtures.py writes: exactly, except the RoPE
 * wavelengths (a `pow`, which libms may round differently in the last bit),
 * which compare to 1e-12 relative.
 *
 * Models, not measurements: see the Python module's docstring for the
 * idealisations (the norm profile's variance argument, the MTP speed-up's
 * independent acceptance and memory-bound decode).
 */
import type { LayerRun, Spec } from "@/lib/arch/costModel";
import { decodeBytes, kvCache, layerParams } from "@/lib/arch/costModel";

const clone = <T>(x: T): T => JSON.parse(JSON.stringify(x)) as T;

// ---------------------------------------------------------------------------
// Spec builders
// ---------------------------------------------------------------------------

/** Total layers of a run-length layout. */
export function layerTotal(spec: Spec): number {
  return spec.layout.reduce((n, r) => n + r.n, 0);
}

/** Replace a dense model's FFN with a mixture of experts (chapter_model.moe_spec). */
export function moeSpec(
  base: Spec,
  experts: number,
  active: number,
  shared: number,
  granularity: number,
  densePrefix: number,
): Spec {
  const s = clone(base);
  const dense = s.ffns.dense!;
  const layers = layerTotal(s);
  const mixer = s.layout[0]!.mixer;
  if (experts <= 1) return s;
  const dExpert = Math.floor(dense.d_ff! / granularity);
  const moe: Spec["ffns"][string] = {
    type: "moe",
    experts,
    active,
    d_expert: dExpert,
    gated: dense.gated ?? true,
  };
  if (shared > 0) {
    moe.shared = shared;
    moe.d_shared = dExpert;
  }
  s.ffns = { dense, moe };
  const layout: LayerRun[] = [];
  if (densePrefix > 0) layout.push({ mixer, ffn: "dense", n: densePrefix });
  layout.push({ mixer, ffn: "moe", n: layers - densePrefix });
  s.layout = layout;
  return s;
}

/** A Llama-style decoder of a given depth and width (chapter_model.depth_width_spec). */
export function depthWidthSpec(
  layers: number,
  dModel: number,
  vocab = 128256,
): Spec {
  const heads = Math.floor(dModel / 128);
  return {
    kind: "decoder",
    d_model: dModel,
    vocab,
    tied_embeddings: false,
    mixers: {
      full: {
        type: "attn",
        heads,
        kv_heads: Math.min(8, heads),
        head_dim: 128,
      },
    },
    ffns: {
      dense: {
        type: "dense",
        d_ff: Math.floor(Math.floor((7 * dModel) / 2) / 256) * 256,
        gated: true,
      },
    },
    layout: [{ mixer: "full", ffn: "dense", n: layers }],
  };
}

export function loopedSpec(base: Spec, loops: number): Spec {
  const s = clone(base);
  if (loops > 1) s.loops = loops;
  else delete s.loops;
  return s;
}

/** The same stack run as an ordinary decoder (chapter_model.as_decoder). */
export function asDecoder(spec: Spec): Spec {
  const s = clone(spec);
  s.kind = "decoder";
  delete s.ced_encoder_layers;
  delete s.ced_window;
  return s;
}

// ---------------------------------------------------------------------------
// Closed forms
// ---------------------------------------------------------------------------

export function rotaryDims(headDim: number, fraction: number): number {
  return Math.floor(Math.trunc(headDim * fraction) / 2) * 2;
}

/** Wavelength in tokens of each rotated pair i: 2π θ^(2i/d_rot). */
export function ropeWavelengths(
  headDim: number,
  fraction: number,
  theta: number,
): number[] {
  const d = rotaryDims(headDim, fraction);
  const out: number[] = [];
  for (let i = 0; i < d / 2; i++)
    out.push(2.0 * Math.PI * theta ** ((2.0 * i) / d));
  return out;
}

/** Pairs whose wavelength exceeds the context: they turn less than once across it. */
export function ropeLongPairs(
  headDim: number,
  fraction: number,
  theta: number,
  context: number,
): number {
  return ropeWavelengths(headDim, fraction, theta).filter((w) => w > context)
    .length;
}

export const PLACEMENTS = ["post-ln", "pre-norm", "output-norm"] as const;
export type Placement = (typeof PLACEMENTS)[number];

export type NormProfile = {
  rms: number[];
  update: number[];
  embedding_share: number;
};

/** Residual RMS, relative update and embedding share per sub-block (chapter_model.norm_profile). */
export function normProfile(
  placement: Placement,
  layers: number,
  gain: number,
  gamma = 1.0,
): NormProfile {
  let v = 1.0;
  let emb = 1.0;
  const rms: number[] = [];
  const update: number[] = [];
  for (let k = 0; k < 2 * layers; k++) {
    if (placement === "post-ln") {
      update.push(gain / Math.sqrt(v));
      const total = v + gain * gain;
      emb = emb / Math.sqrt(total);
      v = 1.0;
    } else {
      const add = placement === "pre-norm" ? gain : gamma;
      update.push(add / Math.sqrt(v));
      v = v + add * add;
      emb = 1.0 / Math.sqrt(v);
    }
    rms.push(Math.sqrt(v));
  }
  return { rms, update, embedding_share: emb };
}

/** 1 + a + a² + … + a^depth. */
export function mtpExpectedTokens(acceptance: number, depth: number): number {
  let total = 0.0;
  let term = 1.0;
  for (let k = 0; k < depth + 1; k++) {
    total = total + term;
    term = term * acceptance;
  }
  return total;
}

/** Weights one MTP module touches per token (chapter_model.mtp_module_active). */
export function mtpModuleActive(spec: Spec): number {
  const d = spec.d_model;
  const run = spec.mtp_layer ?? spec.layout[spec.layout.length - 1]!;
  const [m, , a] = layerParams(spec, run);
  return m + a + 2.0 * d * d + 4.0 * d;
}

export type MtpSpeedup = {
  tokens_per_step: number;
  bytes_per_step: number;
  bytes_per_token: number;
  speedup: number;
};

export function mtpSpeedup(
  spec: Spec,
  context: number,
  depth: number,
  acceptance: number,
  weightBytes: number,
  kvBytes: number,
): MtpSpeedup {
  const main = decodeBytes(spec, context, weightBytes, kvBytes).total;
  const extra = depth * mtpModuleActive(spec) * weightBytes;
  const tokens = mtpExpectedTokens(acceptance, depth);
  const step = main + extra;
  return {
    tokens_per_step: tokens,
    bytes_per_step: step,
    bytes_per_token: step / tokens,
    speedup: (tokens * main) / step,
  };
}

/** KV of a looped model: one cache per pass, or one shared cache (chapter_model.looped_kv_bytes). */
export function loopedKvBytes(
  spec: Spec,
  context: number,
  kvBytes: number,
  perLoopCache: boolean,
): number {
  const kv = kvCache(spec, context, kvBytes).kv_bytes;
  const loops = spec.loops || 1;
  return perLoopCache ? loops * kv : kv;
}
