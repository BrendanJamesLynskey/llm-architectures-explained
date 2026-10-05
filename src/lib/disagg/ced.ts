/*
 * src/lib/disagg/ced.ts
 *
 * Operation: the experiment of results.md sections 16–18 (brief 11), as
 *            configurations of the vendored engine: the three model variants,
 *            the four prompt:output workloads, the pool splits, the TTFT SLO
 *            rule, and the section 16 cost-model numbers.
 * Shapes:    cedConfig(variant, nPrefill, nDecode, slo, prefillGpus?) →
 *            SimConfig; prefillStep(variant, prompt) → StepCost.
 * Intuition: decoder-only Llama-3-70B against the same shape split into a
 *            40-layer causal encoder and a 40-layer decoder; a prompt token
 *            runs only the encoder (plus the decoder's K/V projections and a
 *            128-token replay).
 * MDX:       /learn/09-encoder-decoder-and-ced.
 *
 * Illustrative: the dense 70B shape split half and half is a proxy, not
 * DeepSeek-V4.1-Flash (a 552B MoE with compressed sparse attention).
 */
import {
  BASE,
  costModelOf,
  modelOf,
  type SimConfig,
  type StepCost,
} from "./engine";

export const VARIANTS = [
  {
    key: "decoder-only",
    label: "Decoder-only",
    cfg: { model: "llama3-70b" },
  },
  {
    key: "CED, replay on prefill",
    label: "CED, replay on prefill",
    cfg: { model: "llama3-70b-ced" },
  },
  {
    key: "CED, replay on decode",
    label: "CED, replay on decode",
    cfg: { model: "llama3-70b-ced", cedReplayOn: "decode" as const },
  },
] as const;

export type VariantKey = (typeof VARIANTS)[number]["key"];

/** Sections 17–18: mean prompt and output of each workload (cv 0.5, 1,000 requests, seed 1). */
export const WORKLOADS = [
  { prompt: 2048, output: 512 },
  { prompt: 4096, output: 256 },
  { prompt: 8192, output: 128 },
  { prompt: 16384, output: 64 },
] as const;

/** Section 17: six instances of 4×H100, split nP + (6 − nP)D. */
export const CLUSTER = 6;
/** Section 18: (prefill, decode) instances with 2-GPU prefill instances, 24 GPUs in all. */
export const SMALL_PREFILL = [
  [2, 5],
  [4, 4],
  [6, 3],
  [8, 2],
] as const;

export function variant(key: VariantKey): (typeof VARIANTS)[number] {
  return VARIANTS.find((v) => v.key === key)!;
}

/** The engine configuration of one cell of sections 17–18. */
export function cedConfig(
  key: VariantKey,
  nPrefill: number,
  nDecode: number,
  ttftSlo: number,
  prefillGpus?: number,
): SimConfig {
  return {
    ...BASE,
    ...variant(key).cfg,
    nPrefill,
    nDecode,
    ttftSlo,
    ...(prefillGpus ? { prefillDevicesPerInstance: prefillGpus } : {}),
  };
}

/** TTFT SLO = 5× the decoder-only model's unloaded prefill of the mean prompt (one 4×H100 instance). */
export function ttftSlo(prompt: number): number {
  return 5.0 * costModelOf(BASE, 4).prefill([prompt]).time;
}

/**
 * One prefill step of one prompt on 4×H100 (section 16): the decoder-only
 * model, CED with the replay on the prefill instance, and the encoder-only
 * prefill instance of CED with the replay on decode.
 */
export function prefillStep(
  which: "base" | "ced" | "enc",
  prompt: number,
): StepCost {
  if (which === "base") return costModelOf(BASE, 4).prefill([prompt]);
  if (which === "ced")
    return costModelOf(cedConfig("CED, replay on prefill", 1, 1, 1), 4).prefill(
      [prompt],
    );
  return costModelOf(
    cedConfig("CED, replay on decode", 1, 1, 1),
    4,
    true,
  ).prefill([prompt]);
}

/** Section 16's parameter counts: what a prompt token and a decode token touch. */
export function cedParams(): {
  decoderOnlyToken: number;
  cedPromptToken: number;
  cedDecodeToken: number;
  cedKvProj: number;
} {
  const base = modelOf(BASE);
  const ced = modelOf(cedConfig("CED, replay on prefill", 1, 1, 1));
  const kv = 8 * (ced.d / 64); // kv heads × head_dim of the Llama-3-70B shape
  return {
    decoderOnlyToken: base.matmul,
    cedPromptToken: ced.cedPrompt,
    cedDecodeToken: ced.matmul,
    cedKvProj: (ced.L - ced.cedE) * 2 * ced.d * kv,
  };
}

/** Section 16's memory table: resident weights and KV room of a prefill instance on n H100s. */
export function prefillRoom(
  which: "base" | "enc",
  n: number,
): { weights: number; kvTokens: number | null } {
  if (which === "base") {
    const cm = costModelOf(BASE, n);
    return { weights: modelOf(BASE).weightBytes, kvTokens: cm.kvCap };
  }
  const cfg = cedConfig("CED, replay on decode", 1, 1, 1);
  const cm = costModelOf(cfg, n, true);
  return { weights: modelOf(cfg).cedResident * 2, kvTokens: cm.kvCap };
}

/** Python's f"{x:.2f}" for the positive rates and ratios shown here (no exact ties occur). */
export function f2(x: number): string {
  return x.toFixed(2);
}
