/**
 * The learning layer's numbers (brief 44). This site's chapters quote their
 * numbers as text (tests/unit/chapters/numbers.test.ts recomputes each one),
 * so the self-checks get theirs here, from the same tested calls: the cost
 * model (src/lib/arch/costModel.ts), the chapter models
 * (src/lib/chapters/model.ts) and the simulator's recorded capacity search
 * (src/lib/disagg/cedGains.ts). A self-check cites one as `<Check v="…">`,
 * a reveal prints one as `<V v="…" />`; tests/unit/learn.test.ts checks
 * every one. Values are raw (bytes, counts, FLOPs, ratios); `formatValue`
 * prints them as the chapters do.
 */
import {
  decodeFlops,
  kvCache,
  params,
  prefillFlops,
  type Spec,
} from "@/lib/arch/costModel";
import { specOf } from "@/lib/arch/features";
import { CHAPTERS } from "@/lib/chapters/data";
import {
  asDecoder,
  depthWidthSpec,
  loopedKvBytes,
  loopedSpec,
  moeSpec,
  mtpExpectedTokens,
  mtpSpeedup,
  normProfile,
  ropeLongPairs,
} from "@/lib/chapters/model";
import { MODELS } from "@/lib/data";
import { cedGains } from "@/lib/disagg/cedGains";

export type Fmt =
  | "num"
  | "int"
  | "d2"
  | "d3"
  | "pct1"
  | "gib"
  | "kib"
  | "bn"
  | "gflop"
  | "x2"
  | "raw";

const GIB = 1024 ** 3;
const K128 = 131072;
const M1 = 1048576;

function trim(x: number, digits = 3): string {
  return Number(x.toPrecision(digits)).toLocaleString("en-GB", {
    maximumFractionDigits: 6,
  });
}

export function formatValue(v: number | string, fmt: Fmt = "num"): string {
  if (typeof v === "string") return v;
  switch (fmt) {
    case "int":
      return Math.round(v).toLocaleString("en-GB");
    case "d2":
      return v.toFixed(2);
    case "d3":
      return v.toFixed(3);
    case "pct1":
      return `${(100 * v).toFixed(1)}%`;
    case "gib":
      return `${trim(v / GIB)} GiB`;
    case "kib":
      return `${trim(v / 1024)} KiB`;
    case "bn":
      return `${trim(v / 1e9)}B`;
    case "gflop":
      return `${trim(v / 1e9)} GFLOP`;
    case "x2":
      return `${v.toFixed(2)}×`;
    case "raw":
      return String(v);
    default:
      return trim(v);
  }
}

const spec = (id: string): Spec => specOf(MODELS.find((m) => m.id === id)!)!;
const variant = (id: string): Spec =>
  CHAPTERS.attention.variants.find((x) => x.id === id)!.spec;
const rope = (id: string, ctx: number) => {
  const r = CHAPTERS.rope.find((x) => x.id === id)!;
  return ropeLongPairs(r.head_dim, r.fraction, r.theta, ctx);
};
const moe = (...a: [number, number, number, number, number]) =>
  params(moeSpec(CHAPTERS.moe.spec, ...a));
const attnShare = (s: Spec, ctx: number) => {
  const f = decodeFlops(s, ctx);
  return (f - 2 * params(s).matmul_active) / f;
};
const gain = (workload: string, variantName: string): number => {
  const g = cedGains().find(
    (x) => x.workload === workload && x.variant === variantName,
  );
  if (!g) throw new Error(`no CED gain ${workload} ${variantName}`);
  return g.gain;
};

const VALUES: Record<string, () => number> = {
  // chapter 1: one sequence's cache at 128K tokens, BF16, on Llama 3 8B's body
  "ch01.gqa128k": () => kvCache(variant("gqa"), K128, 2, 2).total_bytes,
  "ch01.mha128k": () => kvCache(variant("mha"), K128, 2, 2).total_bytes,
  "ch01.mhaOverGqa": () =>
    kvCache(variant("mha"), K128, 2, 2).total_bytes /
    kvCache(variant("gqa"), K128, 2, 2).total_bytes,
  "ch01.gqaFlops": () => decodeFlops(variant("gqa"), K128),
  "ch01.mlaFlops": () => decodeFlops(variant("mla"), K128),
  // chapter 2: RoPE pairs slower than a 32K context
  "ch02.mistral32k": () => rope("mistral-7b", 32768),
  "ch02.qwen32k": () => rope("qwen3-8b", 32768),
  // chapter 3: 32 layers, gain 1 and 2
  "ch03.preRms": () => normProfile("pre-norm", 32, 1).rms.at(-1)!,
  "ch03.preRms2": () => normProfile("pre-norm", 32, 2).rms.at(-1)!,
  "ch03.outRms2": () => normProfile("output-norm", 32, 2).rms.at(-1)!,
  // chapter 4: MoE on Llama 3 8B's body
  "ch04.dense": () => params(CHAPTERS.moe.spec).total,
  "ch04.mixtralTotal": () => moe(8, 2, 0, 1, 0).total,
  "ch04.mixtralActive": () => moe(8, 2, 0, 1, 0).active,
  "ch04.dsmTotal": () => moe(64, 6, 2, 8, 1).total,
  "ch04.dsmActive": () => moe(64, 6, 2, 8, 1).active,
  // chapter 5: depth against width
  "ch05.deepKv": () =>
    kvCache(depthWidthSpec(80, 2048), 32768, 2, 2).bytes_per_token_unbounded,
  "ch05.wideKv": () =>
    kvCache(depthWidthSpec(12, 8192), 32768, 2, 2).bytes_per_token_unbounded,
  "ch05.deepAttn": () => attnShare(depthWidthSpec(80, 2048), 32768),
  "ch05.wideAttn": () => attnShare(depthWidthSpec(12, 8192), 32768),
  // chapter 6: long context on real models
  "ch06.l405b128k": () =>
    kvCache(spec("llama-3.1-405b"), K128, 2, 2).total_bytes,
  "ch06.l405b1m": () => kvCache(spec("llama-3.1-405b"), M1, 2, 2).total_bytes,
  "ch06.v4flash1m": () =>
    kvCache(spec("deepseek-v4-flash"), M1, 2, 2).total_bytes,
  "ch06.l405bAttn1m": () => attnShare(spec("llama-3.1-405b"), M1),
  // chapter 7: DeepSeek-V3's MTP module
  "ch07.tokens085": () => mtpExpectedTokens(0.85, 1),
  "ch07.speedup085": () =>
    mtpSpeedup(spec("deepseek-v3"), 8192, 1, 0.85, 2, 2).speedup,
  // chapter 8: Ouro 2.6B looped 1 and 4 times, 8K context
  "ch08.flops1": () => decodeFlops(loopedSpec(CHAPTERS.loops.spec, 1), 8192),
  "ch08.flops4": () => decodeFlops(loopedSpec(CHAPTERS.loops.spec, 4), 8192),
  "ch08.kvAll4": () =>
    loopedKvBytes(loopedSpec(CHAPTERS.loops.spec, 4), 8192, 2, true),
  "ch08.kvLast4": () =>
    loopedKvBytes(loopedSpec(CHAPTERS.loops.spec, 4), 8192, 2, false),
  // chapter 9: CED's prefill and capacity
  "ch09.prefill8k": () =>
    prefillFlops(CHAPTERS.ced.spec, 8192) /
    prefillFlops(asDecoder(CHAPTERS.ced.spec), 8192),
  "ch09.prefill1m": () =>
    prefillFlops(CHAPTERS.ced.spec, M1) /
    prefillFlops(asDecoder(CHAPTERS.ced.spec), M1),
  "ch09.gainPrefill": () => gain("2048:512", "CED, replay on prefill"),
  "ch09.gainLong": () => gain("4096:256", "CED, replay on prefill"),
};

export const LEARN_VALUE_NAMES = Object.keys(VALUES);

const CACHE = new Map<string, number>();

export function learnValue(name: string): number {
  const cached = CACHE.get(name);
  if (cached !== undefined) return cached;
  const f = VALUES[name];
  if (!f) throw new Error(`no learn value ${name}`);
  const v = f();
  if (!Number.isFinite(v)) throw new Error(`learn value ${name} is ${v}`);
  CACHE.set(name, v);
  return v;
}

/** The adapter's lookup: on this site every path is a learn value. */
export const lookup = learnValue;
