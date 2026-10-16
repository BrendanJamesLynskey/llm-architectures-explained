/**
 * The states behind brief 41's three animations, as pure functions; the
 * TypeScript port of reference/visuals.py (tests/unit/chapters/visuals.test.ts
 * compares every state with tests/fixtures/visual_fixtures.json).
 *
 * - `ropeBank`: rotary position embedding as a bank of phasors. Pair i of a
 *   head turns by m·ω_i at position m, with ω_i = θ^(−2i/d). With the query
 *   and the key equal in every pair (unit vectors, the best case), the score
 *   of a query at m against a key at n is the mean over the pairs of
 *   cos((m − n)·ω_i), which depends only on m − n.
 * - `headSharing`: MHA, GQA, MQA and MLA on Llama 3 8B's body, with the KV
 *   bytes per token from the cost model.
 * - `moeRouting`: top-k routing by sigmoid affinity, with DeepSeek-V3's
 *   auxiliary-loss-free balancing (arXiv:2412.19437, §2.1.2): a bias per
 *   expert, used only to choose the experts, nudged by γ after each step.
 */
import { kvCache } from "@/lib/arch/costModel";
import type { Spec } from "@/lib/arch/costModel";

// ─── RoPE ───────────────────────────────────────────────────────────────

export const ROPE_POSITIONS = 96;
export const ROPE_SHOWN = 8;
/** "both": the query is this many tokens after the key. */
export const ROPE_GAP = 8;

export type RopeMode = "query" | "both";

export type RopePair = {
  i: number;
  omega: number;
  /** Tokens per full turn, 2π/ω. */
  wavelength: number;
  angQ: number;
  angK: number;
  /** cos·cos + sin·sin of the two rotated unit vectors: cos(angQ − angK). */
  contrib: number;
};

export type RopeState = {
  t: number;
  m: number;
  n: number;
  /** Mean over every pair of the head (not only the ones drawn). */
  score: number;
  /** The pairs drawn, spread from the fastest to the slowest. */
  pairs: RopePair[];
};

export function ropeOmegas(headDim: number, theta: number): number[] {
  const d = headDim;
  const out: number[] = [];
  for (let i = 0; i < d / 2; i++) out.push(theta ** ((-2.0 * i) / d));
  return out;
}

export function ropeShown(pairs: number, shown = ROPE_SHOWN): number[] {
  return Array.from({ length: shown }, (_, j) =>
    Math.round((j * (pairs - 1)) / (shown - 1)),
  );
}

export function ropeBank(
  headDim: number,
  theta: number,
  mode: RopeMode,
  positions = ROPE_POSITIONS,
): RopeState[] {
  const om = ropeOmegas(headDim, theta);
  const show = new Set(ropeShown(om.length));
  const out: RopeState[] = [];
  for (let t = 0; t < positions; t++) {
    const [m, n] = mode === "query" ? [t, 0] : [t + ROPE_GAP, t];
    let total = 0.0;
    const pairs: RopePair[] = [];
    om.forEach((w, i) => {
      const aq = m * w;
      const ak = n * w;
      const c = Math.cos(aq) * Math.cos(ak) + Math.sin(aq) * Math.sin(ak);
      total += c;
      if (show.has(i))
        pairs.push({
          i,
          omega: w,
          wavelength: (2.0 * Math.PI) / w,
          angQ: aq,
          angK: ak,
          contrib: c,
        });
    });
    out.push({ t, m, n, score: total / om.length, pairs });
  }
  return out;
}

const tokens = (w: number): string =>
  w >= 1e9
    ? `${(w / 1e9).toFixed(1)}B`
    : w >= 1e6
      ? `${(w / 1e6).toFixed(1)}M`
      : w >= 1e4
        ? `${Math.round(w / 1e3)}K`
        : w.toFixed(w >= 100 ? 0 : 1);

export function ropeCaption(s: RopeState, mode: RopeMode): string {
  const fast = s.pairs[0]!;
  const slow = s.pairs[s.pairs.length - 1]!;
  const d = s.m - s.n;
  const apart = `${d} token${d === 1 ? "" : "s"} apart`;
  const head =
    mode === "query"
      ? `Query at position ${s.m}, key at ${s.n}: ${apart}.`
      : `Query at ${s.m}, key at ${s.n}: both moved, still ${apart}.`;
  return `${head} The fastest pair turns once every ${tokens(fast.wavelength)} tokens, the slowest once every ${tokens(slow.wavelength)} tokens; the score (mean of every pair's cos(Δφ)) is ${s.score.toFixed(3)}.`;
}

// ─── head sharing ──────────────────────────────────────────────────────

export type SharingId = "mha" | "gqa" | "mqa" | "mla";
export const SHARING: readonly SharingId[] = ["mha", "gqa", "mqa", "mla"];

export type Sharing = {
  id: SharingId;
  label: string;
  heads: number;
  /** K/V units cached per token per layer: KV heads, or MLA's one latent. */
  units: number;
  /** Numbers each unit holds per token: K and V (2 × head_dim), or the latent plus the rotary key. */
  unitElems: number;
  layers: number;
  bytesPerToken: number;
  bytesPerTokenLayer: number;
};

export function headSharing(
  variants: readonly { id: string; label: string; spec: Spec }[],
): Sharing[] {
  const out: Sharing[] = [];
  for (const v of variants) {
    if (!(SHARING as readonly string[]).includes(v.id)) continue;
    const s = v.spec;
    const mx = Object.values(s.mixers)[0]! as unknown as Record<
      string,
      number | string
    >;
    let layers = 0;
    for (const r of s.layout) layers += r.n;
    const kv = kvCache(s, 1.0, 2.0);
    const mla = mx.type === "mla";
    out.push({
      id: v.id as SharingId,
      label: v.label,
      heads: mx.heads as number,
      units: mla ? 1 : (mx.kv_heads as number),
      unitElems: mla
        ? (mx.kv_lora_rank as number) + (mx.qk_rope as number)
        : 2 * (mx.head_dim as number),
      layers,
      bytesPerToken: kv.bytes_per_token_unbounded,
      bytesPerTokenLayer: kv.bytes_per_token_unbounded / layers,
    });
  }
  return out;
}

const kib = (b: number): string =>
  b >= 1024 ? `${(b / 1024).toFixed(b % 1024 ? 1 : 0)} KiB` : `${b} B`;

export function sharingCaption(v: Sharing, base: Sharing): string {
  const per = v.heads / v.units;
  const what =
    v.id === "mla"
      ? `every head reads one shared latent of ${v.unitElems} numbers, decompressed on the fly`
      : v.units === v.heads
        ? "every query head has its own key and value"
        : `each key/value head serves ${per} query heads`;
  const vs =
    v.id === "mha"
      ? "the baseline"
      : `${(base.bytesPerToken / v.bytesPerToken).toFixed(1)}× less than MHA`;
  return `${v.label}: ${what}. KV cache ${kib(v.bytesPerToken)} per token over ${v.layers} layers, ${vs}.`;
}

// ─── MoE routing ───────────────────────────────────────────────────────

export const MOE = {
  experts: 8,
  top_k: 2,
  tokens: 64,
  dim: 8,
  steps: 32,
  gamma: 0.02,
  hot: [0, 1, 2],
  spread: 0.6,
  seed: 7,
};
export type MoeCfg = typeof MOE;

export type Route = { experts: number[]; gates: number[] };

export type MoeState = {
  step: number;
  routes: Route[];
  loads: number[];
  /** The biases this step chose with (they change after it). */
  bias: number[];
  /** Busiest expert's load over the mean. */
  imbalance: number;
};

function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normal(rng: () => number): () => number {
  return () => {
    const u1 = Math.max(rng(), 1e-12);
    const u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
}

export function moeRouting(gamma: number, cfg: MoeCfg = MOE): MoeState[] {
  const { experts: E, top_k: K, tokens: T, dim: D } = cfg;
  const g = normal(mulberry32(cfg.seed));
  const cent = Array.from({ length: E }, () =>
    Array.from({ length: D }, () => g()),
  );
  const bias = new Array<number>(E).fill(0.0);
  const mean = (T * K) / E;
  const out: MoeState[] = [];
  for (let step = 0; step < cfg.steps; step++) {
    const tok = normal(mulberry32(cfg.seed * 1000 + step + 1));
    const loads = new Array<number>(E).fill(0);
    const routes: Route[] = [];
    for (let t = 0; t < T; t++) {
      const c = cent[cfg.hot[t % cfg.hot.length]!]!;
      const u = Array.from({ length: D }, (_, j) => c[j]! + cfg.spread * tok());
      const s: number[] = [];
      for (let e = 0; e < E; e++) {
        let x = 0.0;
        for (let j = 0; j < D; j++) x += u[j]! * cent[e]![j]!;
        s.push(1.0 / (1.0 + Math.exp(-x / Math.sqrt(D))));
      }
      const order = Array.from({ length: E }, (_, e) => e).sort(
        (a, b) => s[b]! + bias[b]! - (s[a]! + bias[a]!) || a - b,
      );
      const pick = order.slice(0, K);
      let tot = 0.0;
      for (const e of pick) tot += s[e]!;
      routes.push({ experts: pick, gates: pick.map((e) => s[e]! / tot) });
      for (const e of pick) loads[e]! += 1;
    }
    out.push({
      step,
      routes,
      loads,
      bias: bias.slice(),
      imbalance: Math.max(...loads) / mean,
    });
    for (let e = 0; e < E; e++) {
      if (loads[e]! > mean) bias[e]! -= gamma;
      else if (loads[e]! < mean) bias[e]! += gamma;
    }
  }
  return out;
}

export function moeCaption(
  s: MoeState,
  cfg: MoeCfg,
  balanced: boolean,
): string {
  const mean = (cfg.tokens * cfg.top_k) / cfg.experts;
  let hi = 0;
  let lo = 0;
  s.loads.forEach((l, e) => {
    if (l > s.loads[hi]!) hi = e;
    if (l < s.loads[lo]!) lo = e;
  });
  const tail = balanced
    ? ` Next step the bias of every expert above ${mean} tokens drops by ${cfg.gamma}, of every one below rises by ${cfg.gamma}.`
    : " No bias: the hot experts stay hot.";
  return `Step ${s.step + 1}: ${cfg.tokens} tokens each pick their top ${cfg.top_k} of ${cfg.experts} experts. Busiest: expert ${hi} with ${s.loads[hi]} (${s.imbalance.toFixed(2)}× the mean of ${mean}); quietest: expert ${lo} with ${s.loads[lo]}.${tail}`;
}
