/*
 * src/lib/disagg/engine.ts
 *
 * Operation: a typed wrapper around Disaggregated_Inference_Sim's own
 *            JavaScript engine (`vendor/sim_engine.js`, vendored byte for
 *            byte at the commit in `vendor/VENDORED.json` by
 *            `pnpm vendor:sim`), plus a port of the simulator's capacity
 *            search (`search.analytic_capacity` and
 *            `search.max_sustainable_rate`), so the encoder-decoder chapter
 *            can re-measure results.md sections 16–18 in the browser.
 * Shapes:    simulate(cfg, rows) → SimResult; maxSustainableRate(cfg, w) →
 *            { rate, bracket, simulations }.
 * Intuition: a discrete-event simulation of a disaggregated serving
 *            cluster; the search bisects on the Poisson rate until at least
 *            90% of requests meet both SLOs. Every timestamp equals the
 *            Python package's (tests/unit/disagg/parity.test.ts) and every
 *            rate the search finds equals the recorded one
 *            (tests/unit/disagg/search.test.ts).
 * MDX:       /learn/09-encoder-decoder-and-ced.
 *
 * The engine is a classic script that sets `globalThis.DisaggSim`; importing
 * it for its side effect keeps the vendored file unmodified.
 */
import "./vendor/sim_engine.js";

/** One request of a workload: [arrival s, prompt tokens, output tokens]. */
export type Row = [number, number, number];

/** A recorded workload row: [unit exponential draw, prompt, output]. */
export type DrawRow = [number, number, number];

/** The engine's configuration keys (camelCase versions of SimConfig's) used here. */
export type SimConfig = {
  model: string;
  device: string;
  devicesPerInstance: number;
  mode: "disagg" | "colocated";
  nPrefill: number;
  nDecode: number;
  nColocated: number;
  link: string;
  ttftSlo: number;
  tpotSlo: number;
  powerCap?: number;
  dvfs?: boolean;
  decodeDevice?: string;
  prefillDevicesPerInstance?: number;
  lmHead?: "all" | "last";
  kvCompress?: string;
  kvCompressAt?: "transit" | "endpoint";
  cedEncoderLayers?: number;
  cedReplay?: number;
  cedReplayOn?: "prefill" | "decode";
};

export type SimRequest = {
  arrival: number;
  prompt: number;
  output: number;
  prefillStart: number | null;
  firstToken: number | null;
  prefillDone: number | null;
  kvStart: number | null;
  kvReady: number | null;
  decodeStart: number | null;
  finish: number | null;
};

export type SimInstance = {
  ec: number;
  em: number;
  busy: number;
  peakW: number;
  steps: number;
  batchSum: number;
};

export type SimResult = {
  reqs: SimRequest[];
  insts: SimInstance[];
  link: { energy: number; bytes: number; wait: number; transitJ: number };
};

type Dist = { mean: number; p50: number; p90: number; p99: number };

export type Summary = {
  ttft: Dist;
  tpot: Dist;
  sloAttain: number;
  energy: { totalJ: number };
};

/** A step's cost, as the engine's cost model returns it. */
export type StepCost = {
  flops: number;
  bytes: number;
  time: number;
  bound: string;
};

export type CostModel = {
  prefill: (lens: number[]) => StepCost;
  decode: (ctx: number[]) => StepCost;
  kvCap: number | null;
  fits: boolean;
};

type Model = {
  name: string;
  L: number;
  d: number;
  cedE: number;
  kvTok: number;
  matmul: number;
  cedPrompt: number;
  cedResident: number;
  weightBytes: number;
};

type Engine = {
  MODELS: Record<string, { name: string; L: number; d: number }>;
  DEVICES: Record<string, { name: string }>;
  LINKS: Record<string, { name: string; bw: number; lat: number }>;
  derive: (m: object) => Model;
  deviceFor: (key: string, cfg: object) => object;
  costModel: (
    model: Model,
    dev: object,
    n: number,
    overhead: number,
    powerCap?: number | null,
    dvfs?: boolean,
    sMin?: number,
    prefillOnly?: boolean,
  ) => CostModel;
  simulate: (cfg: SimConfig, rows: Row[]) => SimResult;
  summarise: (res: SimResult) => Summary;
};

/** The vendored engine's API. */
export const engine: Engine = (globalThis as unknown as { DisaggSim: Engine })
  .DisaggSim;

/** The simulator's defaults (SimConfig): 4×H100 per instance, InfiniBand NDR, TPOT SLO 25 ms. */
export const BASE: SimConfig = {
  model: "llama3-70b",
  device: "h100",
  devicesPerInstance: 4,
  mode: "disagg",
  nPrefill: 1,
  nDecode: 1,
  nColocated: 2,
  link: "ib-ndr",
  ttftSlo: 1.0,
  tpotSlo: 0.025,
};

/** SimConfig.step_overhead, max_prefill_tokens and max_decode_batch. */
const STEP_OVERHEAD = 0.5e-3;
const MAX_PREFILL_TOKENS = 8192;
const MAX_DECODE_BATCH = 256;

/** The engine's model with any CED overrides from the configuration applied (as `simulate` does). */
export function modelOf(cfg: SimConfig): Model {
  return engine.derive({
    ...engine.MODELS[cfg.model],
    ...(cfg.lmHead ? { lmHead: cfg.lmHead } : {}),
    ...(cfg.cedEncoderLayers != null ? { cedE: cfg.cedEncoderLayers } : {}),
    ...(cfg.cedReplay != null ? { cedW: cfg.cedReplay } : {}),
    ...(cfg.cedReplayOn ? { cedOn: cfg.cedReplayOn } : {}),
  });
}

/** The cost model of one instance (H100s), as `hardware.CostModel(model, H100_SXM, n, prefill_only=…)`. */
export function costModelOf(
  cfg: SimConfig,
  n: number,
  prefillOnly = false,
): CostModel {
  return engine.costModel(
    modelOf(cfg),
    engine.deviceFor(cfg.device, cfg),
    n,
    STEP_OVERHEAD,
    cfg.powerCap ?? null,
    !!cfg.dvfs,
    undefined,
    prefillOnly,
  );
}

/** Rebuild the arrivals at `rate` from recorded unit draws: Python's expovariate is −log(1−u) / rate. */
export function workloadAt(draws: DrawRow[], rate: number): Row[] {
  let t = 0.0;
  return draws.map(([e, p, o]) => {
    t += e / rate;
    return [t, p, o];
  });
}

/** Share of requests inside both SLOs (metrics.summarise → slo_attainment). */
export function attainment(cfg: SimConfig, rows: Row[]): number {
  return engine.summarise(engine.simulate(cfg, rows)).sloAttain;
}

/**
 * search.analytic_capacity(cfg, wl)["ceiling"] for a disaggregated
 * configuration: the optimistic throughput bound of each pool and the link.
 * As in Python, the cost model is the whole model's on `devicesPerInstance`
 * GPUs (not prefill-only, not the per-pool device count): it only brackets
 * the search, so the bisection's result does not depend on it.
 */
export function analyticCeiling(
  cfg: SimConfig,
  prompt: number,
  output: number,
): number {
  const cm = costModelOf(cfg, cfg.devicesPerInstance);
  const p = prompt;
  const o = output;
  const perBatch = Math.max(1, Math.floor(MAX_PREFILL_TOKENS / p));
  const prefillRate =
    perBatch / cm.prefill(new Array<number>(perBatch).fill(Math.trunc(p))).time;
  const b = Math.max(
    1,
    Math.min(MAX_DECODE_BATCH, Math.floor((cm.kvCap as number) / (p + o))),
  );
  const step = cm.decode(new Array<number>(b).fill(Math.trunc(p + o / 2))).time;
  const decodeRate = b / step / Math.max(1.0, o - 1);
  const link = engine.LINKS[cfg.link]!;
  const nbytes = p * modelOf(cfg).kvTok;
  const bounds = [
    cfg.nPrefill * prefillRate,
    cfg.nDecode * decodeRate,
    1 / (link.lat + nbytes / link.bw),
  ];
  return Math.min(...bounds);
}

export type SearchResult = {
  rate: number;
  bracket: [number, number];
  simulations: number;
};

/**
 * search.max_sustainable_rate: the highest Poisson rate whose SLO
 * attainment is ≥ `target`, by bisection to `relTol`, starting from the
 * analytic ceiling. `onRun` sees every simulation (rate, attainment).
 */
export function maxSustainableRate(
  cfg: SimConfig,
  draws: DrawRow[],
  prompt: number,
  output: number,
  {
    target = 0.9,
    relTol = 0.02,
    onRun,
  }: {
    target?: number;
    relTol?: number;
    onRun?: (rate: number, att: number) => void;
  } = {},
): SearchResult {
  const att = (rate: number): number => {
    const a = attainment(cfg, workloadAt(draws, rate));
    onRun?.(rate, a);
    return a;
  };
  let lo = 0.0;
  let hi = analyticCeiling(cfg, prompt, output);
  let runs = 0;
  while (att(hi) >= target) {
    runs += 1;
    lo = hi;
    hi = hi * 1.5;
  }
  runs += 1;
  while (hi - lo > relTol * hi) {
    const mid = (lo + hi) / 2;
    runs += 1;
    if (att(mid) >= target) lo = mid;
    else hi = mid;
  }
  return { rate: lo, bracket: [lo, hi], simulations: runs };
}
