/**
 * Architecture cost model: a line-by-line TypeScript port of
 * reference/arch_model.py.
 *
 * Parameter counts, KV-cache and state bytes, prefill and decode FLOPs and
 * decode-step memory traffic for any normalised architecture in
 * data/models/*.yaml. tests/unit/costModel.test.ts checks every number
 * against fixtures the Python reference writes (scripts/make_fixtures.py)
 * with exact equality, so keep the operation order identical to the Python:
 * floating-point addition is not associative.
 */

export type LayerRun = {
  mixer: string;
  ffn: string;
  n: number;
  ratio?: number;
  kv_source?: boolean;
  kv_shared?: boolean;
  indexer?: boolean;
};

export type Indexer = { heads: number; head_dim: number; topk: number };

export type Mixer = {
  type: string;
  heads?: number;
  kv_heads?: number;
  head_dim?: number;
  v_head_dim?: number | null;
  window?: number | null;
  gate?: string | null;
  k_eq_v?: boolean;
  bias?: boolean;
  qk_norm?: boolean;
  q_lora_rank?: number | null;
  kv_lora_rank?: number;
  qk_nope?: number;
  qk_rope?: number;
  indexer?: Indexer | null;
  o_lora_rank?: number;
  o_groups?: number;
  k_heads?: number;
  v_heads?: number;
  k_head_dim?: number;
  conv_kernel?: number;
  state?: number;
  groups?: number;
  d_inner?: number;
  dt_rank?: number;
  kernel?: number;
  mats?: number;
  qk_dim?: number;
  v_dim?: number;
  [k: string]: unknown;
};

export type Ffn = {
  type: string;
  d_ff?: number;
  gated?: boolean;
  bias?: boolean;
  receptance?: boolean;
  experts?: number;
  active?: number;
  d_expert?: number;
  shared?: number;
  d_shared?: number;
  latent?: number | null;
  dense_parallel_d_ff?: number | null;
};

export type Spec = {
  kind: string;
  d_model: number;
  vocab: number;
  tied_embeddings?: boolean;
  extra_embedding_params?: number;
  norms_per_layer?: number;
  mixers: Record<string, Mixer>;
  ffns: Record<string, Ffn>;
  layout: LayerRun[];
  encoder_layout?: LayerRun[];
  cross_mixer?: string;
  mtp_layers?: number;
  mtp_layer?: LayerRun;
  loops?: number;
  ced_encoder_layers?: number;
  ced_window?: number;
};

const num = (x: number | null | undefined): number => (x ? x : 0);

export function expandLayout(
  spec: Spec,
  key: "layout" | "encoder_layout" = "layout",
): LayerRun[] {
  const out: LayerRun[] = [];
  for (const run of spec[key] ?? []) {
    for (let i = 0; i < run.n; i++) out.push(run);
  }
  return out;
}

/** sum_{t=1..n} min(t, w); w <= 0 means no limit. */
export function sumMin(n: number, w: number): number {
  if (w <= 0 || n <= w) return (n * (n + 1.0)) / 2.0;
  return (w * (w + 1.0)) / 2.0 + (n - w) * w;
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

export function mixerParams(m: Mixer, d: number, kvShared = false): number {
  const t = m.type;
  if (t === "attn" && kvShared) {
    const h = m.heads!;
    const hd = m.head_dim!;
    const vd = m.v_head_dim || m.head_dim!;
    return d * h * hd + h * vd * d;
  }
  if (t === "attn") {
    const h = m.heads!;
    const kv = m.kv_heads!;
    const hd = m.head_dim!;
    const vd = m.v_head_dim || m.head_dim!;
    let q = d * h * hd;
    if (m.gate === "elementwise") q = q + d * h * vd;
    else if (m.gate === "headwise") q = q + d * h;
    const k = d * kv * hd;
    const v = m.k_eq_v ? 0.0 : d * kv * vd;
    const o = h * vd * d;
    let p = q + k + v + o;
    if (m.bias) p = p + h * hd + kv * hd + kv * vd;
    if (m.qk_norm) p = p + 2.0 * hd;
    const ix = m.indexer;
    if (ix) {
      const ih = ix.heads;
      const idd = ix.head_dim;
      p = p + d * ih * idd + d * idd + d * ih;
    }
    return p;
  }
  if (t === "mla") {
    const h = m.heads!;
    const qk = m.qk_nope! + m.qk_rope!;
    const vd = m.v_head_dim!;
    const rKv = m.kv_lora_rank!;
    const rope = m.qk_rope!;
    let q: number;
    if (m.q_lora_rank) {
      const rQ = m.q_lora_rank;
      q = d * rQ + rQ + rQ * h * qk;
    } else {
      q = d * h * qk;
    }
    const kv = d * (rKv + rope) + rKv + rKv * h * (m.qk_nope! + vd);
    const o = h * vd * d;
    let p = q + kv + o;
    if (m.gate === "elementwise") p = p + d * h * vd;
    const ix = m.indexer;
    if (ix) {
      const ih = ix.heads;
      const idd = ix.head_dim;
      const qIn = m.q_lora_rank ? m.q_lora_rank : d;
      p = p + qIn * ih * idd + d * idd + d * ih;
    }
    return p;
  }
  if (t === "csa") {
    const h = m.heads!;
    const hd = m.head_dim!;
    const rQ = m.q_lora_rank!;
    const rO = m.o_lora_rank!;
    const g = m.o_groups!;
    const q = d * rQ + rQ * h * hd;
    const kv = d * hd;
    const o = h * hd * rO + g * rO * d;
    let p = q + kv + o;
    const ix = m.indexer;
    if (ix) {
      const ih = ix.heads;
      const idd = ix.head_dim;
      p = p + rQ * ih * idd + d * idd + d * ih;
    }
    return p;
  }
  if (t === "deltanet" || t === "kda") {
    const kh = m.k_heads!;
    const vh = m.v_heads!;
    const kd = m.k_head_dim!;
    const vdd = m.v_head_dim!;
    const ck = num(m.conv_kernel);
    const qkDim = kh * kd;
    const vDim = vh * vdd;
    let proj = d * (2.0 * qkDim + vDim);
    if (t === "deltanet") {
      proj = proj + d * vDim + d * 2.0 * vh;
    } else {
      proj = proj + d * vDim + d * vh + 2.0 * d * kd + 2.0 * kd * qkDim;
    }
    const conv = (2.0 * qkDim + vDim) * ck;
    const out = vDim * d;
    return proj + conv + out + 2.0 * vh;
  }
  if (t === "linear") {
    return 5.0 * d * m.heads! * m.head_dim!;
  }
  if (t === "mamba2") {
    const nh = m.heads!;
    const hd = m.head_dim!;
    const n = m.state!;
    const g = m.groups!;
    const ck = m.conv_kernel!;
    const di = nh * hd;
    const inProj = d * (2.0 * di + 2.0 * g * n + nh);
    const conv = (di + 2.0 * g * n) * ck;
    return inProj + conv + 3.0 * nh + di + di * d;
  }
  if (t === "mamba1") {
    const di = m.d_inner!;
    const n = m.state!;
    const ck = m.conv_kernel!;
    const r = m.dt_rank!;
    return (
      d * 2.0 * di +
      di * ck +
      di * (r + 2.0 * n) +
      r * di +
      di * n +
      di +
      di * d
    );
  }
  if (t === "conv") {
    const ck = m.kernel!;
    return d * 3.0 * d + d * ck + d * d;
  }
  if (t === "rwkv") return m.mats! * d * d;
  if (t === "mlstm") {
    const qk = m.qk_dim!;
    const vd = m.v_dim!;
    const h = m.heads!;
    return d * qk * 2.0 + d * vd + d * vd + 2.0 * d * h + vd * d;
  }
  if (t === "none") return 0.0;
  throw new Error(`unknown mixer type ${t}`);
}

export function ffnParams(f: Ffn, d: number): [number, number] {
  const t = f.type;
  if (t === "none") return [0.0, 0.0];
  if (t === "dense") {
    const mats = (f.gated ?? true) ? 3.0 : 2.0;
    let p = mats * d * f.d_ff!;
    if (f.receptance) p = p + d * d;
    if (f.bias) p = p + f.d_ff! + d;
    return [p, p];
  }
  if (t === "moe") {
    const mats = (f.gated ?? true) ? 3.0 : 2.0;
    const e = f.experts!;
    const k = f.active!;
    const de = f.d_expert!;
    const width = f.latent ? f.latent : d;
    const perExpert = mats * width * de;
    const router = d * e;
    let shared = 0.0;
    if (f.shared) {
      const ds = f.d_shared || f.d_expert!;
      shared = f.shared * mats * d * ds;
    }
    const latent = f.latent ? 2.0 * d * width : 0.0;
    let parallel = 0.0;
    if (f.dense_parallel_d_ff) parallel = mats * d * f.dense_parallel_d_ff;
    const total = e * perExpert + router + shared + latent + parallel;
    const active = k * perExpert + router + shared + latent + parallel;
    return [total, active];
  }
  throw new Error(`unknown ffn type ${t}`);
}

export type Params = {
  embedding: number;
  norms: number;
  lm_head: number;
  extra_embedding: number;
  mixers: number;
  ffn_total: number;
  ffn_active: number;
  cross_attention: number;
  mtp: number;
  total: number;
  active: number;
  matmul_active: number;
  non_embedding_total: number;
  non_embedding_active: number;
};

export function layerParams(
  spec: Spec,
  run: LayerRun,
): [number, number, number] {
  const d = spec.d_model;
  const mix =
    run.mixer !== "none"
      ? mixerParams(spec.mixers[run.mixer]!, d, Boolean(run.kv_shared))
      : 0.0;
  let ft = 0.0;
  let fa = 0.0;
  if ((run.ffn ?? "none") !== "none") {
    [ft, fa] = ffnParams(spec.ffns[run.ffn]!, d);
  }
  return [mix, ft, fa];
}

export function params(spec: Spec): Params {
  const d = spec.d_model;
  const vocab = spec.vocab;
  const emb = vocab * d;
  const head = spec.tied_embeddings ? 0.0 : vocab * d;
  const extra = num(spec.extra_embedding_params);
  const norms = spec.norms_per_layer || 2;
  let mixers = 0.0;
  let ft = 0.0;
  let fa = 0.0;
  const layers = expandLayout(spec);
  let normW = 0.0;
  for (const run of layers) {
    const [m, t, a] = layerParams(spec, run);
    mixers = mixers + m;
    normW = normW + norms * d;
    ft = ft + t;
    fa = fa + a;
  }
  let cross = 0.0;
  if (spec.kind === "encoder-decoder") {
    for (const run of expandLayout(spec, "encoder_layout")) {
      const [m, t, a] = layerParams(spec, run);
      mixers = mixers + m;
      normW = normW + norms * d;
      ft = ft + t;
      fa = fa + a;
    }
    const cm = spec.mixers[spec.cross_mixer ?? "full"]!;
    for (let i = 0; i < layers.length; i++) {
      cross = cross + mixerParams(cm, d);
      normW = normW + d;
    }
  }
  let mtp = 0.0;
  const nMtp = num(spec.mtp_layers);
  if (nMtp > 0) {
    const run = spec.mtp_layer ?? spec.layout[spec.layout.length - 1]!;
    const [m, t] = layerParams(spec, run);
    mtp = nMtp * (m + t + 2.0 * d * d + 4.0 * d);
  }
  const total = emb + head + extra + mixers + normW + ft + cross;
  const active = emb + head + extra + mixers + normW + fa + cross;
  let matmulActive = head + mixers + fa + cross;
  if (spec.tied_embeddings) matmulActive = matmulActive + emb;
  const neTotal = mixers + normW + ft + cross;
  const neActive = mixers + normW + fa + cross;
  return {
    embedding: emb,
    norms: normW,
    lm_head: head,
    extra_embedding: extra,
    mixers,
    ffn_total: ft,
    ffn_active: fa,
    cross_attention: cross,
    mtp,
    total,
    active,
    matmul_active: matmulActive,
    non_embedding_total: neTotal,
    non_embedding_active: neActive,
  };
}

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

export function kvElemsPerToken(spec: Spec, run: LayerRun): number {
  if (run.kv_shared) return 0.0;
  if (run.mixer === "none") return 0.0;
  const m = spec.mixers[run.mixer]!;
  const t = m.type;
  if (t === "attn") {
    const kv = m.kv_heads!;
    const hd = m.head_dim!;
    const vd = m.v_head_dim || m.head_dim!;
    let e = m.k_eq_v ? kv * hd : kv * (hd + vd);
    const ix = m.indexer;
    if (ix && (run.indexer ?? true)) e = e + ix.head_dim;
    return e;
  }
  if (t === "mla") {
    let e = m.kv_lora_rank! + m.qk_rope!;
    const ix = m.indexer;
    if (ix && (run.indexer ?? true)) e = e + ix.head_dim;
    return e;
  }
  if (t === "csa") {
    const ratio = num(run.ratio);
    if (ratio <= 0 || run.kv_source === false) return 0.0;
    let e = m.head_dim! / ratio;
    const ix = m.indexer;
    if (ix && run.indexer) e = e + ix.head_dim / ratio;
    return e;
  }
  return 0.0;
}

export function windowOf(spec: Spec, run: LayerRun): number {
  if (run.mixer === "none") return 0.0;
  const m = spec.mixers[run.mixer]!;
  if (m.type === "csa") return 0.0;
  return num(m.window);
}

export function stateElems(spec: Spec, run: LayerRun): number {
  if (run.mixer === "none") return 0.0;
  const m = spec.mixers[run.mixer]!;
  const t = m.type;
  if (t === "deltanet" || t === "kda") {
    const vh = m.v_heads!;
    const kd = m.k_head_dim!;
    const vdd = m.v_head_dim!;
    const ck = num(m.conv_kernel);
    const conv = ck > 0 ? (2.0 * m.k_heads! * kd + vh * vdd) * (ck - 1.0) : 0.0;
    return vh * kd * vdd + conv;
  }
  if (t === "linear") {
    const h = m.heads!;
    const hd = m.head_dim!;
    return h * hd * hd;
  }
  if (t === "mamba2") {
    const nh = m.heads!;
    const hd = m.head_dim!;
    const n = m.state!;
    const g = m.groups!;
    const ck = m.conv_kernel!;
    return nh * hd * n + (nh * hd + 2.0 * g * n) * (ck - 1.0);
  }
  if (t === "mamba1") {
    const di = m.d_inner!;
    return di * m.state! + di * (m.conv_kernel! - 1.0);
  }
  if (t === "conv") return spec.d_model * (m.kernel! - 1.0);
  if (t === "rwkv") {
    const d = spec.d_model;
    return d * m.head_dim! + 2.0 * d;
  }
  if (t === "mlstm") {
    const h = m.heads!;
    const qk = m.qk_dim! / h;
    const vd = m.v_dim! / h;
    return h * (qk * vd + qk + 1.0);
  }
  if (t === "csa") return num(m.window) * m.head_dim!;
  return 0.0;
}

export type KvCache = {
  kv_bytes: number;
  state_bytes: number;
  total_bytes: number;
  bytes_per_token_unbounded: number;
};

function crossKvElems(spec: Spec): number {
  const cm = spec.mixers[spec.cross_mixer ?? "full"]!;
  return cm.kv_heads! * (cm.head_dim! + (cm.v_head_dim || cm.head_dim!));
}

export function kvCache(
  spec: Spec,
  context: number,
  bytesPerElem: number,
  stateBytes = 2.0,
): KvCache {
  let growing = 0.0;
  let windowed = 0.0;
  let state = 0.0;
  let perTokenUnbounded = 0.0;
  const layers = expandLayout(spec);
  for (const run of layers) {
    const e = kvElemsPerToken(spec, run);
    const w = windowOf(spec, run);
    if (e > 0) {
      if (w > 0) {
        windowed = windowed + e * Math.min(context, w);
      } else {
        growing = growing + e * context;
        perTokenUnbounded = perTokenUnbounded + e;
      }
    }
    state = state + stateElems(spec, run);
  }
  if (spec.kind === "encoder-decoder") {
    const e = crossKvElems(spec);
    growing = growing + e * context * layers.length;
    perTokenUnbounded = perTokenUnbounded + e * layers.length;
  }
  const total = (growing + windowed) * bytesPerElem + state * stateBytes;
  return {
    kv_bytes: (growing + windowed) * bytesPerElem,
    state_bytes: state * stateBytes,
    total_bytes: total,
    bytes_per_token_unbounded: perTokenUnbounded * bytesPerElem,
  };
}

// ---------------------------------------------------------------------------
// FLOPs
// ---------------------------------------------------------------------------

export function attended(spec: Spec, run: LayerRun, t: number): number {
  if (run.mixer === "none") return 0.0;
  const m = spec.mixers[run.mixer]!;
  if (m.type === "attn") {
    const ix = m.indexer;
    if (ix) return Math.min(t, ix.topk);
    const w = num(m.window);
    return w > 0 ? Math.min(t, w) : t;
  }
  if (m.type === "mla") {
    const ix = m.indexer;
    if (ix) return Math.min(t, ix.topk);
    return t;
  }
  if (m.type === "csa") {
    const ratio = num(run.ratio);
    const w = num(m.window);
    let e = Math.min(t, w);
    if (ratio > 0) {
      let c = t / ratio;
      const ix = m.indexer;
      if (ix && run.indexer) c = Math.min(c, ix.topk);
      e = e + c;
    }
    return e;
  }
  return 0.0;
}

export function attnFlopsPerEntry(spec: Spec, run: LayerRun): number {
  if (run.mixer === "none") return 0.0;
  const m = spec.mixers[run.mixer]!;
  if (m.type === "attn") {
    const h = m.heads!;
    return 2.0 * h * (m.head_dim! + (m.v_head_dim || m.head_dim!));
  }
  if (m.type === "mla") {
    const h = m.heads!;
    return 2.0 * h * (m.qk_nope! + m.qk_rope! + m.v_head_dim!);
  }
  if (m.type === "csa") {
    const h = m.heads!;
    return 2.0 * h * (2.0 * m.head_dim!);
  }
  return 0.0;
}

export function indexerFlopsPerEntry(spec: Spec, run: LayerRun): number {
  if (run.mixer === "none") return 0.0;
  const m = spec.mixers[run.mixer]!;
  const ix = m.indexer;
  if (!ix) return 0.0;
  if (m.type === "csa" && !run.indexer) return 0.0;
  return 2.0 * ix.heads * ix.head_dim;
}

export function recurrentFlops(spec: Spec, run: LayerRun): number {
  if (run.mixer === "none") return 0.0;
  const m = spec.mixers[run.mixer]!;
  const t = m.type;
  if (t === "deltanet" || t === "kda")
    return 6.0 * m.v_heads! * m.k_head_dim! * m.v_head_dim!;
  if (t === "linear") {
    const h = m.heads!;
    const hd = m.head_dim!;
    return 4.0 * h * hd * hd;
  }
  if (t === "mamba2") return 6.0 * m.heads! * m.head_dim! * m.state!;
  if (t === "mamba1") return 6.0 * m.d_inner! * m.state!;
  if (t === "rwkv") return 4.0 * spec.d_model * m.head_dim!;
  if (t === "mlstm") {
    const h = m.heads!;
    return 4.0 * h * (m.qk_dim! / h) * (m.v_dim! / h);
  }
  return 0.0;
}

function sumAttended(spec: Spec, run: LayerRun, n: number): number {
  if (run.mixer === "none") return 0.0;
  const m = spec.mixers[run.mixer]!;
  if (m.type === "attn") {
    const ix = m.indexer;
    if (ix) return sumMin(n, ix.topk);
    return sumMin(n, num(m.window));
  }
  if (m.type === "mla") {
    const ix = m.indexer;
    if (ix) return sumMin(n, ix.topk);
    return (n * (n + 1.0)) / 2.0;
  }
  if (m.type === "csa") {
    const w = num(m.window);
    let s = sumMin(n, w);
    const ratio = num(run.ratio);
    if (ratio > 0) {
      const ix = m.indexer;
      if (ix && run.indexer) {
        s = s + sumMin(n, ix.topk * ratio) / ratio;
      } else {
        s = s + (n * (n + 1.0)) / 2.0 / ratio;
      }
    }
    return s;
  }
  return 0.0;
}

export function layerMatmul(spec: Spec, run: LayerRun): number {
  const [mix, , fa] = layerParams(spec, run);
  return mix + fa;
}

export function decodeFlops(spec: Spec, context: number): number {
  const p = params(spec);
  const f = 2.0 * p.matmul_active;
  const loops = spec.loops || 1;
  let attn = 0.0;
  const runs = expandLayout(spec);
  for (const run of runs) {
    attn = attn + attended(spec, run, context) * attnFlopsPerEntry(spec, run);
    attn = attn + context * indexerFlopsPerEntry(spec, run);
    attn = attn + recurrentFlops(spec, run);
  }
  if (spec.kind === "encoder-decoder") {
    const cm = spec.mixers[spec.cross_mixer ?? "full"]!;
    const per =
      2.0 * cm.heads! * (cm.head_dim! + (cm.v_head_dim || cm.head_dim!));
    attn = attn + per * context * runs.length;
  }
  if (loops > 1) {
    const stack =
      2.0 *
      (p.matmul_active -
        p.lm_head -
        (spec.tied_embeddings ? p.embedding : 0.0));
    return f + (loops - 1.0) * stack + loops * attn;
  }
  return f + attn;
}

function layersFlopsPrefill(spec: Spec, runs: LayerRun[], n: number): number {
  let total = 0.0;
  for (const run of runs) {
    total = total + 2.0 * layerMatmul(spec, run) * n;
    total = total + sumAttended(spec, run, n) * attnFlopsPerEntry(spec, run);
    total = total + ((n * (n + 1.0)) / 2.0) * indexerFlopsPerEntry(spec, run);
    total = total + recurrentFlops(spec, run) * n;
  }
  return total;
}

export function prefillFlops(spec: Spec, n: number): number {
  const p = params(spec);
  const runs = expandLayout(spec);
  const head = 2.0 * (!spec.tied_embeddings ? p.lm_head : p.embedding);
  const loops = spec.loops || 1;
  if (spec.kind === "ced") {
    const k = spec.ced_encoder_layers!;
    const enc = runs.slice(0, k);
    const dec = runs.slice(k);
    const w = Math.min(n, num(spec.ced_window));
    let total = layersFlopsPrefill(spec, enc, n);
    const d = spec.d_model;
    let kvProj = 0.0;
    for (const run of dec) kvProj = kvProj + kvElemsPerToken(spec, run) * d;
    total = total + 2.0 * kvProj * n;
    total = total + layersFlopsPrefill(spec, dec, w);
    return total + head;
  }
  let total = layersFlopsPrefill(spec, runs, n);
  if (spec.kind === "encoder-decoder") {
    total =
      total + layersFlopsPrefill(spec, expandLayout(spec, "encoder_layout"), n);
  }
  return loops * total + head;
}

export function attendedStored(spec: Spec, run: LayerRun, t: number): number {
  const m = spec.mixers[run.mixer]!;
  if (m.type === "csa") {
    const ratio = num(run.ratio);
    return ratio > 0 ? t / ratio : 0.0;
  }
  const w = num(m.window);
  return w > 0 ? Math.min(t, w) : t;
}

export type DecodeBytes = {
  weights: number;
  kv: number;
  state: number;
  total: number;
};

export function decodeBytes(
  spec: Spec,
  context: number,
  weightBytes: number,
  kvBytes: number,
  stateBytes = 2.0,
): DecodeBytes {
  const p = params(spec);
  let w = (p.active - p.embedding - p.extra_embedding) * weightBytes;
  if (spec.tied_embeddings) w = w + p.embedding * weightBytes;
  let kv = 0.0;
  let state = 0.0;
  const runs = expandLayout(spec);
  for (const run of runs) {
    const e = kvElemsPerToken(spec, run);
    if (e > 0) kv = kv + attendedStored(spec, run, context) * e;
    state = state + stateElems(spec, run);
  }
  if (spec.kind === "encoder-decoder") {
    kv = kv + crossKvElems(spec) * context * runs.length;
  }
  const total = w + kv * kvBytes + state * stateBytes * 2.0;
  return {
    weights: w,
    kv: kv * kvBytes,
    state: state * stateBytes * 2.0,
    total,
  };
}
