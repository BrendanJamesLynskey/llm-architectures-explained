/**
 * Every number the chapters quote, recomputed with the tested TypeScript
 * (the cost model, src/lib/chapters/model.ts and the vendored simulator's
 * fixtures) and checked to appear, formatted the same way, in the MDX. If a
 * model or the data changes, this fails until the prose is updated too.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import {
  decodeBytes,
  decodeFlops,
  kvCache,
  params,
  prefillFlops,
  type Spec,
} from "@/lib/arch/costModel";
import { specOf } from "@/lib/arch/features";
import { formatBytes, formatCount, formatFlops } from "@/lib/arch/format";
import { CHAPTERS } from "@/lib/chapters/data";
import {
  asDecoder,
  depthWidthSpec,
  loopedKvBytes,
  loopedSpec,
  moeSpec,
  mtpExpectedTokens,
  mtpModuleActive,
  mtpSpeedup,
  normProfile,
  ropeLongPairs,
} from "@/lib/chapters/model";
import { MODELS } from "@/lib/data";

import { results } from "../disagg/helpers";

const mdx = (slug: string) =>
  readFileSync(
    join(process.cwd(), "content/chapters", `${slug}.mdx`),
    "utf-8",
  ).replace(/\s+/g, " ");
const spec = (id: string): Spec => specOf(MODELS.find((m) => m.id === id)!)!;
const v = (id: string): Spec =>
  CHAPTERS.attention.variants.find((x) => x.id === id)!.spec;
const K128 = 131072;
const M1 = 1048576;

function quotes(slug: string, list: string[]): void {
  const text = mdx(slug);
  it(`${slug} quotes ${list.length} computed values`, () => {
    for (const q of list) expect(text, q).toContain(q);
  });
}

describe("chapter 1: attention", () => {
  const kv = (id: string) =>
    formatBytes(kvCache(v(id), K128, 2, 2).total_bytes);
  const tok = (id: string) =>
    formatBytes(kvCache(v(id), K128, 2, 2).bytes_per_token_unbounded);
  const fl = (id: string) =>
    formatFlops(decodeFlops(v(id), K128)).replace(" GFLOP", "");
  quotes("01-attention", [
    `cache is ${kv("gqa")} per sequence`,
    `MHA's is 4× that, ${kv("mha")}`,
    `MQA's ${kv("mqa")}`,
    `MLA's ${kv("mla")}`,
    `holds ${kv("swa")}`,
    `Mamba-2 hybrid ${kv("mamba")}`,
    `needs ${fl("swa")} GFLOP per token against GQA's ${fl("gqa")}`,
    `MLA needs more (${fl("mla")} GFLOP)`,
    `${tok("mha")} per token for MHA, ${tok("gqa")} for GQA, ${tok("mqa")} for MQA`,
    `${tok("mla")} for MLA`,
  ]);
  it("MHA's cache is exactly 4× GQA's", () =>
    expect(kvCache(v("mha"), K128, 2, 2).total_bytes).toBe(
      4 * kvCache(v("gqa"), K128, 2, 2).total_bytes,
    ));
});

describe("chapter 2: positional encoding", () => {
  const r = (id: string) => CHAPTERS.rope.find((x) => x.id === id)!;
  const n = (id: string, c: number) =>
    ropeLongPairs(r(id).head_dim, r(id).fraction, r(id).theta, c);
  quotes("02-positional-encoding", [
    `leaves ${n("mistral-7b", 32768)} of its 64 pairs slower than a 32K context`,
    `θ = 1,000,000 leaves ${n("qwen3-8b", 32768)}`,
  ]);
  it("DeepSeek-V3: no pair slower than 128K", () =>
    expect(n("deepseek-v3", K128)).toBe(0));
  it("GLM-4.5 rotates half, Qwen3-Next a quarter", () => {
    expect(r("glm-4.5").fraction).toBe(0.5);
    expect(r("qwen3-next-80b-a3b").fraction).toBe(0.25);
    expect(r("mistral-7b").theta).toBe(10000);
    expect(r("qwen3-8b").theta).toBe(1000000);
  });
});

describe("chapter 3: normalisation", () => {
  const pre1 = normProfile("pre-norm", 32, 1);
  const pre2 = normProfile("pre-norm", 32, 2);
  const out2 = normProfile("output-norm", 32, 2);
  const post = normProfile("post-ln", 32, 1);
  quotes("03-normalisation", [
    `stream ends ${pre1.rms.at(-1)!.toFixed(2)} times`,
    `changes it by ${(100 * pre1.update.at(-1)!).toFixed(1)}%`,
    `still ${(100 * pre1.embedding_share).toFixed(1)}% of it`,
    `doubles to ${pre2.rms.at(-1)!.toFixed(2)}`,
  ]);
  it("post-LN's embedding share is 2^-32; output norm ignores the gain", () => {
    // 64 divisions by √2 (rounded each time): 2^-32 to 1e-12
    expect(Math.abs(post.embedding_share / 2 ** -32 - 1)).toBeLessThan(1e-12);
    expect(out2.rms.at(-1)).toBe(pre1.rms.at(-1));
  });
});

describe("chapter 4: dense and MoE", () => {
  const base = CHAPTERS.moe.spec;
  const p = (...a: [number, number, number, number, number]) =>
    params(moeSpec(base, ...a));
  const mix = p(8, 2, 0, 1, 0);
  const dsm = p(64, 6, 2, 8, 1);
  const v3 = p(256, 8, 1, 16, 3);
  quotes("04-dense-and-moe", [
    `(${formatCount(params(base).total)} parameters dense)`,
    `give ${formatCount(mix.total)} total and ${formatCount(mix.active)} active`,
    `give ${formatCount(dsm.total)} total but only ${formatCount(dsm.active)} active`,
    `give ${formatCount(v3.total)} total and ${formatCount(v3.active)} active`,
    `${((100 * v3.active) / v3.total).toFixed(1)}% of the total`,
    `(${formatCount(v3.total)} parameters in BF16 is ${formatBytes(2 * v3.total)})`,
  ]);
  it("the fine-grained preset holds about six times the dense weights", () =>
    expect(Math.round(dsm.total / params(base).total)).toBe(6));
});

describe("chapter 5: depth and width", () => {
  const row = (l: number, d: number) => {
    const s = depthWidthSpec(l, d);
    const q = params(s);
    const f = decodeFlops(s, 32768);
    return {
      total: formatCount(q.total),
      tok: formatBytes(kvCache(s, 32768, 2, 2).bytes_per_token_unbounded),
      emb: ((100 * (q.embedding + q.lm_head)) / q.total).toFixed(1),
      attn: ((100 * (f - 2 * q.matmul_active)) / f).toFixed(1),
    };
  };
  const deep = row(80, 2048);
  const mid = row(32, 4096);
  const wide = row(12, 8192);
  quotes("05-depth-and-width", [
    `caches ${deep.tok} per token`,
    `12-layer, 8,192-wide one ${wide.tok}`,
    `(${wide.total} against ${deep.total})`,
    `spends ${deep.attn}% of its decode FLOPs`,
    `Llama 3 8B shape ${mid.attn}%`,
    `shallow, wide model ${wide.attn}%`,
    `${deep.emb}% of the deep, thin model's parameters and ${wide.emb}%`,
  ]);
  it("Llama 3.1 405B and Gemma 3 270M shapes", () => {
    const a = spec("llama-3.1-405b");
    const b = spec("gemma-3-270m");
    expect([a.d_model, a.layout.reduce((n, r) => n + r.n, 0)]).toEqual([
      16384, 126,
    ]);
    expect([b.d_model, b.layout.reduce((n, r) => n + r.n, 0)]).toEqual([
      640, 18,
    ]);
  });
});

describe("chapter 6: long context", () => {
  const kv = (id: string, c: number) =>
    formatBytes(kvCache(spec(id), c, 2, 2).total_bytes);
  const fl = (id: string) => formatFlops(decodeFlops(spec(id), M1));
  const share = (id: string) =>
    (
      100 *
      (1 - (2 * params(spec(id)).matmul_active) / decodeFlops(spec(id), M1))
    ).toFixed(1);
  quotes("06-long-context", [
    `holds ${kv("llama-3.1-405b", K128)} of cache per 128K-token sequence`,
    `need ${kv("llama-3.1-405b", M1)} for one sequence, and ${fl("llama-3.1-405b")} per generated token, ${share("llama-3.1-405b")}% of it attention`,
    `1M-token cache to ${kv("gemma-3-27b", M1)}`,
    `latent holds ${kv("deepseek-v3", M1)} at 1M`,
    `${fl("deepseek-v3")} per token, ${share("deepseek-v3")}% attention`,
    `decode drops to ${fl("deepseek-v3.2")}`,
    `needs ${kv("qwen3-next-80b-a3b", M1)} at 1M`,
    `${kv("minimax-text-01", M1)}.`,
    `${kv("deepseek-v4-flash", M1)} at 1M tokens, the smallest here`,
  ]);
  it("the smallest share of token mixing at 1M is still most of the step", () => {
    const min = Math.min(...CHAPTERS.context.map((id) => Number(share(id))));
    expect(min).toBeGreaterThan(50);
  });
});

describe("chapter 7: multi-token prediction", () => {
  const s = spec("deepseek-v3");
  quotes("07-multi-token-prediction", [
    `touches ${formatCount(mtpModuleActive(s))} weights per token`,
    `the ${formatCount(params(s).active)} the main model activates`,
    `(${mtpExpectedTokens(0.85, 1).toFixed(2)} at $a = 0.85$)`,
    `speed-up is ${mtpSpeedup(s, 8192, 1, 0.85, 2, 2).speedup.toFixed(2)}× at $a = 0.85$ and ${mtpSpeedup(s, 8192, 1, 0.9, 2, 2).speedup.toFixed(2)}× at $a = 0.9$`,
  ]);
  it("one module is under 2% of the active weights; Step 3.5 Flash ships three", () => {
    expect(mtpModuleActive(s) / params(s).active).toBeLessThan(0.02);
    expect(spec("step-3.5-flash").mtp_layers).toBe(3);
  });
});

describe("chapter 8: looped and parallel blocks", () => {
  const one = loopedSpec(CHAPTERS.loops.spec, 1);
  const four = loopedSpec(CHAPTERS.loops.spec, 4);
  quotes("08-looped-and-parallel-blocks", [
    `${formatCount(params(four).total)} parameters stay put`,
    `go from ${formatFlops(decodeFlops(one, 8192)).replace(" GFLOP", "")} to ${formatFlops(decodeFlops(four, 8192))}`,
    `${formatBytes(loopedKvBytes(four, 8192, 2, true))} instead of ${formatBytes(loopedKvBytes(four, 8192, 2, false))}`,
  ]);
  it("looping leaves the parameters unchanged", () =>
    expect(params(four).total).toBe(params(one).total));
});

describe("chapter 9: encoder-decoder and CED", () => {
  const ced = CHAPTERS.ced.spec;
  const r = (n: number) =>
    (prefillFlops(ced, n) / prefillFlops(asDecoder(ced), n)).toFixed(3);
  const s16 = results.s16;
  const w = (k: string) => results.s17.find((x) => x.workload === k)!;
  const cell = (k: string, v: string, n: string) => w(k).rows[v]!.cells[n]!;
  const gain = (k: string, v: string) => {
    const best = (name: string) =>
      Math.max(...Object.values(w(k).rows[name]!.cells).map((c) => c.rate));
    return `${(best(v) / best("decoder-only")).toFixed(2)}×`;
  };
  quotes("09-encoder-decoder-and-ced", [
    `CED's prefill is ${r(8192)} of the same stack`,
    `at 1M tokens ${r(M1)}`,
    `touches ${s16.params.ced_prompt_token!.text} parameters against ${s16.params.decoder_only_token!.text}`,
    `takes ${s16.steps["8192"]!.cells.ced.ms} with the replay`,
    `and ${s16.steps["8192"]!.cells.enc.ms} encoder-only, against ${s16.steps["8192"]!.cells.base.ms}`,
    `room for ${s16.room["2"]!.enc.kv_tokens!.toLocaleString("en-GB")} tokens of cache, against ${s16.room["2"]!.base.kv_tokens!.toLocaleString("en-GB")}`,
    `4P2D at ${cell("2048:512", "decoder-only", "4").text} req/s`,
    `at ${cell("2048:512", "CED, replay on prefill", "3").text} req/s with the replay on prefill and ${cell("2048:512", "CED, replay on decode", "3").text} on decode: ${gain("2048:512", "CED, replay on prefill")} and ${gain("2048:512", "CED, replay on decode")}`,
    `At 4,096 : 256 the gains are ${gain("4096:256", "CED, replay on prefill")} and ${gain("4096:256", "CED, replay on decode")}`,
    `0.67B of decoder K/V projections`,
  ]);
  it("quoted gains are the ones results.md records", () => {
    expect(w("2048:512").rows["CED, replay on prefill"]!.line).toContain(
      "| 1.52x |",
    );
    expect(w("2048:512").rows["CED, replay on decode"]!.line).toContain(
      "| 1.60x |",
    );
    expect(w("4096:256").rows["CED, replay on prefill"]!.line).toContain(
      "| 2.06x |",
    );
    expect(w("4096:256").rows["CED, replay on decode"]!.line).toContain(
      "| 2.13x |",
    );
    expect(s16.params.ced_kv_proj!.text).toBe("0.67B");
    expect(results.quotes.s17_intro).toContain(
      "Gains above 2x are queueing, not FLOPs",
    );
  });
  it("best splits: decoder-only 4P2D, CED 3P3D at 2048:512", () => {
    expect(w("2048:512").rows["decoder-only"]!.best).toBe(4);
    expect(w("2048:512").rows["CED, replay on prefill"]!.best).toBe(3);
    expect(w("2048:512").rows["CED, replay on decode"]!.best).toBe(3);
  });
});
