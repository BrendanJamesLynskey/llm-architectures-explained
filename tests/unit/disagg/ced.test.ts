/**
 * Section 16 of results.md (the CED cost model), recomputed with the
 * vendored engine's own cost model: every parameter count, prefill step
 * (FLOPs and time) and memory figure equals the recorded value exactly, and
 * formats to the text results.md shows.
 */
import { describe, expect, it } from "vitest";

import {
  CLUSTER,
  cedConfig,
  cedParams,
  f2,
  prefillRoom,
  prefillStep,
  SMALL_PREFILL,
  ttftSlo,
  VARIANTS,
  WORKLOADS,
} from "@/lib/disagg/ced";

import { results } from "./helpers";

const ms = (x: number): string =>
  x < 1
    ? `${(1e3 * x).toFixed(1)} ms`
    : `${Math.round(1e3 * x).toLocaleString("en-GB")} ms`;

describe("section 16: parameters a token touches", () => {
  const p = cedParams();
  const rec = results.s16.params;
  it("equal the recorded counts", () => {
    expect(p.decoderOnlyToken).toBe(rec.decoder_only_token!.v);
    expect(p.cedPromptToken).toBe(rec.ced_prompt_token!.v);
    expect(p.cedDecodeToken).toBe(rec.ced_decode_token!.v);
    expect(p.cedKvProj).toBe(rec.ced_kv_proj!.v);
  });
  it("format as results.md shows them", () => {
    expect(`${(p.cedPromptToken / 1e9).toFixed(2)}B`).toBe(
      rec.ced_prompt_token!.text,
    );
    expect(rec.ced_prompt_token!.line).toContain("0.502");
    expect((p.cedPromptToken / p.cedDecodeToken).toFixed(3)).toBe("0.502");
  });
});

describe("section 16: one prefill step on 4×H100", () => {
  for (const [s, row] of Object.entries(results.s16.steps)) {
    it(`${s} tokens`, () => {
      for (const k of ["base", "ced", "enc"] as const) {
        const c = prefillStep(k, Number(s));
        expect(c.flops).toBe(row.cells[k].flops);
        expect(c.time).toBe(row.cells[k].time);
        expect(c.bound).toBe(row.cells[k].bound);
        expect((c.flops / 1e15).toFixed(3)).toBe(row.cells[k].pflop);
        expect(ms(c.time)).toBe(row.cells[k].ms);
        expect(row.line).toContain(row.cells[k].ms);
      }
      const base = prefillStep("base", Number(s)).time;
      expect((prefillStep("enc", Number(s)).time / base).toFixed(3)).toBe(
        row.cells.enc.ratio,
      );
    });
  }
});

describe("section 16: what a prefill instance holds", () => {
  for (const [n, row] of Object.entries(results.s16.room)) {
    it(`${n}× H100`, () => {
      for (const k of ["base", "enc"] as const) {
        const r = prefillRoom(k, Number(n));
        expect(r.weights).toBe(row[k].weights);
        expect(r.kvTokens).toBe(row[k].kv_tokens);
      }
    });
  }
});

describe("the experiment's configurations", () => {
  it("formats rates as results.py does", () => {
    for (const w of results.s17)
      for (const r of Object.values(w.rows))
        for (const c of Object.values(r.cells)) expect(f2(c.rate)).toBe(c.text);
  });
  it("TTFT SLO = 5× the decoder-only unloaded prefill, as recorded", () => {
    for (const w of results.s17) {
      expect(ttftSlo(w.prompt)).toBe(w.ttft_slo);
      expect(ms(w.ttft_slo)).toBe(w.slo_text);
    }
    expect(ttftSlo(results.s18.prompt)).toBe(results.s18.ttft_slo);
  });
  it("cells use the configurations the fixtures were measured with", () => {
    expect(WORKLOADS.map((w) => `${w.prompt}:${w.output}`)).toEqual(
      results.s17.map((w) => w.workload),
    );
    for (const w of results.s17) {
      for (const v of VARIANTS) {
        for (let np = 1; np < CLUSTER; np++) {
          const cell = w.rows[v.key]!.cells[String(np)]!;
          expect(cedConfig(v.key, np, CLUSTER - np, w.ttft_slo)).toEqual(
            cell.cfg,
          );
        }
      }
    }
    for (const v of VARIANTS) {
      for (const [np, nd] of SMALL_PREFILL) {
        const cell = results.s18.rows[v.key]!.cells[`${np}x2+${nd}x4`]!;
        expect(cedConfig(v.key, np, nd, results.s18.ttft_slo, 2)).toEqual(
          cell.cfg,
        );
      }
    }
  });
});
