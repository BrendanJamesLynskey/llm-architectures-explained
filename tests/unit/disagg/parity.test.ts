/**
 * Engine parity for the CED option: the vendored engine against the Python
 * package at the same commit, on the simulator's own ten JS-parity
 * configurations (tests/test_ced.py), with fixtures from
 * scripts/disagg_reference.py. Every request's seven timestamps
 * (prefill_done included), every instance's energy counters and the link's
 * must be identical. The exception the simulator's own test makes: runs
 * whose steps hit the power roof solve a cubic with cube roots, which can
 * differ by an ulp between libms, so those runs compare to 1e-9 relative.
 */
import { describe, expect, it } from "vitest";

import { engine, type Row, type SimConfig } from "@/lib/disagg/engine";

import { readJson } from "./helpers";

type Case = {
  name: string;
  cfg: SimConfig;
  rows: Row[];
  stamps: (number | null)[][];
  inst: number[][];
  link: number[];
  total: number;
  slo: number;
  capped: boolean;
};
const fixture = readJson<{ cases: Case[] }>(
  "tests/unit/fixtures/disagg_ced_parity.json",
);

function close(a: number | null, b: number | null, tol: number): boolean {
  if (a === null || b === null) return a === b;
  return tol === 0
    ? a === b
    : Math.abs(a - b) <= tol * Math.max(1, Math.abs(a));
}

describe("vendored engine = Python package on the CED option", () => {
  it("covers ten configurations, at least eight off the power roof", () => {
    expect(fixture.cases).toHaveLength(10);
    expect(
      fixture.cases.filter((c) => !c.capped).length,
    ).toBeGreaterThanOrEqual(8);
  });

  for (const c of fixture.cases) {
    it(c.name, () => {
      const tol = c.capped ? 1e-9 : 0;
      const r = engine.simulate(c.cfg, c.rows);
      const s = engine.summarise(r);
      let bad = 0;
      r.reqs.forEach((q, i) => {
        const js = [
          q.prefillStart,
          q.firstToken,
          q.prefillDone,
          q.kvStart,
          q.kvReady,
          q.decodeStart,
          q.finish,
        ];
        js.forEach((x, k) => {
          if (!close(c.stamps[i]![k]!, x, tol)) bad++;
        });
      });
      expect(bad).toBe(0);
      expect(r.insts).toHaveLength(c.inst.length);
      r.insts.forEach((ins, i) => {
        const js = [
          ins.ec,
          ins.em,
          ins.busy,
          ins.peakW,
          ins.steps,
          ins.batchSum,
        ];
        js.forEach((x, k) => expect(close(c.inst[i]![k]!, x, tol)).toBe(true));
      });
      const l = [r.link.energy, r.link.bytes, r.link.wait, r.link.transitJ];
      l.forEach((x, k) => expect(close(c.link[k]!, x, tol)).toBe(true));
      expect(close(c.total, s.energy.totalJ, tol)).toBe(true);
      expect(s.sloAttain).toBe(c.slo);
    });
  }
});
