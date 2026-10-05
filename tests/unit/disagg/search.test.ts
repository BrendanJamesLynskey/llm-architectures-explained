/**
 * The capacity search, re-run in TypeScript on the vendored engine, finds
 * exactly the rates results.md sections 17 and 18 record (the full-precision
 * values of the simulator's examples/results_ced.json): the analytic
 * ceiling it starts from equals Python's, the arrivals rebuilt from the
 * recorded unit draws equal Python's at every rate, and every bisection
 * lands on the same float.
 */
import { describe, expect, it } from "vitest";

import {
  analyticCeiling,
  maxSustainableRate,
  workloadAt,
} from "@/lib/disagg/engine";

import { draws, results, type Cell } from "./helpers";

const all: { label: string; p: number; o: number; cell: Cell }[] = [];
for (const w of results.s17)
  for (const [name, row] of Object.entries(w.rows))
    for (const [k, cell] of Object.entries(row.cells))
      all.push({
        label: `§17 ${w.workload} ${name} ${k}P`,
        p: w.prompt,
        o: w.output,
        cell,
      });
for (const [name, row] of Object.entries(results.s18.rows))
  for (const [k, cell] of Object.entries(row.cells))
    all.push({
      label: `§18 ${name} ${k}`,
      p: results.s18.prompt,
      o: results.s18.output,
      cell,
    });

describe("arrivals rebuilt from unit draws", () => {
  it("are increasing and scale with 1/rate", () => {
    const d = draws(2048, 512).rows;
    const a = workloadAt(d, 2.0);
    const b = workloadAt(d, 4.0);
    expect(a).toHaveLength(1000);
    expect(a[0]![0]).toBe(d[0]![0] / 2.0);
    expect(b[999]![0]).toBeLessThan(a[999]![0]);
    expect(a.every((r, i) => i === 0 || r[0] > a[i - 1]![0])).toBe(true);
  });
});

describe("analytic ceilings = search.analytic_capacity", () => {
  it(`covers all ${all.length} cells (60 in §17, 12 in §18)`, () => {
    expect(all).toHaveLength(72);
  });
  for (const { label, p, o, cell } of all) {
    it(label, () => {
      expect(analyticCeiling(cell.cfg, p, o)).toBe(cell.ceiling);
    });
  }
});

describe("maxSustainableRate = the recorded results.md rates, exactly", () => {
  for (const { label, p, o, cell } of all) {
    it(label, () => {
      const r = maxSustainableRate(cell.cfg, draws(p, o).rows, p, o);
      expect(r.rate).toBe(cell.rate);
      expect(r.rate.toFixed(2)).toBe(cell.text);
      expect(r.bracket[1]).toBeGreaterThan(r.rate);
    });
  }
});
