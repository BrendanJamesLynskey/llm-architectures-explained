/**
 * The TypeScript cost model against fixtures written by the Python
 * reference (scripts/make_fixtures.py): every number must be identical.
 * No tolerance is needed, because both sides evaluate the same expressions
 * in the same order on IEEE doubles.
 */
import { readFileSync } from "node:fs";
import path from "node:path";

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
import { MODELS } from "@/lib/data";

type Rec = Record<string, number>;
type Case = {
  id: string;
  spec: Spec;
  out: {
    params: Rec;
    kv: {
      context: number;
      kv_elem_bytes: number;
      state_elem_bytes: number;
      out: Rec;
    }[];
    decode_flops: { context: number; v: number }[];
    prefill_flops: { context: number; v: number }[];
    decode_bytes: {
      context: number;
      weight_elem_bytes: number;
      kv_elem_bytes: number;
      out: Rec;
    }[];
  };
};

const FIX = JSON.parse(
  readFileSync(path.join(__dirname, "../fixtures/arch_fixtures.json"), "utf8"),
) as { cases: Case[] };

describe("cost model parity with the Python reference", () => {
  it("covers every model that has an architecture", () => {
    const withArch = MODELS.filter((m) => m.arch !== null).map((m) => m.id);
    expect(FIX.cases.map((c) => c.id).sort()).toEqual(withArch.sort());
  });

  it("the site's spec for each model is the fixture's spec", () => {
    for (const c of FIX.cases) {
      const m = MODELS.find((x) => x.id === c.id)!;
      expect(specOf(m)).toEqual(c.spec);
    }
  });

  let compared = 0;
  for (const c of FIX.cases) {
    it(`${c.id}: every number matches exactly`, () => {
      const p = params(c.spec) as unknown as Rec;
      for (const [k, v] of Object.entries(c.out.params)) {
        expect(p[k], `params.${k}`).toBe(v);
        compared++;
      }
      for (const row of c.out.kv) {
        const got = kvCache(
          c.spec,
          row.context,
          row.kv_elem_bytes,
          row.state_elem_bytes,
        ) as unknown as Rec;
        for (const [k, v] of Object.entries(row.out)) {
          expect(got[k], `kv.${k} @ ${row.context}`).toBe(v);
          compared++;
        }
      }
      for (const row of c.out.decode_flops) {
        expect(
          decodeFlops(c.spec, row.context),
          `decode @ ${row.context}`,
        ).toBe(row.v);
        compared++;
      }
      for (const row of c.out.prefill_flops) {
        expect(
          prefillFlops(c.spec, row.context),
          `prefill @ ${row.context}`,
        ).toBe(row.v);
        compared++;
      }
      for (const row of c.out.decode_bytes) {
        const got = decodeBytes(
          c.spec,
          row.context,
          row.weight_elem_bytes,
          row.kv_elem_bytes,
        ) as unknown as Rec;
        for (const [k, v] of Object.entries(row.out)) {
          expect(got[k], `bytes.${k} @ ${row.context}`).toBe(v);
          compared++;
        }
      }
    });
  }

  it("compared a meaningful number of values", () => {
    expect(compared).toBeGreaterThan(30_000);
  });
});
