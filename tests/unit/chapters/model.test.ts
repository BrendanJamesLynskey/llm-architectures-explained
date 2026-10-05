/**
 * The chapter interactives' TypeScript (src/lib/chapters/model.ts and the
 * cost model it drives) against the Python reference
 * (reference/chapter_model.py), via the fixtures of
 * scripts/make_chapter_fixtures.py: exact equality everywhere except the
 * RoPE wavelengths (a pow; 1e-12 relative).
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
import chapters from "@/data/chapters.json";
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
  PLACEMENTS,
  ropeLongPairs,
  ropeWavelengths,
  rotaryDims,
  type Placement,
} from "@/lib/chapters/model";

type Costs = {
  params: Record<string, number>;
  kv: { context: number; out: Record<string, number> }[];
  decode_flops: { context: number; v: number }[];
  prefill_flops: { context: number; v: number }[];
  decode_bytes: { context: number; out: Record<string, number> }[];
};
const fx = JSON.parse(
  readFileSync(
    join(process.cwd(), "tests/fixtures/chapter_fixtures.json"),
    "utf-8",
  ),
) as Record<string, unknown> & {
  attention: { id: string; out: Costs }[];
  moe: {
    args: number[];
    spec: Spec;
    params: Record<string, number>;
    decode_bytes: Record<string, number>;
  }[];
  depth_width: { args: number[]; spec: Spec; out: Costs }[];
  rope: {
    id: string;
    wavelengths: number[];
    long_pairs: { context: number; v: number }[];
  }[];
  rotary_dims: { args: [number, number]; v: number }[];
  norms: { args: [Placement, number, number]; out: Record<string, unknown> }[];
  norms_gamma: Record<string, unknown>;
  mtp: {
    id: string;
    module_active: number;
    mtp_params: number;
    speedup: {
      depth: number;
      acceptance: number;
      out: Record<string, number>;
    }[];
  }[];
  mtp_tokens: { args: [number, number]; v: number }[];
  loops: {
    loops: number;
    spec: Spec;
    out: Costs;
    kv_per_loop: number[];
    kv_shared: number[];
  }[];
  ced: { as_decoder: Spec; ced: Costs; decoder: Costs; t5: Costs };
};

const BF16 = 2.0;

function expectCosts(spec: Spec, want: Costs): void {
  expect({ ...params(spec) }).toEqual(want.params);
  for (const k of want.kv)
    expect(kvCache(spec, k.context, BF16, BF16)).toEqual(k.out);
  for (const f of want.decode_flops)
    expect(decodeFlops(spec, f.context)).toBe(f.v);
  for (const f of want.prefill_flops)
    expect(prefillFlops(spec, f.context)).toBe(f.v);
  for (const b of want.decode_bytes)
    expect(decodeBytes(spec, b.context, BF16, BF16)).toEqual(b.out);
}

const specs = chapters as unknown as {
  attention: { variants: { id: string; spec: Spec }[] };
  moe: { spec: Spec };
  rope: { id: string; theta: number; head_dim: number; fraction: number }[];
  mtp: string[];
  loops: { spec: Spec };
  ced: { spec: Spec; t5_spec: Spec };
};

describe("attention variants (chapter 1)", () => {
  it("are the eight mixers, in order", () => {
    expect(specs.attention.variants.map((v) => v.id)).toEqual(
      fx.attention.map((v) => v.id),
    );
    expect(fx.attention).toHaveLength(8);
  });
  for (const v of specs.attention.variants) {
    it(v.id, () =>
      expectCosts(v.spec, fx.attention.find((a) => a.id === v.id)!.out),
    );
  }
});

describe("moeSpec (chapter 4)", () => {
  for (const c of fx.moe) {
    it(c.args.join(","), () => {
      const [e, k, s, g, p] = c.args as [
        number,
        number,
        number,
        number,
        number,
      ];
      const spec = moeSpec(specs.moe.spec, e, k, s, g, p);
      expect(spec).toEqual(c.spec);
      expect({ ...params(spec) }).toEqual(c.params);
      expect(decodeBytes(spec, 8192, BF16, BF16)).toEqual(c.decode_bytes);
    });
  }
});

describe("depthWidthSpec (chapter 5)", () => {
  for (const c of fx.depth_width) {
    it(c.args.join("x"), () => {
      const spec = depthWidthSpec(c.args[0]!, c.args[1]!);
      expect(spec).toEqual(c.spec);
      expectCosts(spec, c.out);
    });
  }
});

describe("RoPE (chapter 2)", () => {
  for (const c of fx.rotary_dims)
    it(`rotaryDims(${c.args.join(", ")})`, () =>
      expect(rotaryDims(...c.args)).toBe(c.v));
  for (const r of specs.rope) {
    it(r.id, () => {
      const want = fx.rope.find((x) => x.id === r.id)!;
      const got = ropeWavelengths(r.head_dim, r.fraction, r.theta);
      expect(got).toHaveLength(want.wavelengths.length);
      got.forEach((w, i) =>
        expect(Math.abs(w - want.wavelengths[i]!)).toBeLessThanOrEqual(
          1e-12 * want.wavelengths[i]!,
        ),
      );
      for (const lp of want.long_pairs)
        expect(ropeLongPairs(r.head_dim, r.fraction, r.theta, lp.context)).toBe(
          lp.v,
        );
    });
  }
});

describe("norm placement profiles (chapter 3)", () => {
  it("covers every placement", () => expect([...PLACEMENTS]).toHaveLength(3));
  for (const c of fx.norms)
    it(c.args.join(" "), () => expect(normProfile(...c.args)).toEqual(c.out));
  it("output norm uses its scale, not the gain", () =>
    expect(normProfile("output-norm", 32, 3.0, 0.5)).toEqual(fx.norms_gamma));
});

describe("multi-token prediction (chapter 7)", () => {
  for (const c of fx.mtp_tokens)
    it(`E[tokens](${c.args.join(", ")})`, () =>
      expect(mtpExpectedTokens(...c.args)).toBe(c.v));
  it("lists every model with MTP layers", () =>
    expect(specs.mtp).toEqual(fx.mtp.map((m) => m.id)));
  for (const m of fx.mtp) {
    it(m.id, async () => {
      const { MODELS } = await import("@/lib/data");
      const rec = MODELS.find((x) => x.id === m.id)!;
      const { specOf } = await import("@/lib/arch/features");
      const spec = specOf(rec)!;
      expect(mtpModuleActive(spec)).toBe(m.module_active);
      expect(params(spec).mtp).toBe(m.mtp_params);
      for (const s of m.speedup)
        expect(
          mtpSpeedup(spec, 8192, s.depth, s.acceptance, BF16, BF16),
        ).toEqual(s.out);
    });
  }
});

describe("looped blocks (chapter 8)", () => {
  for (const c of fx.loops) {
    it(`${c.loops} loop(s)`, () => {
      const spec = loopedSpec(specs.loops.spec, c.loops);
      expect(spec).toEqual(c.spec);
      expectCosts(spec, c.out);
      const ctx = c.out.kv.map((k) => k.context);
      ctx.forEach((x, i) => {
        expect(loopedKvBytes(spec, x, BF16, true)).toBe(c.kv_per_loop[i]);
        expect(loopedKvBytes(spec, x, BF16, false)).toBe(c.kv_shared[i]);
      });
    });
  }
});

describe("encoder-decoder and CED (chapter 9)", () => {
  it("CED and the same stack as a decoder", () => {
    const dec = asDecoder(specs.ced.spec);
    expect(dec).toEqual(fx.ced.as_decoder);
    expectCosts(specs.ced.spec, fx.ced.ced);
    expectCosts(dec, fx.ced.decoder);
    expectCosts(specs.ced.t5_spec, fx.ced.t5);
  });
});
