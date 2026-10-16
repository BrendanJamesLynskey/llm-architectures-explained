/**
 * Brief 41's animations against reference/visuals.py
 * (tests/fixtures/visual_fixtures.json, rounded there to 12 significant
 * digits because sin, cos, exp and pow may differ in the last bit between
 * libms): the RoPE bank's every phase and score, the head-sharing bytes,
 * and every MoE step's routes, loads and biases (integers and choices
 * exactly). Captions are pinned where they quote the model.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { CHAPTERS } from "@/lib/chapters/data";
import {
  MOE,
  ROPE_GAP,
  ROPE_POSITIONS,
  ROPE_SHOWN,
  headSharing,
  moeCaption,
  moeRouting,
  ropeBank,
  ropeCaption,
  ropeOmegas,
  ropeShown,
  sharingCaption,
  type RopeMode,
} from "@/lib/chapters/visuals";

const fx = JSON.parse(
  readFileSync(
    join(process.cwd(), "tests/fixtures/visual_fixtures.json"),
    "utf-8",
  ),
) as {
  rope: {
    positions: number;
    shown: number;
    gap: number;
    banks: Record<
      string,
      { head_dim: number; theta: number; states: unknown[] }
    >;
  };
  sharing: unknown[];
  moe: { cfg: unknown; on: unknown[]; off: unknown[] };
};

function close(a: unknown, b: unknown, tol = 1e-10, p = "$"): string | null {
  if (typeof b === "number" && typeof a === "number")
    return Math.abs(a - b) <= tol * Math.max(1, Math.abs(b))
      ? null
      : `${p}: ${a} vs ${b}`;
  if (Array.isArray(b)) {
    if (!Array.isArray(a) || a.length !== b.length) return `${p}: length`;
    for (let i = 0; i < b.length; i++) {
      const m = close(a[i], b[i], tol, `${p}[${i}]`);
      if (m) return m;
    }
    return null;
  }
  if (b && typeof b === "object") {
    const ao = a as Record<string, unknown>;
    const bo = b as Record<string, unknown>;
    for (const k of new Set([...Object.keys(ao), ...Object.keys(bo)])) {
      const m = close(ao[k], bo[k], tol, `${p}.${k}`);
      if (m) return m;
    }
    return null;
  }
  return a === b ? null : `${p}: ${String(a)} vs ${String(b)}`;
}

describe("RoPE as a phasor bank = reference/visuals.py", () => {
  it("the bank's constants", () => {
    expect([ROPE_POSITIONS, ROPE_SHOWN, ROPE_GAP]).toEqual([
      fx.rope.positions,
      fx.rope.shown,
      fx.rope.gap,
    ]);
    expect(ropeShown(64)).toEqual([0, 9, 18, 27, 36, 45, 54, 63]);
  });
  for (const [key, b] of Object.entries(fx.rope.banks)) {
    const mode = key.split("|")[1] as RopeMode;
    it(key, () => {
      const got = ropeBank(b.head_dim, b.theta, mode);
      expect(close(got, b.states)).toBe(null);
    });
  }
  it("the score depends only on the distance: moving both changes nothing", () => {
    const r = CHAPTERS.rope.find((x) => x.id === "qwen3-8b")!;
    const both = ropeBank(r.head_dim, r.theta, "both");
    const q = ropeBank(r.head_dim, r.theta, "query");
    both.forEach((s) => expect(s.score).toBeCloseTo(q[ROPE_GAP]!.score, 12));
    // each pair's contribution is cos(Δφ), as rotating unit vectors gives
    for (const s of q)
      for (const p of s.pairs)
        expect(p.contrib).toBeCloseTo(Math.cos(p.angQ - p.angK), 12);
    expect(q[0]!.score).toBe(1);
    // the fastest pair turns once every 2π tokens
    expect(q[0]!.pairs[0]!.wavelength).toBeCloseTo(2 * Math.PI, 12);
    expect(ropeOmegas(128, 1e6)).toHaveLength(64);
  });
  it("captions", () => {
    const r = CHAPTERS.rope.find((x) => x.id === "mistral-7b")!;
    const q = ropeBank(r.head_dim, r.theta, "query");
    expect(ropeCaption(q[5]!, "query")).toMatch(
      /^Query at position 5, key at 0: 5 tokens apart\. The fastest pair turns once every 6\.3 tokens, the slowest once every \d+K tokens; the score \(mean of every pair's cos\(Δφ\)\) is 0\.\d{3}\.$/,
    );
    const b = ropeBank(r.head_dim, r.theta, "both");
    expect(ropeCaption(b[3]!, "both")).toMatch(
      /^Query at 11, key at 3: both moved, still 8 tokens apart\./,
    );
    const big = ropeBank(128, 5e8, "query");
    expect(ropeCaption(big[1]!, "query")).toMatch(
      /1 token apart.*\d\.\dB tokens/,
    );
    expect(ropeCaption(ropeBank(128, 2e6, "query")[1]!, "query")).toMatch(
      /\d\.\dM tokens/,
    );
  });
});

describe("head sharing = the cost model", () => {
  const got = headSharing(CHAPTERS.attention.variants);
  it("MHA, GQA, MQA and MLA, their units and KV bytes per token", () => {
    expect(close(got, fx.sharing, 0)).toBe(null);
    expect(got.map((v) => v.bytesPerToken)).toEqual([
      524288, 131072, 16384, 36864,
    ]);
  });
  it("captions", () => {
    expect(sharingCaption(got[1]!, got[0]!)).toBe(
      "GQA (8 KV heads): each key/value head serves 4 query heads. KV cache 128 KiB per token over 32 layers, 4.0× less than MHA.",
    );
    expect(sharingCaption(got[3]!, got[0]!)).toMatch(
      /^MLA: every head reads one shared latent of 576 numbers/,
    );
    expect(sharingCaption(got[0]!, got[0]!)).toMatch(/own key and value/);
    expect(
      sharingCaption({ ...got[2]!, bytesPerToken: 512 }, got[0]!),
    ).toContain("512 B per token");
  });
});

describe("MoE routing = reference/visuals.py", () => {
  it("the configuration", () => expect(MOE).toEqual(fx.moe.cfg));
  for (const k of ["on", "off"] as const)
    it(`balancing ${k}: every step's routes, loads and biases`, () => {
      const got = moeRouting(k === "on" ? MOE.gamma : 0);
      expect(got).toHaveLength(MOE.steps);
      expect(close(got, fx.moe[k])).toBe(null);
    });
  it("the bias evens the load; without it the hot experts stay hot", () => {
    const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
    const on = moeRouting(MOE.gamma);
    const off = moeRouting(0);
    const late = (s: typeof on) => mean(s.slice(-8).map((x) => x.imbalance));
    expect(on[0]!.imbalance).toBeGreaterThan(2);
    expect(late(on)).toBeLessThan(1.4);
    expect(late(off)).toBeGreaterThan(2);
    // every token picks top_k distinct experts, gates summing to 1
    for (const r of on[5]!.routes) {
      expect(new Set(r.experts).size).toBe(MOE.top_k);
      expect(r.gates.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
    }
    on.forEach((s) =>
      expect(s.loads.reduce((a, b) => a + b, 0)).toBe(MOE.tokens * MOE.top_k),
    );
  });
  it("captions", () => {
    const on = moeRouting(MOE.gamma);
    expect(moeCaption(on[0]!, MOE, true)).toMatch(
      /^Step 1: 64 tokens each pick their top 2 of 8 experts\. Busiest: expert \d with \d+ \(\d\.\d\d× the mean of 16\); quietest: expert \d with \d+\. Next step the bias of every expert above 16 tokens drops by 0\.02, of every one below rises by 0\.02\.$/,
    );
    expect(moeCaption(on[0]!, MOE, false)).toMatch(
      /No bias: the hot experts stay hot\.$/,
    );
  });
});
