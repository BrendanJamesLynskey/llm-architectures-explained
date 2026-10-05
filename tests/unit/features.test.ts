/**
 * Features are derived from the data, so these tests pin them to models
 * whose designs are well known, and check the formatting helpers.
 */
import { describe, expect, it } from "vitest";

import {
  type Spec,
  ffnParams,
  mixerParams,
  sumMin,
} from "@/lib/arch/costModel";
import {
  FEATURES,
  attentionSummary,
  features,
  layerCount,
  released,
  specOf,
  stripFields,
} from "@/lib/arch/features";
import {
  formatBytes,
  formatCount,
  formatFlops,
  formatTokens,
} from "@/lib/arch/format";
import { MODELS, byRelease, getModel } from "@/lib/data";

const f = (id: string) => features(getModel(id)!);

describe("features from data", () => {
  it("attention families", () => {
    expect(f("llama-2-70b").has("gqa")).toBe(true);
    expect(f("gpt-2-xl").has("mha")).toBe(true);
    expect(f("palm-540b").has("mqa")).toBe(true);
    expect(f("deepseek-v3").has("mla")).toBe(true);
    expect(f("deepseek-v3.2").has("sparse")).toBe(true);
    expect(f("gemma-3-27b").has("sliding")).toBe(true);
    expect(f("qwen3-next-80b-a3b").has("deltanet")).toBe(true);
    expect(f("qwen3-next-80b-a3b").has("hybrid")).toBe(true);
    expect(f("jamba").has("ssm")).toBe(true);
    expect(f("lfm2.5-1.2b").has("conv")).toBe(true);
    expect(f("deepseek-v4-flash").has("compressed")).toBe(true);
    expect(f("rwkv-4-14b").has("linear")).toBe(true);
    expect(f("minimax-m3").has("sparse")).toBe(true);
    expect(f("motif-3-beta").has("sliding")).toBe(true);
  });

  it("positions, norms and blocks", () => {
    expect(f("smollm3-3b").has("nope")).toBe(true);
    expect(f("jamba").has("nope")).toBe(true);
    expect(f("rwkv-4-14b").has("nope")).toBe(false);
    expect(f("glm-4.5").has("partial-rope")).toBe(true);
    expect(f("bloom").has("alibi")).toBe(true);
    expect(f("gpt-2-xl").has("learned")).toBe(true);
    expect(f("t5-11b").has("relative-bias")).toBe(true);
    expect(f("olmo-2-7b").has("post-norm")).toBe(true);
    expect(f("gemma-3-27b").has("sandwich")).toBe(true);
    expect(f("transformer-base").has("post-ln")).toBe(true);
    expect(f("qwen3-8b").has("qk-norm")).toBe(true);
    expect(f("llama-3-8b").has("pre-norm")).toBe(true);
    expect(f("gpt-j-6b").has("parallel")).toBe(true);
  });

  it("feed-forward and the rest", () => {
    expect(f("deepseek-v3").has("moe")).toBe(true);
    expect(f("deepseek-v3").has("shared-expert")).toBe(true);
    expect(f("deepseek-v3").has("dense-prefix")).toBe(true);
    expect(f("deepseek-v3").has("mtp")).toBe(true);
    expect(f("nemotron-3-super").has("latent-moe")).toBe(true);
    expect(f("ouro-2.6b").has("looped")).toBe(true);
    expect(f("gemma-4-e2b").has("kv-sharing")).toBe(true);
    expect(f("deepseek-v4.1-flash").has("kv-sharing")).toBe(true);
    expect(f("t5-11b").has("encoder-decoder")).toBe(true);
    expect(f("deepseek-v4.1-flash").has("ced")).toBe(true);
    expect(f("gemini-3-pro").has("moe")).toBe(true);
    expect(f("llama-4-maverick").has("moe")).toBe(true);
    expect(f("llama-4-maverick").has("shared-expert")).toBe(true);
    expect(f("llama-4-maverick").has("chunked")).toBe(true);
    expect(f("llama-4-maverick").has("nope")).toBe(true);
    expect(f("llama-4-maverick").has("sliding")).toBe(false);
    // Dense layers interleaved with MoE layers are not a dense prefix.
    expect(f("llama-4-maverick").has("dense-prefix")).toBe(false);
    expect(f("glam").has("dense-prefix")).toBe(false);
    expect(f("jamba").has("dense-prefix")).toBe(false);
    expect(f("step-3.5-flash").has("dense-prefix")).toBe(true);
    expect(f("deepseek-moe-16b").has("dense-prefix")).toBe(true);
    expect(f("gpt-4").size).toBe(0);
  });

  it("every feature is used by at least one model", () => {
    const used = new Set<string>();
    for (const m of MODELS) for (const x of features(m)) used.add(x);
    for (const x of FEATURES) expect(used.has(x.id), x.id).toBe(true);
  });

  it("summaries", () => {
    expect(attentionSummary(getModel("llama-3-8b")!)).toBe("GQA 32q/8kv");
    expect(attentionSummary(getModel("gemma-3-27b")!)).toContain("window 1024");
    expect(attentionSummary(getModel("deepseek-v3.2")!)).toBe(
      "MLA 128h, latent 512, top-2048",
    );
    expect(attentionSummary(getModel("qwen3-next-80b-a3b")!)).toContain(
      "Gated DeltaNet",
    );
    expect(attentionSummary(getModel("deepseek-v4-flash")!)).toContain(
      "compressed",
    );
    expect(attentionSummary(getModel("palm-540b")!)).toBe("MQA 48q/1kv");
    expect(attentionSummary(getModel("gpt-2-xl")!)).toBe("MHA 25q/25kv");
    expect(attentionSummary(getModel("motif-3-beta")!)).toContain("window");
    expect(attentionSummary(getModel("gpt-4")!)).toBe("not disclosed");
    expect(attentionSummary(getModel("llama-4-maverick")!)).toBe(
      "36× GQA 40q/8kv, chunks of 8192 + 12× GQA 40q/8kv",
    );
    expect(layerCount(specOf(getModel("llama-4-maverick")!)!)).toBe(48);
    expect(layerCount(specOf(getModel("llama-3-8b")!)!)).toBe(32);
  });

  it("release dates and ordering", () => {
    expect(released(getModel("gpt-3")!)).toBeCloseTo(2020 + 4 / 12);
    const order = byRelease().map((m) => m.id);
    expect(order.indexOf("transformer-base")).toBeLessThan(
      order.indexOf("gpt-3"),
    );
    expect(
      released({
        ...getModel("gpt-3")!,
        facts: {
          ...getModel("gpt-3")!.facts,
          released: { v: null, st: "not-disclosed" },
        },
      }),
    ).toBeNull();
  });

  it("stripFields and specOf", () => {
    expect(
      stripFields({ a: { v: 1, st: "config" }, b: [{ v: 2, st: "code" }] }),
    ).toEqual({ a: 1, b: [2] });
    expect(specOf(getModel("gpt-4")!)).toBeNull();
  });
});

describe("formatting", () => {
  it("counts", () => {
    expect(formatCount(235e9)).toBe("235B");
    expect(formatCount(1.6e12)).toBe("1.6T");
    expect(formatCount(760e6)).toBe("760M");
    expect(formatCount(1500)).toBe("1.5K");
    expect(formatCount(12)).toBe("12");
    expect(formatCount(null)).toBe("—");
  });
  it("bytes", () => {
    expect(formatBytes(1536)).toBe("1.5 KiB");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(3 * 1024 ** 3)).toBe("3 GiB");
    expect(formatBytes(undefined)).toBe("—");
  });
  it("flops and tokens", () => {
    expect(formatFlops(3.2e12)).toBe("3.2 TFLOP");
    expect(formatFlops(5e20)).toBe("500 EFLOP");
    expect(formatFlops(12)).toBe("12 FLOP");
    expect(formatFlops(NaN)).toBe("—");
    expect(formatTokens(131072)).toBe("128K");
    expect(formatTokens(1048576)).toBe("1M");
    expect(formatTokens(1050000)).toBe("1,050,000");
    expect(formatTokens(null)).toBe("—");
  });
});

describe("cost model edge cases", () => {
  it("rejects unknown block types and handles empty blocks", () => {
    expect(() => mixerParams({ type: "bogus" }, 8)).toThrow();
    expect(() => ffnParams({ type: "bogus" }, 8)).toThrow();
    expect(mixerParams({ type: "none" }, 8)).toBe(0);
    expect(ffnParams({ type: "none" }, 8)).toEqual([0, 0]);
    expect(sumMin(10, 0)).toBe(55);
    expect(sumMin(10, 4)).toBe(10 + 6 * 4);
  });
  it("a spec without an encoder layout expands to nothing", () => {
    const s = specOf(getModel("llama-3-8b")!) as Spec;
    expect(s.encoder_layout).toBeUndefined();
  });
});
