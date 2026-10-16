/**
 * The landing page's figure (brief 41): one point per model with
 * dimensions and a release month, its KV bytes per token at 32K from the
 * cost model, and its attention family.
 */
import { describe, expect, it } from "vitest";

import { kvCache } from "@/lib/arch/costModel";
import { specOf } from "@/lib/arch/features";
import {
  FAMILIES,
  LANDSCAPE_CONTEXT,
  familyOf,
  kvLandscape,
} from "@/lib/arch/landscape";
import { getModel } from "@/lib/data";

describe("the KV landscape", () => {
  const pts = kvLandscape();
  it("covers every model with dimensions and a release month", () => {
    expect(pts.length).toBeGreaterThan(130);
    for (const p of pts) {
      expect(FAMILIES).toContain(p.family);
      expect(p.year).toBeGreaterThan(2017);
      expect(p.year).toBeLessThan(2027);
    }
  });
  it("bytes per token are the cost model's, averaged over 32K", () => {
    const p = pts.find((x) => x.id === "llama-3-8b")!;
    expect(p.bytesPerToken).toBe(131072);
    expect(p.family).toBe("GQA");
    const d = pts.find((x) => x.id === "deepseek-v3")!;
    expect(d.family).toBe("MLA");
    expect(d.bytesPerToken).toBe(
      kvCache(specOf(getModel("deepseek-v3")!)!, LANDSCAPE_CONTEXT, 2)
        .total_bytes / LANDSCAPE_CONTEXT,
    );
    // a sliding-window model counts what its window holds
    const m = pts.find((x) => x.id === "mistral-7b")!;
    expect(m.family).toBe("sliding window");
    expect(m.bytesPerToken).toBeGreaterThan(0);
    expect(m.bytesPerToken).toBeLessThan(131072);
  });
  it("families from attention summaries", () => {
    expect(familyOf("GQA 32q/8kv")).toBe("GQA");
    expect(familyOf("MHA 25q/25kv")).toBe("MHA");
    expect(familyOf("MQA 71q/1kv")).toBe("MQA");
    expect(familyOf("23× Mamba-2 + 6× GQA 32q/2kv")).toBe("linear or hybrid");
    expect(familyOf("MLA 128h, latent 512")).toBe("MLA");
    expect(familyOf("GQA 32q/8kv, window 4096")).toBe("sliding window");
    expect(familyOf("something else")).toBeNull();
  });
});
