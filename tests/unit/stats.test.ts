import { describe, expect, it } from "vitest";

import { toRow } from "@/lib/arch/rows";
import { MODELS, getModel } from "@/lib/data";
import { repoFile } from "@/lib/site";
import { stats, weightChecks } from "@/lib/stats";

describe("counts quoted in the prose", () => {
  it("match the data", () => {
    const s = stats();
    expect(s.models).toBe(MODELS.length);
    expect(s.gallery).toBe(109);
    expect(s.models - s.gallery).toBe(s.extras);
    expect(s.estimates).toBeGreaterThan(0);
  });
  it("every estimate comes from one source (the about page says so)", () => {
    const srcs = new Set(
      MODELS.flatMap((m) => (m.facts.estimates ?? []).map((e) => e.src)),
    );
    expect(srcs.size).toBe(1);
  });
  it("weight checks: most open models match their published weights", () => {
    const w = weightChecks();
    expect(w.compared).toBeGreaterThan(60);
    expect(w.within / w.compared).toBeGreaterThan(0.85);
  });
  it("table rows", () => {
    const r = toRow(getModel("deepseek-v3")!);
    expect(r.total).toBe(671e9);
    expect(r.totalSt).toBe("disclosed");
    expect(r.layers).toBe(61);
    const g = toRow(getModel("gpt-4")!);
    expect(g.total).toBeNull();
    expect(g.estimate?.v).toBe(1.76e12);
    const glm = toRow(getModel("glm-4.7")!);
    expect(glm.totalSt).toBe("modelled");
    expect(toRow(getModel("llama-3-8b")!).active).toBeNull();
    expect(repoFile("README.md")).toContain("/blob/main/README.md");
  });
});
