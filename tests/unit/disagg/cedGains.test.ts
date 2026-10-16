/** The landing page's CED range is results.md §17's gains (brief 41). */
import { describe, expect, it } from "vitest";

import { cedGains, cedRange } from "@/lib/disagg/cedGains";

import { results } from "./helpers";

describe("CED gains", () => {
  it("are results.md's cells, for every workload and both replay placements", () => {
    const g = cedGains();
    expect(g).toHaveLength(8); // four workloads, two replay placements
    for (const x of g) {
      const line = results.s17.find((w) => w.workload === x.workload)!.rows[
        x.variant
      ]!.line;
      expect(line).toContain(`| ${x.gain.toFixed(2)}x |`);
    }
    expect(cedRange()).toEqual({ lo: "1.52", hi: "2.14" });
  });
});
