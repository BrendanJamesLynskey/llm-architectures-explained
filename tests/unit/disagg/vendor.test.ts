/**
 * The vendored engine is the file at the pinned commit (brief 11's CED
 * commit), byte for byte, and every fixture and workload was generated at
 * that same commit.
 */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { engine } from "@/lib/disagg/engine";

import { draws, readJson, results } from "./helpers";

const vendored = readJson<{ commit: string; sha256: string; path: string }>(
  "src/lib/disagg/vendor/VENDORED.json",
);

describe("vendored sim_engine.js", () => {
  it("matches the SHA-256 recorded with its commit", () => {
    const src = readFileSync(
      join(process.cwd(), "src/lib/disagg/vendor/sim_engine.js"),
    );
    expect(createHash("sha256").update(src).digest("hex")).toBe(
      vendored.sha256,
    );
    expect(vendored.path).toBe("web/sim_engine.js");
  });

  it("is pinned at brief 11's commit, which added the CED option", () => {
    expect(vendored.commit).toBe("e674e189c50d45aa5d87bfd774687398120c724c");
    expect(engine.MODELS["llama3-70b-ced"]).toBeDefined();
    expect(engine.MODELS["llama3-8b-ced"]).toBeDefined();
  });

  it("fixtures and workloads come from the same commit", () => {
    expect(
      readJson<{ commit: string }>("tests/unit/fixtures/disagg_ced_parity.json")
        .commit,
    ).toBe(vendored.commit);
    expect(results.commit).toBe(vendored.commit);
    for (const w of results.s17) {
      const d = draws(w.prompt, w.output);
      expect(d.commit).toBe(vendored.commit);
      expect(d.rows).toHaveLength(1000);
    }
  });
});
