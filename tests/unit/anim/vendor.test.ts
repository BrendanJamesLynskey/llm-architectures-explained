/**
 * The animation frame and its parts copied from Inference Trade-offs
 * Explained (brief 41) are byte for byte the files src/lib/anim/VENDORED.json
 * records.
 */
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

const V = JSON.parse(
  readFileSync(join(process.cwd(), "src/lib/anim/VENDORED.json"), "utf-8"),
) as { commit: string; files: { path: string; sha256: string }[] };

describe("vendored from Inference Trade-offs Explained", () => {
  it("names its commit", () => expect(V.commit).toMatch(/^[0-9a-f]{40}$/));
  for (const f of V.files)
    it(f.path, () =>
      expect(
        createHash("sha256")
          .update(readFileSync(join(process.cwd(), f.path)))
          .digest("hex"),
      ).toBe(f.sha256),
    );
});
