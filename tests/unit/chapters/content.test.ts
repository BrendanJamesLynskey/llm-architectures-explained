/**
 * The chapters' MDX against the code and data it describes:
 * - every chapter in the catalogue has a file, and every file is listed;
 * - every code block is cut from the file its first line names (compared
 *   with whitespace collapsed, since Prettier re-wraps code in MDX);
 * - every component a chapter uses is in the MDX components map, every
 *   `<ModelsWith feature>` is a real feature with at least one model;
 * - every internal link resolves (chapter slugs, model ids);
 * - every arXiv id cited was checked at export.arxiv.org
 *   (data/sources/arxiv.json, written by scripts/verify_arxiv.py).
 */
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { modelsWith } from "@/components/mdx/ModelsWith";
import { FEATURES, type FeatureId } from "@/lib/arch/features";
import { MODELS } from "@/lib/data";
import {
  getSectionMeta,
  isValidSlug,
  readSectionMdx,
  SECTIONS,
} from "@/lib/mdx/sections";

const root = process.cwd();
const dir = join(root, "content/chapters");
const read = (p: string) => readFileSync(join(root, p), "utf-8");
const files = readdirSync(dir).filter((f) => f.endsWith(".mdx"));
const mdx = Object.fromEntries(
  files.map((f) => [f.replace(/\.mdx$/, ""), read(`content/chapters/${f}`)]),
);
const collapse = (s: string) => s.replace(/\s+/g, " ").trim();
const arxiv = JSON.parse(read("data/sources/arxiv.json")) as {
  papers: Record<string, { title: string }>;
};
const componentsSrc = read("src/lib/mdx/components.ts");

describe("chapter catalogue", () => {
  it("lists exactly the MDX files", () => {
    expect(SECTIONS.map((s) => s.slug).sort()).toEqual(Object.keys(mdx).sort());
    expect(SECTIONS).toHaveLength(9);
  });
  it("validates slugs and reads the MDX from disk", async () => {
    expect(isValidSlug("01-attention")).toBe(true);
    expect(isValidSlug("../etc/passwd")).toBe(false);
    expect(getSectionMeta("09-encoder-decoder-and-ced").title).toContain(
      "causal encoder-decoder",
    );
    expect(await readSectionMdx("01-attention")).toBe(mdx["01-attention"]);
    expect(
      await readSectionMdx("no-such-chapter" as "01-attention"),
    ).toBeNull();
  });
});

for (const [slug, src] of Object.entries(mdx)) {
  describe(slug, () => {
    it("code blocks are cut from the files they name", () => {
      const blocks = [...src.matchAll(/```ts\n([\s\S]*?)```/g)].map(
        (m) => m[1]!,
      );
      expect(blocks.length).toBeGreaterThan(0);
      for (const b of blocks) {
        const lines = b.split("\n");
        const head = /^\/\/ (src\/\S+\.ts)/.exec(lines[0]!);
        expect(head, `first line names a file: ${lines[0]}`).not.toBeNull();
        const body = lines
          .slice(1)
          .filter((l) => !l.trim().startsWith("//"))
          .join("\n");
        expect(collapse(read(head![1]!))).toContain(collapse(body));
      }
    });

    it("uses only registered components and real features", () => {
      const tags = new Set(
        [...src.matchAll(/<([A-Z][A-Za-z]+)/g)].map((m) => m[1]!),
      );
      for (const t of tags) expect(componentsSrc).toContain(`  ${t},`);
      for (const m of src.matchAll(/<ModelsWith feature="([^"]+)"/g)) {
        expect(FEATURES.map((f) => f.id)).toContain(m[1]);
        expect(modelsWith(m[1] as FeatureId).length).toBeGreaterThan(0);
      }
    });

    it("internal links resolve", () => {
      for (const m of src.matchAll(/\]\((\/[^)\s]*)\)/g)) {
        const url = new URL(m[1]!, "https://x.invalid");
        const [, top, rest] = url.pathname.split("/");
        if (top === "learn") {
          expect(SECTIONS.map((s) => s.slug)).toContain(rest);
        } else if (top === "models" && rest) {
          expect(MODELS.map((x) => x.id)).toContain(rest);
        } else if (top === "compare") {
          for (const id of url.searchParams.get("m")!.split(","))
            expect(MODELS.map((x) => x.id)).toContain(id);
        } else {
          expect(["models", "about", "timeline"]).toContain(top);
        }
      }
    });

    it("cites only arXiv papers checked at export.arxiv.org", () => {
      const ids = [...src.matchAll(/arxiv\.org\/abs\/(\d{4}\.\d{4,5})/g)].map(
        (m) => m[1]!,
      );
      for (const id of ids) expect(Object.keys(arxiv.papers)).toContain(id);
      for (const m of src.matchAll(/arXiv:(\d{4}\.\d{4,5})/g))
        expect(ids).toContain(m[1]);
    });
  });
}
