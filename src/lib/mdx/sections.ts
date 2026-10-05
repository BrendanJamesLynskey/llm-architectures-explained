/**
 * Chapter catalogue + filesystem loader for /learn content.
 *
 * MDX sources live under `/content/chapters/`, one per axis along which LLM
 * architectures vary (brief 13 §4). Their slugs and order are defined here
 * (single source of truth); the `[slug]` route validates incoming params
 * against this list before reading from disk. Same shape as LLM Inference
 * Explained's `src/lib/mdx/sections.ts`.
 */
import { readFile } from "node:fs/promises";
import { join } from "node:path";

export const SECTIONS = [
  {
    slug: "01-attention",
    title: "The attention family",
    summary:
      "MHA, GQA, MQA, MLA, sliding windows, sparse attention, and linear and state-space hybrids.",
  },
  {
    slug: "02-positional-encoding",
    title: "Positional encoding",
    summary: "Learned, sinusoidal, ALiBi, RoPE, partial RoPE and NoPE layers.",
  },
  {
    slug: "03-normalisation",
    title: "Where the norms go",
    summary: "Post-LN, pre-norm, sandwich and post-norm, and QK-norm.",
  },
  {
    slug: "04-dense-and-moe",
    title: "Dense and mixture of experts",
    summary:
      "Expert counts, active experts, shared experts and dense first layers.",
  },
  {
    slug: "05-depth-and-width",
    title: "Depth and width",
    summary: "How layer count and model width trade parameters, KV and FLOPs.",
  },
  {
    slug: "06-long-context",
    title: "Long-context techniques",
    summary:
      "RoPE scaling, windows, compressed and sparse attention, and hybrids, at 1M tokens.",
  },
  {
    slug: "07-multi-token-prediction",
    title: "Multi-token prediction",
    summary: "Extra heads that draft the next tokens, and what they buy.",
  },
  {
    slug: "08-looped-and-parallel-blocks",
    title: "Looped and parallel blocks",
    summary:
      "Reusing the layer stack, and running attention and FFN side by side.",
  },
  {
    slug: "09-encoder-decoder-and-ced",
    title: "Encoder-decoder and the causal encoder-decoder",
    summary:
      "T5's split, and DeepSeek-V4.1-Flash's encoder-only prefill, simulated live.",
  },
] as const;

export type SectionSlug = (typeof SECTIONS)[number]["slug"];

const SLUG_SET = new Set<string>(SECTIONS.map((s) => s.slug));

export function isValidSlug(slug: string): slug is SectionSlug {
  return SLUG_SET.has(slug);
}

export function getSectionMeta(slug: SectionSlug): (typeof SECTIONS)[number] {
  return SECTIONS.find((s) => s.slug === slug) ?? SECTIONS[0];
}

/** Read the raw MDX source for a chapter, or `null` if it doesn't exist. */
export async function readSectionMdx(
  slug: SectionSlug,
): Promise<string | null> {
  const path = join(process.cwd(), "content", "chapters", `${slug}.mdx`);
  try {
    return await readFile(path, "utf-8");
  } catch {
    return null;
  }
}
