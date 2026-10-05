/**
 * The chapter interactives' starting data, generated with the Python
 * reference by scripts/make_chapter_fixtures.py (src/data/chapters.json):
 * the attention variants, the base specs the sliders rebuild, the RoPE
 * settings read from pinned config.json snapshots, and the model lists.
 */
import raw from "@/data/chapters.json";

import type { Spec } from "@/lib/arch/costModel";

export type RopeModel = {
  id: string;
  name: string;
  theta: number;
  head_dim: number;
  fraction: number;
  rotated: string;
  config: string;
};

export type ChapterData = {
  attention: {
    base: string;
    variants: { id: string; label: string; spec: Spec }[];
  };
  moe: { base: string; spec: Spec };
  depth_width: { vocab: number };
  rope: RopeModel[];
  context: string[];
  mtp: string[];
  loops: { id: string; spec: Spec };
  ced: { id: string; spec: Spec; t5: string; t5_spec: Spec };
};

export const CHAPTERS = raw as unknown as ChapterData;
