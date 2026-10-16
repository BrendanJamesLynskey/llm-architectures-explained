/**
 * The landing page's figure: every model with dimensions, placed by its
 * release month and the KV cache one token costs it at a 32K context
 * (BF16, from the cost model; windowed layers and recurrent state averaged
 * over the context, so a sliding-window or linear-attention layer counts
 * for what it really holds). Colour: the attention family the model's
 * layers use, the most distinctive one winning for a hybrid.
 */
import { MODELS } from "@/lib/data";

import { kvCache } from "./costModel";
import { attentionSummary, specOf } from "./features";

export const LANDSCAPE_CONTEXT = 32768;

export const FAMILIES = [
  "MHA",
  "GQA",
  "MQA",
  "sliding window",
  "MLA",
  "linear or hybrid",
] as const;
export type Family = (typeof FAMILIES)[number];

/** The attention family of a model, from its attention summary. */
export function familyOf(attention: string): Family | null {
  if (
    /Mamba|DeltaNet|Delta Attention|Linear attention|RWKV|mLSTM|convolution/.test(
      attention,
    )
  )
    return "linear or hybrid";
  if (/MLA|compressed/.test(attention)) return "MLA";
  if (/window|chunks/.test(attention)) return "sliding window";
  if (/MQA/.test(attention)) return "MQA";
  if (/GQA/.test(attention)) return "GQA";
  if (/MHA/.test(attention)) return "MHA";
  return null;
}

export type LandscapePoint = {
  id: string;
  name: string;
  /** Release as a fractional year (month centre). */
  year: number;
  bytesPerToken: number;
  family: Family;
};

/** Every model with dimensions and a release month, one point each. */
export function kvLandscape(): LandscapePoint[] {
  const out: LandscapePoint[] = [];
  for (const m of MODELS) {
    const spec = specOf(m);
    const rel = m.facts.released.v;
    if (!spec || !rel) continue;
    const fam = familyOf(attentionSummary(m));
    if (!fam) continue;
    const [y, mo] = rel.split("-").map(Number);
    const kv = kvCache(spec, LANDSCAPE_CONTEXT, 2.0);
    out.push({
      id: m.id,
      name: m.name,
      year: y! + ((mo ?? 6) - 0.5) / 12,
      bytesPerToken: kv.total_bytes / LANDSCAPE_CONTEXT,
      family: fam,
    });
  }
  return out;
}
