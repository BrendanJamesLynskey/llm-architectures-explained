/**
 * The model records, bundled by scripts/build_models.py into
 * src/data/models.json. Imported only by Server Components and tests, so the
 * full data never ships to the browser; client widgets fetch the compact
 * public/data/specs.json instead.
 */
import raw from "@/data/models.json";

import type { ModelRecord } from "@/lib/arch/types";

export const MODELS = raw as unknown as ModelRecord[];

export function getModel(id: string): ModelRecord | undefined {
  return MODELS.find((m) => m.id === id);
}

/** Models sorted by release month, then name. */
export function byRelease(): ModelRecord[] {
  return [...MODELS].sort((a, b) => {
    const ra = a.facts.released.v ?? "9999";
    const rb = b.facts.released.v ?? "9999";
    return ra === rb ? a.name.localeCompare(b.name) : ra.localeCompare(rb);
  });
}
