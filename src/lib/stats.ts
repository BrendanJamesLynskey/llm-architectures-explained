/**
 * Counts quoted on the landing and about pages, computed from the data so
 * the prose can never drift from it (tests/unit/stats.test.ts).
 */
import { params } from "@/lib/arch/costModel";
import { specOf } from "@/lib/arch/features";
import { MODELS } from "@/lib/data";

export function stats() {
  const gallery = MODELS.filter((m) => m.in_gallery).length;
  const closed = MODELS.filter((m) => !m.open_weights).length;
  const withArch = MODELS.filter((m) => m.arch !== null).length;
  const estimates = MODELS.reduce(
    (n, m) => n + (m.facts.estimates?.length ?? 0),
    0,
  );
  const pinned = MODELS.filter(
    (m) =>
      (m.sources.hf as { config_available?: boolean } | undefined)
        ?.config_available,
  ).length;
  return {
    models: MODELS.length,
    gallery,
    extras: MODELS.length - gallery,
    closed,
    withArch,
    estimates,
    pinned,
  };
}

/** Open models whose published weight count can be compared with the model's. */
export function weightChecks() {
  let compared = 0;
  let within = 0;
  for (const m of MODELS) {
    const hw = m.checks.hf_weight_count;
    const s = specOf(m);
    if (!s || !hw?.v || hw.packed || m.multimodal || m.arch_from) continue;
    compared++;
    const p = params(s);
    const r = Math.min(
      Math.abs(p.total / hw.v - 1),
      Math.abs((p.total + p.mtp) / hw.v - 1),
    );
    if (r <= 0.005) within++;
  }
  return { compared, within };
}
