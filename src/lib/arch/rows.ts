/**
 * The model table's rows, computed on the server from the records so the
 * client receives only what it displays.
 */
import { kvCache, params } from "./costModel";
import {
  attentionSummary,
  features,
  layerCount,
  specOf,
  type FeatureId,
} from "./features";
import type { ModelRecord, Status } from "./types";

export type Row = {
  id: string;
  name: string;
  lab: string;
  released: string | null;
  open: boolean;
  inGallery: boolean;
  total: number | null;
  totalSt: Status | "modelled";
  active: number | null;
  activeSt: Status | "modelled";
  estimate: { v: number; publisher: string; date: string } | null;
  layers: number | null;
  dModel: number | null;
  attention: string;
  kvPerToken: number | null;
  context: number | null;
  features: FeatureId[];
};

export function toRow(m: ModelRecord): Row {
  const spec = specOf(m);
  const p = spec ? params(spec) : null;
  const statedTotal = m.facts.total_params.v;
  const statedActive = m.facts.active_params.v;
  const est = (m.facts.estimates ?? []).find((e) => e.field === "total_params");
  const estSrc = est ? m.sources[est.src]! : null;
  return {
    id: m.id,
    name: m.name,
    lab: m.lab,
    released: m.facts.released.v,
    open: m.open_weights,
    inGallery: m.in_gallery,
    total: statedTotal ?? p?.total ?? null,
    totalSt:
      statedTotal !== null
        ? m.facts.total_params.st
        : p
          ? "modelled"
          : "not-disclosed",
    active:
      statedActive ??
      (p && p.active !== p.total
        ? p.active
        : statedTotal !== null
          ? null
          : (p?.active ?? null)),
    activeSt:
      statedActive !== null
        ? m.facts.active_params.st
        : p
          ? "modelled"
          : "not-disclosed",
    estimate:
      est && estSrc
        ? { v: est.v, publisher: estSrc.publisher!, date: estSrc.date! }
        : null,
    layers: spec ? layerCount(spec) : null,
    dModel: spec ? spec.d_model : null,
    attention: attentionSummary(m),
    kvPerToken: spec ? kvCache(spec, 1, 2.0).bytes_per_token_unbounded : null,
    context: m.facts.context.v,
    features: [...features(m)],
  };
}
