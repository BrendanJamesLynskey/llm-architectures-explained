/**
 * Types of the model records in data/models/*.yaml (bundled as
 * src/data/models.json by scripts/build_models.py). The JSON Schema in
 * data/schema/model.schema.json is the authority; these types mirror it.
 */
import type { Spec } from "./costModel";

export type Status =
  | "disclosed"
  | "config"
  | "paper"
  | "code"
  | "not-disclosed";

/** One value with its provenance. */
export type Field<T = unknown> = {
  v: T;
  st: Status;
  src?: string;
  ref?: string;
  note?: string;
};

export type Source = {
  type:
    | "config"
    | "card"
    | "paper"
    | "blog"
    | "doc"
    | "code"
    | "reported-estimate";
  url: string;
  title?: string;
  repo?: string;
  revision?: string;
  arxiv?: string;
  publisher?: string;
  date?: string;
  where?: string;
  confidence?: string;
  transcription?: string;
  snapshot?: string;
};

export type Estimate = {
  field: "total_params" | "active_params" | "context";
  v: number;
  st: "reported-estimate";
  src: string;
  quote: string;
};

export type Position = {
  scheme:
    | "rope"
    | "mrope"
    | "learned"
    | "alibi"
    | "relative-bias"
    | "sinusoidal"
    | "none";
  rope_fraction: number | null;
  nope: string | null;
};

export type Facts = {
  released: Field<string | null>;
  licence: Field<string | null>;
  total_params: Field<number | null>;
  active_params: Field<number | null>;
  context: Field<number | null>;
  norm_placement?: Field<string>;
  norm_type?: Field<string>;
  qk_norm?: Field<boolean>;
  position?: Field<Position>;
  parallel_block?: Field<boolean>;
  moe?: Field<boolean>;
  experts?: Field<number>;
  estimates?: Estimate[];
};

/** The arch block keeps provenance on every leaf. */
export type ArchFields = Record<string, unknown>;

export type ModelRecord = {
  id: string;
  name: string;
  gallery_name: string | null;
  lab: string;
  family: string;
  open_weights: boolean;
  kind: "decoder" | "encoder" | "encoder-decoder" | "ced";
  multimodal: boolean;
  no_arch: string | null;
  arch_from?: { id: string; ref: string };
  sources: Record<string, Source>;
  facts: Facts;
  arch: ArchFields | null;
  checks: {
    hf_weight_count?: {
      v: number | null;
      dtypes: string[];
      packed: boolean;
      ref: string;
    };
  };
  in_gallery: boolean;
};

/** The compact client-side record in public/data/specs.json. */
export type CompactModel = {
  id: string;
  name: string;
  lab: string;
  open: boolean;
  kind: string;
  released: string | null;
  total: { v: number | null; st: Status };
  active: { v: number | null; st: Status };
  context: number | null;
  norm: string | null;
  estimates: {
    field: string;
    v: number;
    src: string;
    publisher: string;
    date: string;
    confidence: string;
    url: string;
  }[];
  spec: Spec | null;
};
