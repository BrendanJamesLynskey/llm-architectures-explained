/**
 * Provenance shown next to every value: a status chip and a link to the
 * exact source, with the location in a tooltip and in visually hidden text
 * for screen readers. Reported estimates get their own, unmistakable style
 * and always carry their label, source, date and confidence.
 *
 * Server Components (plain markup).
 */
import type { ReactNode } from "react";

import type { Field, Source, Status } from "@/lib/arch/types";

export const STATUS_LABEL: Record<Status | "reported-estimate", string> = {
  config: "config.json",
  disclosed: "lab",
  paper: "paper",
  code: "code",
  "not-disclosed": "not disclosed",
  "reported-estimate": "reported estimate",
};

const STATUS_CLASS: Record<Status | "reported-estimate", string> = {
  config:
    "border-emerald-300 bg-emerald-50 text-emerald-900 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-200",
  disclosed:
    "border-sky-300 bg-sky-50 text-sky-900 dark:border-sky-800 dark:bg-sky-950/40 dark:text-sky-200",
  paper:
    "border-indigo-300 bg-indigo-50 text-indigo-900 dark:border-indigo-800 dark:bg-indigo-950/40 dark:text-indigo-200",
  code: "border-neutral-300 bg-neutral-50 text-neutral-800 dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-200",
  "not-disclosed":
    "border-neutral-300 bg-white text-neutral-700 dark:border-neutral-700 dark:bg-neutral-950 dark:text-neutral-300",
  "reported-estimate":
    "border-dashed border-amber-500 bg-amber-50 text-amber-900 dark:border-amber-500 dark:bg-amber-950/40 dark:text-amber-200",
};

export function StatusChip({
  st,
}: {
  st: Status | "reported-estimate";
}): JSX.Element {
  return (
    <span
      data-status={st}
      className={`inline-block whitespace-nowrap rounded border px-1.5 py-px font-mono text-[0.65rem] ${STATUS_CLASS[st]}`}
    >
      {STATUS_LABEL[st]}
    </span>
  );
}

const A =
  "focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300";

/** The status chip, and a link to the source with the location as text. */
export function Provenance({
  field,
  sources,
}: {
  field: Field;
  sources: Record<string, Source>;
}): JSX.Element {
  const src = field.src ? sources[field.src] : undefined;
  return (
    <span className="inline-flex flex-wrap items-baseline gap-x-1.5 gap-y-0.5">
      <StatusChip st={field.st} />
      {src && (
        <a href={src.url} className={`${A} text-xs`} title={field.ref}>
          {sourceName(field.src!, src)}
        </a>
      )}
      {field.ref && (
        <span className="text-xs text-neutral-600 dark:text-neutral-400">
          {field.ref}
        </span>
      )}
      {field.note && (
        <span className="text-xs italic text-neutral-600 dark:text-neutral-400">
          {field.note}
        </span>
      )}
    </span>
  );
}

export function sourceName(id: string, s: Source): string {
  if (s.type === "config") return "config.json";
  if (s.type === "card") return "model card";
  if (s.type === "paper") return s.arxiv ? `arXiv ${s.arxiv}` : "paper";
  if (s.type === "code")
    return id === "code" ? "modelling code" : (s.title ?? "code");
  if (s.type === "doc") return "official docs";
  if (s.type === "blog") return "announcement";
  if (s.type === "reported-estimate") return s.publisher ?? id;
  return id;
}

/** An estimate, always labelled with its source, date and confidence. */
export function EstimateNote({
  value,
  source,
  quote,
}: {
  value: ReactNode;
  source: Source;
  quote: string;
}): JSX.Element {
  return (
    <span
      data-estimate
      className="inline-flex flex-col gap-0.5 rounded border border-dashed border-amber-500 bg-amber-50 px-2 py-1 text-xs text-amber-950 dark:bg-amber-950/40 dark:text-amber-100"
    >
      <span>
        <StatusChip st="reported-estimate" /> <strong>{value}</strong> (&ldquo;
        {quote}&rdquo;)
      </span>
      <span>
        Source:{" "}
        <a href={source.url} className="underline underline-offset-2">
          {source.publisher}
        </a>
        , {source.date}. {source.where}
      </span>
      <span>Confidence: {source.confidence}</span>
    </span>
  );
}
