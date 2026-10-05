/**
 * /models/[id]: one model, every value with its provenance, a diagram drawn
 * from its data, and what the cost model computes for it.
 *
 * Server Component, statically generated for every model.
 */
import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import {
  EstimateNote,
  Provenance,
  StatusChip,
  sourceName,
} from "@/components/ui/Provenance";
import { BlockDiagram } from "@/components/viz/BlockDiagram";
import { LineChart } from "@/components/viz/LineChart";
import {
  decodeFlops,
  kvCache,
  params,
  prefillFlops,
} from "@/lib/arch/costModel";
import {
  FEATURES,
  attentionSummary,
  features,
  specOf,
} from "@/lib/arch/features";
import {
  formatBytes,
  formatCount,
  formatFlops,
  formatTokens,
} from "@/lib/arch/format";
import type { Field, ModelRecord, Position } from "@/lib/arch/types";
import { MODELS, getModel } from "@/lib/data";

export function generateStaticParams(): { id: string }[] {
  return MODELS.map((m) => ({ id: m.id }));
}

export function generateMetadata({
  params: p,
}: {
  params: { id: string };
}): Metadata {
  const m = getModel(p.id);
  return m
    ? {
        title: m.name,
        description: `${m.name} (${m.lab}): architecture, sources and modelled costs.`,
      }
    : {};
}

const A =
  "focus-ring rounded text-accent underline decoration-accent/40 underline-offset-2 hover:decoration-accent dark:text-indigo-300";

function positionText(p: Position): string {
  const names: Record<string, string> = {
    rope: "RoPE",
    mrope: "multimodal RoPE",
    learned: "learned absolute",
    alibi: "ALiBi",
    "relative-bias": "relative position bias",
    sinusoidal: "sinusoidal",
    none: "none",
  };
  let s = names[p.scheme] ?? p.scheme;
  if (
    (p.scheme === "rope" || p.scheme === "mrope") &&
    p.rope_fraction !== null &&
    p.rope_fraction < 1
  ) {
    s += ` on ${Math.round(p.rope_fraction * 1000) / 10}% of each head`;
  }
  if (p.nope) s += `; ${p.nope}`;
  return s;
}

function valueText(key: string, f: Field): string {
  const v = f.v;
  if (v === null || v === undefined) return "not disclosed";
  if (key === "total_params" || key === "active_params")
    return formatCount(v as number);
  if (key === "context") return `${formatTokens(v as number)} tokens`;
  if (key === "position") return positionText(v as Position);
  if (typeof v === "boolean") return v ? "yes" : "no";
  return String(v);
}

const FACT_ROWS: [string, string][] = [
  ["released", "Released"],
  ["licence", "Licence"],
  ["total_params", "Total parameters"],
  ["active_params", "Active parameters"],
  ["context", "Context length"],
  ["moe", "Mixture of experts"],
  ["experts", "Experts"],
  ["norm_placement", "Norm placement"],
  ["norm_type", "Norm type"],
  ["qk_norm", "QK-norm"],
  ["position", "Positional encoding"],
  ["parallel_block", "Parallel attention and MLP"],
];

/** Flatten the arch block into labelled rows, keeping every field's provenance. */
function archRows(arch: Record<string, unknown>): { path: string; f: Field }[] {
  const out: { path: string; f: Field }[] = [];
  const walk = (x: unknown, path: string) => {
    if (x && typeof x === "object" && !Array.isArray(x)) {
      const o = x as Record<string, unknown>;
      if ("v" in o && "st" in o) {
        out.push({ path, f: o as unknown as Field });
        return;
      }
      for (const [k, v] of Object.entries(o))
        walk(v, path ? `${path}.${k}` : k);
    }
  };
  walk(arch, "");
  return out;
}

function archValue(f: Field): string {
  const v = f.v;
  if (Array.isArray(v)) {
    return v
      .map((r) => {
        const run = r as Record<string, unknown>;
        const extra = Object.entries(run)
          .filter(([k]) => !["mixer", "ffn", "n"].includes(k))
          .map(([k, x]) => `${k}=${String(x)}`)
          .join(",");
        return `${run.n}× ${run.mixer}/${run.ffn}${extra ? ` (${extra})` : ""}`;
      })
      .join(" · ");
  }
  if (v && typeof v === "object") return JSON.stringify(v);
  if (typeof v === "number") return v.toLocaleString("en-GB");
  return String(v);
}

function Modelled({ m }: { m: ModelRecord }): JSX.Element | null {
  const spec = specOf(m);
  if (!spec) return null;
  const p = params(spec);
  const ctx = m.facts.context.v ?? 131072;
  const kv = kvCache(spec, ctx, 2.0);
  const hw = m.checks.hf_weight_count;
  const rows: [string, string][] = [
    ["Parameters (modelled)", formatCount(p.total)],
    ["Active per token (modelled)", formatCount(p.active)],
    [
      "Without embeddings and output head",
      `${formatCount(p.non_embedding_total)} total, ${formatCount(p.non_embedding_active)} active`,
    ],
  ];
  if (p.mtp > 0)
    rows.push(["Multi-token-prediction layers (extra)", formatCount(p.mtp)]);
  if (hw?.v) {
    rows.push([
      "Published weights (Hugging Face count)",
      `${formatCount(hw.v)}${hw.packed ? " (packed low-bit tensors, so not comparable)" : ""}`,
    ]);
  }
  rows.push(
    [
      "KV cache per token, BF16 (layers that grow with context)",
      formatBytes(kv.bytes_per_token_unbounded),
    ],
    [
      `KV cache + state at ${formatTokens(ctx)} tokens, BF16`,
      formatBytes(kv.total_bytes),
    ],
    [
      "Decode FLOPs per token at 4K context",
      formatFlops(decodeFlops(spec, 4096)),
    ],
    ["Prefill FLOPs for a 4K prompt", formatFlops(prefillFlops(spec, 4096))],
  );
  return (
    <table className="w-full text-sm" data-testid="modelled">
      <tbody>
        {rows.map(([k, v]) => (
          <tr
            key={k}
            className="border-b border-neutral-200 dark:border-neutral-800"
          >
            <th
              scope="row"
              className="py-1.5 pr-4 text-left font-normal text-neutral-700 dark:text-neutral-300"
            >
              {k}
            </th>
            <td className="py-1.5 text-right font-mono">{v}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function ModelPage({
  params: p,
}: {
  params: { id: string };
}): JSX.Element {
  const m = getModel(p.id);
  if (!m) notFound();
  const spec = specOf(m);
  const feats = features(m);
  const ctxMax = Math.max(m.facts.context.v ?? 131072, 4096);
  const kvSeries = spec
    ? [
        {
          id: m.id,
          label: m.name,
          colour: "#4f46e5",
          points: Array.from({ length: 21 }, (_, i) => 2 ** (i + 1))
            .filter((c) => c <= Math.max(ctxMax, 2048) * 1.001)
            .map(
              (c) => [c, kvCache(spec, c, 2.0).total_bytes] as [number, number],
            ),
        },
      ]
    : [];
  const estimates = m.facts.estimates ?? [];
  return (
    <main className="mx-auto max-w-5xl px-4 py-10 sm:px-6">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        <Link href="/models" className="focus-ring rounded">
          /models
        </Link>
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">{m.name}</h1>
      <p className="mt-2 text-neutral-700 dark:text-neutral-300">
        {m.lab} · {m.family} ·{" "}
        {m.open_weights ? "open weights" : "closed weights"}
        {m.kind !== "decoder" &&
          ` · ${m.kind === "ced" ? "causal encoder-decoder" : m.kind}`}
        {m.multimodal && " · multimodal (text stack modelled)"}
      </p>
      {m.arch_from && (
        <p className="mt-2 text-sm text-neutral-700 dark:text-neutral-300">
          Architecture taken from{" "}
          <Link className={A} href={`/models/${m.arch_from.id}`}>
            {getModel(m.arch_from.id)?.name}
          </Link>
          : {m.arch_from.ref}
        </p>
      )}
      <ul className="mt-4 flex flex-wrap gap-1.5" aria-label="Features">
        {FEATURES.filter((x) => feats.has(x.id)).map((x) => (
          <li
            key={x.id}
            title={x.description}
            className="rounded-full border border-neutral-300 px-2 py-0.5 text-xs text-neutral-700 dark:border-neutral-700 dark:text-neutral-300"
          >
            {x.label}
          </li>
        ))}
      </ul>

      <section className="mt-8">
        <h2 className="text-xl font-semibold tracking-tight">
          Facts and where they come from
        </h2>
        <div
          tabIndex={0}
          role="region"
          aria-label="Scrollable content"
          className="mt-3 overflow-x-auto"
        >
          <table className="w-full text-sm">
            <tbody>
              {FACT_ROWS.filter(([k]) => k in m.facts).map(([k, label]) => {
                const f = (m.facts as unknown as Record<string, Field>)[k]!;
                return (
                  <tr
                    key={k}
                    className="border-b border-neutral-200 align-top dark:border-neutral-800"
                  >
                    <th
                      scope="row"
                      className="w-44 py-2 pr-3 text-left font-normal text-neutral-700 dark:text-neutral-300"
                    >
                      {label}
                    </th>
                    <td className="py-2 pr-3 font-medium">{valueText(k, f)}</td>
                    <td className="py-2">
                      <Provenance field={f} sources={m.sources} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {estimates.length > 0 && (
          <div className="mt-4" data-testid="estimates">
            <h3 className="font-semibold">Reported estimates</h3>
            <p className="mt-1 text-sm text-neutral-700 dark:text-neutral-300">
              {m.lab} has not disclosed these. They are third-party estimates,
              shown only with their label, source, date and confidence, and
              never used by the calculator unless you switch estimates on.
            </p>
            <ul className="mt-2 space-y-2">
              {estimates.map((e) => (
                <li key={e.field + e.src}>
                  <span className="mr-2 text-sm">
                    {e.field === "total_params" ? "Total parameters:" : e.field}
                  </span>
                  <EstimateNote
                    value={`≈${formatCount(e.v)}`}
                    source={m.sources[e.src]!}
                    quote={e.quote}
                  />
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>

      {spec ? (
        <>
          <section className="mt-10">
            <h2 className="text-xl font-semibold tracking-tight">
              Architecture, drawn from the data
            </h2>
            <p className="mt-1 text-sm text-neutral-700 dark:text-neutral-300">
              {attentionSummary(m)}. Each column is one layer: its token mixer
              above, its feed-forward block below. Paler columns reuse another
              layer&rsquo;s keys and values.
            </p>
            <figure
              tabIndex={0}
              className="mt-3 overflow-x-auto rounded-lg border border-neutral-200 bg-white p-3 dark:border-neutral-800 dark:bg-neutral-950"
              data-testid="diagram"
            >
              <BlockDiagram
                spec={spec}
                placement={m.facts.norm_placement?.v ?? "none"}
                title={`${m.name}: layer stack and blocks`}
              />
            </figure>
          </section>
          <section className="mt-10 grid gap-8 lg:grid-cols-2">
            <div className="min-w-0">
              <h2 className="text-xl font-semibold tracking-tight">
                Modelled costs
              </h2>
              <p className="mt-1 text-sm text-neutral-700 dark:text-neutral-300">
                From the{" "}
                <Link className={A} href="/about#cost-model">
                  cost model
                </Link>
                , batch size 1. Totals the lab states are in the table above;
                differences come from rounding, from what a lab counts, or from
                parts the model does not describe (listed on the about page).
              </p>
              <div className="mt-3">
                <Modelled m={m} />
              </div>
            </div>
            <div className="min-w-0">
              <h2 className="text-xl font-semibold tracking-tight">
                KV cache against context
              </h2>
              <div
                tabIndex={0}
                role="region"
                aria-label="Scrollable content"
                className="mt-3 overflow-x-auto rounded-lg border border-neutral-200 p-2 dark:border-neutral-800"
              >
                <LineChart
                  series={kvSeries}
                  xLabel="context (tokens)"
                  yLabel="KV cache + state (BF16)"
                  formatX={formatTokens}
                  formatY={(v) => formatBytes(v, 2)}
                  title={`${m.name}: KV cache bytes against context length`}
                />
              </div>
              <p className="mt-2 text-sm">
                <Link className={A} href={`/compare?m=${m.id}`}>
                  Compare with other models →
                </Link>
              </p>
            </div>
          </section>
          <section className="mt-10">
            <h2 className="text-xl font-semibold tracking-tight">
              Every architecture field
            </h2>
            <div
              tabIndex={0}
              role="region"
              aria-label="Scrollable content"
              className="mt-3 overflow-x-auto"
            >
              <table className="w-full text-sm" data-testid="arch-fields">
                <thead>
                  <tr className="border-b border-neutral-300 text-left dark:border-neutral-700">
                    <th className="py-1.5 pr-3 font-semibold">Field</th>
                    <th className="py-1.5 pr-3 font-semibold">Value</th>
                    <th className="py-1.5 font-semibold">Source</th>
                  </tr>
                </thead>
                <tbody>
                  {archRows(m.arch!).map(({ path, f }) => (
                    <tr
                      key={path}
                      className="border-b border-neutral-200 align-top dark:border-neutral-800"
                    >
                      <td className="py-1.5 pr-3 font-mono text-xs">{path}</td>
                      <td className="max-w-xs py-1.5 pr-3 font-mono text-xs [overflow-wrap:anywhere]">
                        {archValue(f)}
                      </td>
                      <td className="py-1.5">
                        <Provenance field={f} sources={m.sources} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      ) : (
        <section className="mt-10" data-testid="no-arch">
          <h2 className="text-xl font-semibold tracking-tight">Architecture</h2>
          <p className="mt-2 text-neutral-700 dark:text-neutral-300">
            <StatusChip st="not-disclosed" /> No diagram or modelled costs:{" "}
            {m.no_arch}.
          </p>
        </section>
      )}

      <section className="mt-10">
        <h2 className="text-xl font-semibold tracking-tight">Sources</h2>
        <ul className="mt-2 space-y-1 text-sm">
          {Object.entries(m.sources).map(([id, s]) => (
            <li key={id} className="[overflow-wrap:anywhere]">
              <a className={A} href={s.url}>
                {sourceName(id, s)}
              </a>
              {s.revision && (
                <span className="font-mono text-xs">
                  {" "}
                  @ {s.revision.slice(0, 7)}
                </span>
              )}
              {s.title && (
                <span className="text-neutral-600 dark:text-neutral-400">
                  {" "}
                  · {s.title}
                </span>
              )}
              {s.type === "reported-estimate" && (
                <span> · reported estimate, {s.date}</span>
              )}
            </li>
          ))}
        </ul>
        {m.gallery_name && (
          <p className="mt-3 text-xs text-neutral-600 dark:text-neutral-400">
            Listed in the LLM Architecture Gallery checklist as &ldquo;
            {m.gallery_name}&rdquo; (name only; see{" "}
            <Link className={A} href="/about#gallery">
              about
            </Link>
            ).
          </p>
        )}
      </section>
    </main>
  );
}
