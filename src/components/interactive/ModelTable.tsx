"use client";

/**
 * The filterable, sortable model table. Rows arrive pre-computed from the
 * server (src/lib/arch/rows.ts). Reported estimates for closed models stay
 * hidden until the reader switches them on, and are then styled and
 * labelled distinctly.
 */
import Link from "next/link";
import { useMemo, useState } from "react";

import { StatusChip } from "@/components/ui/Provenance";
import { FEATURES, type FeatureId } from "@/lib/arch/features";
import { formatBytes, formatCount, formatTokens } from "@/lib/arch/format";
import type { Row } from "@/lib/arch/rows";

type Key =
  | "name"
  | "lab"
  | "released"
  | "total"
  | "active"
  | "layers"
  | "dModel"
  | "kvPerToken"
  | "context";

const COLUMNS: { key: Key; label: string; numeric?: boolean }[] = [
  { key: "name", label: "Model" },
  { key: "lab", label: "Lab" },
  { key: "released", label: "Released" },
  { key: "total", label: "Total", numeric: true },
  { key: "active", label: "Active", numeric: true },
  { key: "layers", label: "Layers", numeric: true },
  { key: "dModel", label: "Width", numeric: true },
  { key: "kvPerToken", label: "KV / token", numeric: true },
  { key: "context", label: "Context", numeric: true },
];

const FILTER_FEATURES: FeatureId[] = [
  "gqa",
  "mqa",
  "mla",
  "sliding",
  "sparse",
  "linear",
  "deltanet",
  "ssm",
  "hybrid",
  "moe",
  "shared-expert",
  "mtp",
  "qk-norm",
  "nope",
  "partial-rope",
  "post-norm",
  "sandwich",
  "parallel",
];

export function ModelTable({ rows }: { rows: Row[] }): JSX.Element {
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<{ key: Key; dir: 1 | -1 }>({
    key: "released",
    dir: -1,
  });
  const [need, setNeed] = useState<FeatureId[]>([]);
  const [weights, setWeights] = useState<"all" | "open" | "closed">("all");
  const [showEst, setShowEst] = useState(false);

  const shown = useMemo(() => {
    const ql = q.trim().toLowerCase();
    const r = rows.filter(
      (x) =>
        (!ql ||
          `${x.name} ${x.lab} ${x.attention}`.toLowerCase().includes(ql)) &&
        need.every((f) => x.features.includes(f)) &&
        (weights === "all" || (weights === "open") === x.open),
    );
    const val = (x: Row): number | string | null => {
      if (sort.key === "total" && x.total === null && showEst && x.estimate)
        return x.estimate.v;
      return x[sort.key];
    };
    return [...r].sort((a, b) => {
      const va = val(a);
      const vb = val(b);
      if (va === null && vb === null) return a.name.localeCompare(b.name);
      if (va === null) return 1;
      if (vb === null) return -1;
      const c =
        typeof va === "number" && typeof vb === "number"
          ? va - vb
          : String(va).localeCompare(String(vb));
      return c * sort.dir;
    });
  }, [rows, q, need, weights, sort, showEst]);

  const toggle = (f: FeatureId) =>
    setNeed((n) => (n.includes(f) ? n.filter((x) => x !== f) : [...n, f]));

  return (
    <div>
      <div className="flex flex-wrap items-end gap-4">
        <label className="flex min-w-0 flex-col gap-1 text-sm">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            Search
          </span>
          <input
            type="search"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="name, lab or attention"
            className="focus-ring w-64 max-w-full rounded border border-neutral-300 bg-white px-2 py-1 dark:border-neutral-700 dark:bg-neutral-950"
          />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            Weights
          </span>
          <select
            value={weights}
            onChange={(e) => setWeights(e.target.value as typeof weights)}
            className="focus-ring rounded border border-neutral-300 bg-white px-2 py-1 dark:border-neutral-700 dark:bg-neutral-950"
          >
            <option value="all">all</option>
            <option value="open">open</option>
            <option value="closed">closed</option>
          </select>
        </label>
        <label className="flex items-center gap-2 rounded border border-dashed border-amber-500 px-2 py-1 text-sm text-amber-900 dark:text-amber-200">
          <input
            type="checkbox"
            checked={showEst}
            onChange={(e) => setShowEst(e.target.checked)}
            className="focus-ring accent-amber-600"
          />
          Show reported estimates for closed models
        </label>
      </div>
      <fieldset className="mt-4">
        <legend className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
          Must have
        </legend>
        <div className="mt-1 flex flex-wrap gap-1.5">
          {FILTER_FEATURES.map((id) => {
            const f = FEATURES.find((x) => x.id === id)!;
            const on = need.includes(id);
            return (
              <button
                key={id}
                type="button"
                aria-pressed={on}
                title={f.description}
                onClick={() => toggle(id)}
                className={`focus-ring rounded-full border px-2.5 py-0.5 text-xs ${
                  on
                    ? "border-accent bg-accent text-accent-fg"
                    : "border-neutral-300 text-neutral-700 hover:border-accent dark:border-neutral-700 dark:text-neutral-300"
                }`}
              >
                {f.label}
              </button>
            );
          })}
        </div>
      </fieldset>
      <p
        className="mt-3 text-sm text-neutral-600 dark:text-neutral-400"
        aria-live="polite"
      >
        {shown.length} of {rows.length} models
      </p>
      <div
        tabIndex={0}
        role="region"
        aria-label="Scrollable content"
        className="mt-2 overflow-x-auto rounded-lg border border-neutral-200 dark:border-neutral-800"
      >
        <table
          className="w-full min-w-[860px] text-sm"
          data-testid="model-table"
        >
          <thead className="bg-neutral-50 dark:bg-neutral-900">
            <tr>
              {COLUMNS.map((c) => (
                <th
                  key={c.key}
                  scope="col"
                  aria-sort={
                    sort.key === c.key
                      ? sort.dir === 1
                        ? "ascending"
                        : "descending"
                      : "none"
                  }
                  className={`p-2 font-semibold ${c.numeric ? "text-right" : "text-left"}`}
                >
                  <button
                    type="button"
                    onClick={() =>
                      setSort((s) => ({
                        key: c.key,
                        dir:
                          s.key === c.key
                            ? (-s.dir as 1 | -1)
                            : c.numeric
                              ? -1
                              : 1,
                      }))
                    }
                    className="focus-ring rounded"
                  >
                    {c.label}
                    {sort.key === c.key ? (sort.dir === 1 ? " ↑" : " ↓") : ""}
                  </button>
                </th>
              ))}
              <th scope="col" className="p-2 text-left font-semibold">
                Attention
              </th>
            </tr>
          </thead>
          <tbody>
            {shown.map((x) => (
              <tr
                key={x.id}
                className="border-t border-neutral-200 align-top dark:border-neutral-800"
              >
                <td className="px-2 py-1.5">
                  <Link
                    href={`/models/${x.id}`}
                    className="focus-ring rounded font-medium text-accent hover:underline dark:text-indigo-300"
                  >
                    {x.name}
                  </Link>
                  {!x.open && (
                    <span className="ml-1 text-xs text-neutral-500 dark:text-neutral-400">
                      closed
                    </span>
                  )}
                </td>
                <td className="px-2 py-1.5">{x.lab}</td>
                <td className="px-2 py-1.5 font-mono text-xs">
                  {x.released ?? "—"}
                </td>
                <td className="px-2 py-1.5 text-right font-mono">
                  {x.total !== null ? (
                    <span title={x.totalSt}>
                      {formatCount(x.total)}
                      {x.totalSt === "modelled" && (
                        <span className="text-neutral-500 dark:text-neutral-400">
                          *
                        </span>
                      )}
                    </span>
                  ) : showEst && x.estimate ? (
                    <span
                      data-estimate
                      className="inline-flex flex-col items-end gap-0.5"
                    >
                      <span className="rounded border border-dashed border-amber-500 bg-amber-50 px-1 text-amber-950 dark:bg-amber-950/40 dark:text-amber-100">
                        ≈{formatCount(x.estimate.v)}
                      </span>
                      <StatusChip st="reported-estimate" />
                    </span>
                  ) : (
                    <StatusChip st="not-disclosed" />
                  )}
                </td>
                <td className="px-2 py-1.5 text-right font-mono">
                  {x.active !== null ? (
                    <span>
                      {formatCount(x.active)}
                      {x.activeSt === "modelled" && (
                        <span className="text-neutral-500 dark:text-neutral-400">
                          *
                        </span>
                      )}
                    </span>
                  ) : (
                    "—"
                  )}
                </td>
                <td className="px-2 py-1.5 text-right font-mono">
                  {x.layers ?? "—"}
                </td>
                <td className="px-2 py-1.5 text-right font-mono">
                  {x.dModel?.toLocaleString("en-GB") ?? "—"}
                </td>
                <td className="px-2 py-1.5 text-right font-mono">
                  {x.kvPerToken !== null ? formatBytes(x.kvPerToken, 3) : "—"}
                </td>
                <td className="px-2 py-1.5 text-right font-mono">
                  {formatTokens(x.context)}
                </td>
                <td className="px-2 py-1.5 text-xs">{x.attention}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
        * modelled from the configuration where the lab states no figure. KV /
        token: bytes added per token in BF16 by the layers whose cache grows
        with context (windowed and recurrent layers excluded).
      </p>
    </div>
  );
}
