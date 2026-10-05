"use client";

/**
 * Compare two to four models side by side, and calculate their costs at a
 * chosen context length and precision: diagrams, KV cache against context,
 * and prefill and decode cost. Runs the TypeScript cost model (checked
 * against the Python reference) on the compact data in /data/specs.json.
 *
 * Reported estimates for closed models are off by default. Switched on,
 * they appear only in the rows they can fill (parameters, weight memory,
 * and FLOPs under a stated dense assumption), dashed, amber and labelled.
 */
import { useEffect, useMemo, useState } from "react";

import { Segmented, Slider } from "@/components/ui/Controls";
import { StatusChip } from "@/components/ui/Provenance";
import { BlockDiagram } from "@/components/viz/BlockDiagram";
import { LineChart, SERIES_COLOURS } from "@/components/viz/LineChart";
import {
  decodeBytes,
  decodeFlops,
  kvCache,
  params,
  prefillFlops,
} from "@/lib/arch/costModel";
import {
  formatBytes,
  formatCount,
  formatFlops,
  formatTokens,
} from "@/lib/arch/format";
import type { CompactModel } from "@/lib/arch/types";

const DEFAULT = [
  "llama-3-8b",
  "qwen3-next-80b-a3b",
  "deepseek-v3",
  "gemma-3-27b",
];
const WEIGHT_BYTES = [
  { value: "2", label: "BF16" },
  { value: "1", label: "FP8" },
  { value: "0.5", label: "4-bit" },
] as const;
const KV_BYTES = [
  { value: "2", label: "BF16" },
  { value: "1", label: "FP8" },
] as const;

type Cell = { text: string; est?: boolean; note?: string };

function readSelection(): string[] {
  if (typeof window === "undefined") return DEFAULT.slice(0, 3);
  const m = new URLSearchParams(window.location.search).get("m");
  const ids = m ? m.split(",").filter(Boolean).slice(0, 4) : [];
  return ids.length ? ids : DEFAULT.slice(0, 3);
}

export default function CompareTool(): JSX.Element {
  const [all, setAll] = useState<CompactModel[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ids, setIds] = useState<string[]>(DEFAULT.slice(0, 3));
  const [log2ctx, setLog2ctx] = useState(15);
  const [wb, setWb] = useState<(typeof WEIGHT_BYTES)[number]["value"]>("2");
  const [kb, setKb] = useState<(typeof KV_BYTES)[number]["value"]>("2");
  const [est, setEst] = useState(false);

  useEffect(() => {
    setIds(readSelection());
    fetch("/data/specs.json")
      .then((r) =>
        r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)),
      )
      .then((d: CompactModel[]) => setAll(d))
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    const url = new URL(window.location.href);
    url.searchParams.set("m", ids.join(","));
    window.history.replaceState(null, "", url);
  }, [ids]);

  const chosen = useMemo(
    () =>
      all
        ? ids
            .map((id) => all.find((m) => m.id === id))
            .filter((m): m is CompactModel => Boolean(m))
        : [],
    [all, ids],
  );
  const ctx = 2 ** log2ctx;
  const wBytes = Number(wb);
  const kvBytes = Number(kb);

  if (error)
    return (
      <p role="alert" className="text-sm text-red-700 dark:text-red-300">
        Could not load the model data: {error}
      </p>
    );
  if (!all)
    return (
      <p className="text-sm text-neutral-600 dark:text-neutral-400">
        Loading the model data…
      </p>
    );

  const sorted = [...all].sort((a, b) => a.name.localeCompare(b.name));
  const setAt = (i: number, id: string) =>
    setIds((cur) => {
      const next = [...cur];
      if (id === "") next.splice(i, 1);
      else next[i] = id;
      return next;
    });

  const rows: { label: string; cells: Cell[] }[] = [];
  const cell = (
    m: CompactModel,
    f: (s: NonNullable<CompactModel["spec"]>) => string,
    estimate?: () => Cell | null,
  ): Cell => {
    if (m.spec) return { text: f(m.spec) };
    if (est && estimate) {
      const e = estimate();
      if (e) return e;
    }
    return { text: "not disclosed" };
  };
  const totalEst = (m: CompactModel) =>
    m.estimates.find((e) => e.field === "total_params");
  rows.push({
    label: "Total parameters",
    cells: chosen.map((m) =>
      m.total.v !== null
        ? { text: formatCount(m.total.v) }
        : m.spec
          ? { text: `${formatCount(params(m.spec).total)} (modelled)` }
          : est && totalEst(m)
            ? {
                text: `≈${formatCount(totalEst(m)!.v)}`,
                est: true,
                note: `${totalEst(m)!.publisher}, ${totalEst(m)!.date}`,
              }
            : { text: "not disclosed" },
    ),
  });
  rows.push({
    label: "Active parameters",
    cells: chosen.map((m) =>
      m.active.v !== null
        ? { text: formatCount(m.active.v) }
        : cell(m, (s) => `${formatCount(params(s).active)} (modelled)`),
    ),
  });
  rows.push({
    label: `Weights in memory (${WEIGHT_BYTES.find((x) => x.value === wb)!.label})`,
    cells: chosen.map((m) =>
      cell(
        m,
        (s) => formatBytes(params(s).total * wBytes),
        () => {
          const e = totalEst(m);
          return e
            ? {
                text: `≈${formatBytes(e.v * wBytes)}`,
                est: true,
                note: "reported estimate × bytes per weight",
              }
            : null;
        },
      ),
    ),
  });
  rows.push({
    label: `KV cache + state at ${formatTokens(ctx)} tokens`,
    cells: chosen.map((m) =>
      cell(m, (s) => formatBytes(kvCache(s, ctx, kvBytes).total_bytes)),
    ),
  });
  rows.push({
    label: "Decode FLOPs per token",
    cells: chosen.map((m) =>
      cell(
        m,
        (s) => formatFlops(decodeFlops(s, ctx)),
        () => {
          const e = totalEst(m);
          return e
            ? {
                text: `≈${formatFlops(2 * e.v)}`,
                est: true,
                note: "2 × the reported estimate; assumes every weight is used (dense), and ignores attention",
              }
            : null;
        },
      ),
    ),
  });
  rows.push({
    label: `Prefill FLOPs for a ${formatTokens(ctx)}-token prompt`,
    cells: chosen.map((m) => cell(m, (s) => formatFlops(prefillFlops(s, ctx)))),
  });
  rows.push({
    label: "Bytes read per decode step (batch 1)",
    cells: chosen.map((m) =>
      cell(m, (s) => formatBytes(decodeBytes(s, ctx, wBytes, kvBytes).total)),
    ),
  });
  rows.push({
    label: "Decode arithmetic intensity (FLOPs per byte)",
    cells: chosen.map((m) =>
      cell(m, (s) =>
        (
          decodeFlops(s, ctx) / decodeBytes(s, ctx, wBytes, kvBytes).total
        ).toFixed(2),
      ),
    ),
  });

  const series = chosen
    .map((m, i) =>
      m.spec
        ? {
            id: m.id,
            label: m.name,
            colour: SERIES_COLOURS[i]!,
            points: Array.from({ length: 11 }, (_, k) => 2 ** (10 + k)).map(
              (c) =>
                [c, kvCache(m.spec!, c, kvBytes).total_bytes] as [
                  number,
                  number,
                ],
            ),
          }
        : null,
    )
    .filter((s): s is NonNullable<typeof s> => s !== null);

  return (
    <div data-testid="compare-tool">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <label key={i} className="flex min-w-0 flex-col gap-1 text-sm">
            <span className="flex items-center gap-2 font-medium text-neutral-700 dark:text-neutral-300">
              <span
                aria-hidden
                className="inline-block size-3 rounded-sm"
                style={{ background: SERIES_COLOURS[i] }}
              />
              Model {i + 1}
            </span>
            <select
              value={ids[i] ?? ""}
              onChange={(e) => setAt(i, e.target.value)}
              disabled={i > ids.length}
              className="focus-ring w-full min-w-0 rounded border border-neutral-300 bg-white px-2 py-1 dark:border-neutral-700 dark:bg-neutral-950"
            >
              <option value="">{i < 2 ? "choose…" : "(none)"}</option>
              {sorted.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                  {m.spec ? "" : " (no dimensions)"}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>
      <div className="mt-4 grid items-end gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Slider
          label="Context length"
          value={log2ctx}
          min={10}
          max={20}
          onChange={setLog2ctx}
          format={(v) => `${formatTokens(2 ** v)} tokens`}
        />
        <Segmented
          label="Weights"
          value={wb}
          options={WEIGHT_BYTES}
          onChange={setWb}
        />
        <Segmented
          label="KV cache"
          value={kb}
          options={KV_BYTES}
          onChange={setKb}
        />
        <label className="flex items-center gap-2 self-center rounded border border-dashed border-amber-500 px-2 py-1 text-sm text-amber-900 dark:text-amber-200">
          <input
            type="checkbox"
            checked={est}
            onChange={(e) => setEst(e.target.checked)}
            className="focus-ring accent-amber-600"
          />
          Include reported estimates
        </label>
      </div>

      <div
        tabIndex={0}
        role="region"
        aria-label="Scrollable content"
        className="mt-6 overflow-x-auto rounded-lg border border-neutral-200 dark:border-neutral-800"
      >
        <table
          className="w-full min-w-[640px] text-sm"
          data-testid="compare-table"
        >
          <thead className="bg-neutral-50 dark:bg-neutral-900">
            <tr>
              <th scope="col" className="p-2 text-left font-semibold">
                At {formatTokens(ctx)} tokens, batch 1
              </th>
              {chosen.map((m, i) => (
                <th
                  key={m.id}
                  scope="col"
                  className="p-2 text-right font-semibold"
                >
                  <span
                    aria-hidden
                    className="mr-1 inline-block size-2.5 rounded-sm"
                    style={{ background: SERIES_COLOURS[i] }}
                  />
                  <a
                    href={`/models/${m.id}`}
                    className="focus-ring rounded text-accent hover:underline dark:text-indigo-300"
                  >
                    {m.name}
                  </a>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.label}
                className="border-t border-neutral-200 dark:border-neutral-800"
              >
                <th
                  scope="row"
                  className="px-2 py-1.5 text-left font-normal text-neutral-700 dark:text-neutral-300"
                >
                  {r.label}
                </th>
                {r.cells.map((c, i) => (
                  <td key={i} className="px-2 py-1.5 text-right font-mono">
                    {c.est ? (
                      <span
                        data-estimate
                        className="inline-flex flex-col items-end gap-0.5"
                      >
                        <span className="rounded border border-dashed border-amber-500 bg-amber-50 px-1 text-amber-950 dark:bg-amber-950/40 dark:text-amber-100">
                          {c.text}
                        </span>
                        <StatusChip st="reported-estimate" />
                        <span className="max-w-56 text-right font-sans text-[0.7rem] text-neutral-600 dark:text-neutral-400">
                          {c.note}
                        </span>
                      </span>
                    ) : c.text === "not disclosed" ? (
                      <StatusChip st="not-disclosed" />
                    ) : (
                      c.text
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2 className="mt-8 text-lg font-semibold">KV cache against context</h2>
      <div
        tabIndex={0}
        role="region"
        aria-label="Scrollable content"
        className="mt-2 overflow-x-auto rounded-lg border border-neutral-200 p-2 dark:border-neutral-800"
      >
        <LineChart
          series={series}
          xLabel="context (tokens)"
          yLabel={`KV cache + state (${KV_BYTES.find((x) => x.value === kb)!.label})`}
          formatX={formatTokens}
          formatY={(v) => formatBytes(v, 2)}
          title="KV cache bytes against context length for the chosen models"
        />
      </div>
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 rounded bg-neutral-50 px-3 py-2 text-xs dark:bg-neutral-900">
        {series.map((s) => (
          <li key={s.id} className="flex items-center gap-1.5">
            <span
              aria-hidden
              className="inline-block h-0.5 w-4"
              style={{ background: s.colour }}
            />
            {s.label}
          </li>
        ))}
      </ul>

      <h2 className="mt-8 text-lg font-semibold">Architectures</h2>
      <div className="mt-2 space-y-6">
        {chosen.map((m) =>
          m.spec ? (
            <figure
              tabIndex={0}
              key={m.id}
              className="overflow-x-auto rounded-lg border border-neutral-200 bg-white p-3 dark:border-neutral-800 dark:bg-neutral-950"
            >
              <figcaption className="mb-2 text-sm font-medium">
                {m.name}
              </figcaption>
              <BlockDiagram
                spec={m.spec}
                placement={m.norm ?? "none"}
                title={`${m.name}: layer stack and blocks`}
              />
            </figure>
          ) : (
            <p
              key={m.id}
              className="text-sm text-neutral-700 dark:text-neutral-300"
            >
              <strong>{m.name}</strong>: <StatusChip st="not-disclosed" />{" "}
              {m.lab} has not published the dimensions.
            </p>
          ),
        )}
      </div>
    </div>
  );
}
