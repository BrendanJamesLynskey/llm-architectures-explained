"use client";

/**
 * Chapter 9: the causal encoder-decoder (CED) in Disaggregated_Inference_Sim,
 * live. The simulator's own JavaScript engine, vendored byte for byte at
 * brief 11's commit, (a) costs one prefill step for the decoder-only proxy
 * and both CED variants, and (b) re-runs the capacity search of results.md
 * sections 17 and 18 for every prefill/decode pool split, in a Web Worker,
 * checking each rate against the recorded one (a ✓ when they are equal to
 * the last bit).
 *
 * Illustrative: a dense Llama-3-70B shape split 40 + 40 stands in for
 * DeepSeek-V4.1-Flash; the roofline and H100 coefficients are the
 * simulator's.
 */
import { useEffect, useRef, useState } from "react";

import { ActionButton, Segmented, Slider } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import { ScrollBox } from "@/components/viz/Legend";
import { formatTokens } from "@/lib/arch/format";
import {
  CLUSTER,
  SMALL_PREFILL,
  VARIANTS,
  cedConfig,
  f2,
  prefillStep,
  type VariantKey,
} from "@/lib/disagg/ced";
import {
  maxSustainableRate,
  type DrawRow,
  type SimConfig,
} from "@/lib/disagg/engine";
import vendored from "@/lib/disagg/vendor/VENDORED.json";

type Recorded = {
  commit: string;
  s16: {
    steps: Record<
      string,
      Record<"base" | "ced" | "enc", { pflop: string; ms: string }>
    >;
  };
  s17: {
    workload: string;
    prompt: number;
    output: number;
    ttft_slo: number;
    slo_text: string;
    rows: Record<
      string,
      {
        line: string;
        best: number;
        cells: Record<string, { rate: number; text: string }>;
      }
    >;
  }[];
  s18: {
    workload: string;
    prompt: number;
    output: number;
    ttft_slo: number;
    rows: Record<
      string,
      { line: string; cells: Record<string, { rate: number; text: string }> }
    >;
  };
};

const ms = (x: number): string =>
  x < 1
    ? `${(1e3 * x).toFixed(1)} ms`
    : `${Math.round(1e3 * x).toLocaleString("en-GB")} ms`;

const STEP_KEYS = [
  { k: "base", label: "Decoder-only" },
  { k: "ced", label: "CED, replay on prefill" },
  { k: "enc", label: "CED encoder only (replay on decode)" },
] as const;

type Live = Record<string, { rate: number; simulations: number }>;

let worker: Worker | null | undefined;
function getWorker(): Worker | null {
  if (worker !== undefined) return worker;
  try {
    worker = new Worker(new URL("./ced/sim.worker.ts", import.meta.url));
  } catch {
    worker = null;
  }
  return worker;
}

export default function CedSimulatorWidget(): JSX.Element {
  const [rec, setRec] = useState<Recorded | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [log2p, setLog2p] = useState(13);
  const [table, setTable] = useState("8192:128");
  const [live, setLive] = useState<Live>({});
  const [running, setRunning] = useState(false);
  const reqId = useRef(0);

  useEffect(() => {
    fetch("/disagg/ced-results.json")
      .then((r) =>
        r.ok
          ? (r.json() as Promise<Recorded>)
          : Promise.reject(new Error(`HTTP ${r.status}`)),
      )
      .then(setRec)
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    setLive({});
    setRunning(false);
    reqId.current++;
  }, [table]);

  if (error)
    return <p role="alert">Could not load the recorded results: {error}</p>;
  if (!rec)
    return (
      <p
        data-pending-widget
        className="text-sm text-neutral-600 dark:text-neutral-400"
      >
        Loading the simulator…
      </p>
    );

  // ── (a) one prefill step ─────────────────────────────────────────────
  const prompt = 2 ** log2p;
  const recStep = rec.s16.steps[String(prompt)];
  const steps = STEP_KEYS.map((s) => {
    const c = prefillStep(s.k, prompt);
    const pflop = (c.flops / 1e15).toFixed(3);
    const t = ms(c.time);
    const ok = recStep
      ? recStep[s.k].pflop === pflop && recStep[s.k].ms === t
      : null;
    return { ...s, pflop, t, time: c.time, ok };
  });

  // ── (b) the pool-split tables ───────────────────────────────────────
  const small = table === "s18";
  const w = small ? null : rec.s17.find((x) => x.workload === table)!;
  const prompt2 = small ? rec.s18.prompt : w!.prompt;
  const output2 = small ? rec.s18.output : w!.output;
  const slo = small ? rec.s18.ttft_slo : w!.ttft_slo;
  const splits: { key: string; label: string; np: number; nd: number }[] = small
    ? SMALL_PREFILL.map(([np, nd]) => ({
        key: `${np}x2+${nd}x4`,
        label: `${np}×2 + ${nd}×4`,
        np,
        nd,
      }))
    : Array.from({ length: CLUSTER - 1 }, (_, i) => ({
        key: String(i + 1),
        label: `${i + 1}P${CLUSTER - 1 - i}D`,
        np: i + 1,
        nd: CLUSTER - 1 - i,
      }));
  // Section 18's decoder-only row is limited by the simulator's prefill
  // router, not by compute (results.md says so), so it is not shown.
  const variants = VARIANTS.filter((v) => !(small && v.key === "decoder-only"));
  const rows = small ? rec.s18.rows : w!.rows;
  const cells: { key: string; cfg: SimConfig }[] = variants.flatMap((v) =>
    splits.map((s) => ({
      key: `${v.key}|${s.key}`,
      cfg: cedConfig(v.key, s.np, s.nd, slo, small ? 2 : undefined),
    })),
  );
  const done = Object.keys(live).length;
  const sims = Object.values(live).reduce((n, x) => n + x.simulations, 0);
  const matched = Object.entries(live).filter(([k, x]) => {
    const [v, s] = k.split("|") as [VariantKey, string];
    return rows[v]!.cells[s]!.rate === x.rate;
  }).length;

  async function measure(): Promise<void> {
    const id = ++reqId.current;
    setLive({});
    setRunning(true);
    const res = await fetch(`/disagg/workloads/ced-${prompt2}-${output2}.json`);
    const draws = ((await res.json()) as { rows: DrawRow[] }).rows;
    if (id !== reqId.current) return;
    const wk = getWorker();
    if (wk) {
      wk.onmessage = (
        e: MessageEvent<{
          id: number;
          key?: string;
          rate?: number;
          simulations?: number;
          done?: boolean;
          error?: string;
        }>,
      ) => {
        if (e.data.id !== id || id !== reqId.current) return;
        if (e.data.error) {
          setError(e.data.error);
          setRunning(false);
        } else if (e.data.done) setRunning(false);
        else
          setLive((old) => ({
            ...old,
            [e.data.key!]: {
              rate: e.data.rate!,
              simulations: e.data.simulations!,
            },
          }));
      };
      wk.postMessage({ id, prompt: prompt2, output: output2, draws, cells });
      return;
    }
    // no workers: run on the main thread, one cell per task
    for (const c of cells) {
      await new Promise((r) => setTimeout(r, 0));
      if (id !== reqId.current) return;
      const r = maxSustainableRate(c.cfg, draws, prompt2, output2);
      setLive((old) => ({
        ...old,
        [c.key]: { rate: r.rate, simulations: r.simulations },
      }));
    }
    setRunning(false);
  }

  const bestOf = (v: string): number => {
    const r = rows[v]!.cells;
    return Math.max(...Object.values(r).map((x) => x.rate));
  };
  const baseBest = small ? null : bestOf("decoder-only");

  return (
    <WidgetFrame
      testId="ced-simulator"
      title="The causal encoder-decoder in the disaggregated simulator"
      caption={
        <>
          Disaggregated_Inference_Sim&apos;s own JavaScript engine, vendored at
          commit <code>{vendored.commit.slice(0, 7)}</code> and tested against
          the Python package. Illustrative: a dense Llama-3-70B shape split 40 +
          40, on 4×H100 instances, stands in for DeepSeek-V4.1-Flash.
        </>
      }
    >
      <h3 className="text-sm font-semibold">
        One prefill step, one prompt, 4×H100
      </h3>
      <div className="mt-2 max-w-md">
        <Slider
          label="Prompt length"
          value={log2p}
          min={10}
          max={15}
          onChange={setLog2p}
          format={(v) => `${formatTokens(2 ** v)} tokens`}
        />
      </div>
      <ScrollBox label="Prefill step table">
        <table className="w-full min-w-[560px] text-sm" data-testid="ced-steps">
          <thead className="bg-neutral-100 text-left dark:bg-neutral-800">
            <tr>
              <th scope="col" className="p-2">
                Model
              </th>
              <th scope="col" className="p-2 text-right">
                PFLOP
              </th>
              <th scope="col" className="p-2 text-right">
                Time
              </th>
              <th scope="col" className="p-2 text-right">
                vs decoder-only
              </th>
              <th scope="col" className="p-2 text-right">
                results.md §16
              </th>
            </tr>
          </thead>
          <tbody className="font-mono text-xs">
            {steps.map((s) => (
              <tr
                key={s.k}
                className="border-t border-neutral-200 dark:border-neutral-800"
              >
                <th
                  scope="row"
                  className="p-2 text-left font-sans text-sm font-medium"
                >
                  {s.label}
                </th>
                <td className="p-2 text-right">{s.pflop}</td>
                <td className="p-2 text-right">{s.t}</td>
                <td className="p-2 text-right">
                  {(s.time / steps[0]!.time).toFixed(3)}
                </td>
                <td
                  className="p-2 text-right"
                  data-step-check={s.ok === null ? "none" : s.ok ? "ok" : "bad"}
                >
                  {s.ok === null
                    ? "not recorded"
                    : s.ok
                      ? "✓ same"
                      : "✗ differs"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollBox>

      <h3 className="mt-6 text-sm font-semibold">
        Pool sizing: the highest rate with 90% of requests inside both SLOs
      </h3>
      <div className="mt-2 flex flex-wrap items-end gap-4">
        <Segmented
          label="Prompt : output (mean)"
          value={table}
          options={[
            ...rec.s17.map((x) => ({
              value: x.workload,
              label: x.workload.replace(":", " : "),
            })),
            { value: "s18", label: "8192 : 128, 2-GPU prefill" },
          ]}
          onChange={setTable}
        />
        <ActionButton onClick={() => void measure()} disabled={running}>
          {running ? "Measuring…" : "Re-measure live"}
        </ActionButton>
      </div>
      <p
        className="mt-2 text-xs text-neutral-600 dark:text-neutral-400"
        aria-live="polite"
        data-testid="ced-progress"
      >
        {small
          ? "24 GPUs: prefill instances on 2 H100s, decode instances on 4. "
          : `6 instances of 4×H100 split between the pools. `}
        TTFT SLO {ms(slo)} (5× the decoder-only unloaded prefill of the mean
        prompt), TPOT SLO 25 ms, 1,000 requests.{" "}
        {done > 0 &&
          `Live: ${done} of ${cells.length} cells, ${sims} simulations, ${matched} identical to the recorded rate.`}
      </p>
      <ScrollBox label="Pool split table">
        <table className="w-full min-w-[640px] text-sm" data-testid="ced-split">
          <thead className="bg-neutral-100 text-left dark:bg-neutral-800">
            <tr>
              <th scope="col" className="p-2">
                Model (req/s)
              </th>
              {splits.map((s) => (
                <th key={s.key} scope="col" className="p-2 text-right">
                  {s.label}
                </th>
              ))}
              {!small && (
                <th scope="col" className="p-2 text-right">
                  Best vs decoder-only
                </th>
              )}
            </tr>
          </thead>
          <tbody className="font-mono text-xs">
            {variants.map((v) => {
              const best = bestOf(v.key);
              return (
                <tr
                  key={v.key}
                  className="border-t border-neutral-200 dark:border-neutral-800"
                >
                  <th
                    scope="row"
                    className="p-2 text-left font-sans text-sm font-medium"
                  >
                    {v.label}
                  </th>
                  {splits.map((s) => {
                    const r = rows[v.key]!.cells[s.key]!;
                    const l = live[`${v.key}|${s.key}`];
                    return (
                      <td
                        key={s.key}
                        data-cell={`${v.key}|${s.key}`}
                        data-recorded={r.text}
                        data-live={l ? f2(l.rate) : undefined}
                        data-match={l ? String(l.rate === r.rate) : undefined}
                        className={`p-2 text-right ${r.rate === best ? "font-semibold text-accent dark:text-indigo-300" : ""}`}
                      >
                        {l
                          ? `${f2(l.rate)} ${l.rate === r.rate ? "✓" : "✗"}`
                          : r.text}
                      </td>
                    );
                  })}
                  {!small && (
                    <td className="p-2 text-right">
                      {(best / baseBest!).toFixed(2)}×
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </ScrollBox>
      <p className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
        Bold: the best split for each model. Before you re-measure, the cells
        show results.md (simulator commit {rec.commit.slice(0, 7)}); after, the
        rates this page found, with ✓ where they equal the recorded ones.
        {small &&
          " The decoder-only row of this table is left out: results.md notes it is limited by the simulator's prefill router, not by compute."}
      </p>
    </WidgetFrame>
  );
}
