"use client";

/**
 * Chapter 6: what long context costs real architectures. Seven models from
 * the data set, one per long-context technique, costed by the cost model
 * from 1K to 1M tokens: the cache per sequence and the decode FLOPs per
 * token, and at the chosen length how much of a decode step is attention.
 */
import { useState } from "react";

import { Segmented, Slider } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import { Legend, ScrollBox } from "@/components/viz/Legend";
import {
  binaryTicks,
  LineChart,
  PALETTE,
} from "@/components/viz/LineChart";
import { decodeFlops, kvCache, params } from "@/lib/arch/costModel";
import { formatBytes, formatFlops, formatTokens } from "@/lib/arch/format";
import { CHAPTERS } from "@/lib/chapters/data";

import { useSpecs } from "./useSpecs";

const TECHNIQUE: Record<string, string> = {
  "llama-3.1-405b": "GQA, every layer global",
  "gemma-3-27b": "5 sliding-window layers to 1 global",
  "deepseek-v3": "MLA latent cache",
  "deepseek-v3.2": "MLA + sparse attention (indexer)",
  "qwen3-next-80b-a3b": "Gated DeltaNet 3:1 hybrid",
  "minimax-text-01": "Lightning (linear) 7:1 hybrid",
  "deepseek-v4-flash": "Compressed + sparse attention",
};

const METRIC = [
  { value: "kv", label: "Cache per sequence" },
  { value: "flops", label: "Decode FLOPs per token" },
] as const;

export default function ContextWidget(): JSX.Element {
  const { models, error } = useSpecs();
  const [log2ctx, setLog2ctx] = useState(17);
  const [metric, setMetric] = useState<(typeof METRIC)[number]["value"]>("kv");
  const ctx = 2 ** log2ctx;
  if (error) return <p role="alert">Could not load the model data: {error}</p>;
  if (!models)
    return (
      <p
        data-pending-widget
        className="text-sm text-neutral-600 dark:text-neutral-400"
      >
        Loading the model data…
      </p>
    );
  const chosen = CHAPTERS.context
    .map((id) => models.find((m) => m.id === id)!)
    .filter((m) => m.spec);
  const xs = Array.from({ length: 11 }, (_, k) => 2 ** (10 + k));
  const series = chosen.map((m, i) => ({
    id: m.id,
    label: m.name,
    colour: PALETTE[i]!,
    points: xs.map(
      (c) =>
        [
          c,
          metric === "kv"
            ? kvCache(m.spec!, c, 2, 2).total_bytes
            : decodeFlops(m.spec!, c),
        ] as [number, number],
    ),
  }));

  return (
    <WidgetFrame
      testId="context-widget"
      title="Seven ways to reach long context"
      caption="Real models from the data set, costed by this site's cost model: BF16 cache, batch 1. Scaling RoPE (chapter 2) changes none of these numbers; the attention design does."
    >
      <div className="grid items-end gap-4 sm:grid-cols-2">
        <Slider
          label="Context length"
          value={log2ctx}
          min={10}
          max={20}
          onChange={setLog2ctx}
          format={(v) => `${formatTokens(2 ** v)} tokens`}
        />
        <Segmented
          label="Chart"
          value={metric}
          options={METRIC}
          onChange={setMetric}
        />
      </div>
      <ScrollBox label="Long-context models table">
        <table
          className="w-full min-w-[640px] text-sm"
          data-testid="context-table"
        >
          <thead className="bg-neutral-100 text-left dark:bg-neutral-800">
            <tr>
              <th scope="col" className="p-2">
                Model
              </th>
              <th scope="col" className="p-2">
                Technique
              </th>
              <th scope="col" className="p-2 text-right">
                Cache at {formatTokens(ctx)}
              </th>
              <th scope="col" className="p-2 text-right">
                Decode FLOPs / token
              </th>
              <th scope="col" className="p-2 text-right">
                Attention and state share
              </th>
            </tr>
          </thead>
          <tbody className="text-xs">
            {chosen.map((m, i) => {
              const f = decodeFlops(m.spec!, ctx);
              const share = 1 - (2 * params(m.spec!).matmul_active) / f;
              return (
                <tr
                  key={m.id}
                  className="border-t border-neutral-200 dark:border-neutral-800"
                >
                  <th scope="row" className="p-2 text-left text-sm font-medium">
                    <span
                      aria-hidden
                      className="mr-1.5 inline-block size-2.5 rounded-sm"
                      style={{ background: PALETTE[i] }}
                    />
                    <a
                      href={`/models/${m.id}`}
                      className="focus-ring rounded hover:text-accent hover:underline dark:hover:text-indigo-300"
                    >
                      {m.name}
                    </a>
                  </th>
                  <td className="p-2">{TECHNIQUE[m.id]}</td>
                  <td className="p-2 text-right font-mono">
                    {formatBytes(kvCache(m.spec!, ctx, 2, 2).total_bytes)}
                  </td>
                  <td className="p-2 text-right font-mono">{formatFlops(f)}</td>
                  <td className="p-2 text-right font-mono">
                    {(100 * share).toFixed(1)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </ScrollBox>
      <ScrollBox label="Long-context chart">
        <LineChart
          series={series}
          title={
            metric === "kv"
              ? "Cache per sequence against context length"
              : "Decode FLOPs per token against context length"
          }
          xLabel="Context (tokens)"
          yLabel={metric === "kv" ? "Cache per sequence" : "FLOPs per token"}
          formatX={formatTokens}
          formatY={(v) =>
            metric === "kv" ? formatBytes(v, 3) : formatFlops(v, 2)
          }
          yTicks={metric === "kv" ? binaryTicks : undefined}
        />
      </ScrollBox>
      <Legend series={series} />
    </WidgetFrame>
  );
}
