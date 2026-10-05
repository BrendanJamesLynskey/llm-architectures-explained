"use client";

/**
 * Chapter 1: the attention family on one body. Eight token mixers built on
 * Llama 3 8B's sourced shape (scripts/make_chapter_fixtures.py), costed live
 * by the TypeScript cost model at the chosen context length and KV
 * precision: cache per token and in total, decode FLOPs and bytes, and the
 * parameter count, plus the cache against context for each variant.
 */
import { useMemo, useState } from "react";

import { Segmented, Slider } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import { Legend, ScrollBox } from "@/components/viz/Legend";
import { binaryTicks, LineChart, PALETTE } from "@/components/viz/LineChart";
import {
  decodeBytes,
  decodeFlops,
  kvCache,
  params,
} from "@/lib/arch/costModel";
import {
  formatBytes,
  formatCount,
  formatFlops,
  formatTokens,
} from "@/lib/arch/format";
import { CHAPTERS } from "@/lib/chapters/data";

const KV = [
  { value: "2", label: "BF16" },
  { value: "1", label: "FP8" },
] as const;

export default function AttentionWidget(): JSX.Element {
  const [log2ctx, setLog2ctx] = useState(15);
  const [kb, setKb] = useState<(typeof KV)[number]["value"]>("2");
  const ctx = 2 ** log2ctx;
  const bytes = Number(kb);
  const variants = CHAPTERS.attention.variants;

  const rows = useMemo(() => {
    const gqa = kvCache(variants[1]!.spec, ctx, bytes, bytes).total_bytes;
    return variants.map((v, i) => {
      const kv = kvCache(v.spec, ctx, bytes, bytes);
      return {
        ...v,
        colour: PALETTE[i]!,
        total: params(v.spec).total,
        kv: kv.total_bytes,
        perToken: kv.bytes_per_token_unbounded,
        vsGqa: kv.total_bytes / gqa,
        flops: decodeFlops(v.spec, ctx),
        read: decodeBytes(v.spec, ctx, 2, bytes).total,
      };
    });
  }, [variants, ctx, bytes]);

  const series = rows.map((r) => ({
    id: r.id,
    label: r.label,
    colour: r.colour,
    points: Array.from({ length: 11 }, (_, k) => 2 ** (10 + k)).map(
      (c) =>
        [c, kvCache(r.spec, c, bytes, bytes).total_bytes] as [number, number],
    ),
  }));

  return (
    <WidgetFrame
      testId="attention-widget"
      title="Eight token mixers on one Llama 3 8B body"
      caption="Same width, depth, FFN and vocabulary; only the attention changes. Costs from this site's cost model (checked against its Python reference), batch 1, BF16 weights."
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
        <Segmented label="KV cache" value={kb} options={KV} onChange={setKb} />
      </div>
      <ScrollBox label="Attention variants table">
        <table
          className="w-full min-w-[640px] text-sm"
          data-testid="attention-table"
        >
          <thead className="bg-neutral-100 text-left dark:bg-neutral-800">
            <tr>
              <th scope="col" className="p-2">
                Mixer
              </th>
              <th scope="col" className="p-2 text-right">
                Parameters
              </th>
              <th scope="col" className="p-2 text-right">
                Growing KV / token
              </th>
              <th scope="col" className="p-2 text-right">
                Cache at {formatTokens(ctx)}
              </th>
              <th scope="col" className="p-2 text-right">
                vs GQA
              </th>
              <th scope="col" className="p-2 text-right">
                Decode FLOPs / token
              </th>
              <th scope="col" className="p-2 text-right">
                Bytes / decode step
              </th>
            </tr>
          </thead>
          <tbody className="font-mono text-xs">
            {rows.map((r) => (
              <tr
                key={r.id}
                data-variant={r.id}
                className="border-t border-neutral-200 dark:border-neutral-800"
              >
                <th
                  scope="row"
                  className="p-2 text-left font-sans text-sm font-medium"
                >
                  <span
                    aria-hidden
                    className="mr-1.5 inline-block size-2.5 rounded-sm"
                    style={{ background: r.colour }}
                  />
                  {r.label}
                </th>
                <td className="p-2 text-right">{formatCount(r.total)}</td>
                <td className="p-2 text-right">{formatBytes(r.perToken)}</td>
                <td className="p-2 text-right" data-cell="kv">
                  {formatBytes(r.kv)}
                </td>
                <td className="p-2 text-right">{r.vsGqa.toFixed(2)}×</td>
                <td className="p-2 text-right">{formatFlops(r.flops)}</td>
                <td className="p-2 text-right">{formatBytes(r.read)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollBox>
      <ScrollBox label="Cache against context chart">
        <LineChart
          series={series}
          title="KV cache and recurrent state against context length"
          xLabel="Context (tokens)"
          yLabel="Cache per sequence"
          formatX={formatTokens}
          formatY={(v) => formatBytes(v, 3)}
          yTicks={binaryTicks}
        />
      </ScrollBox>
      <Legend series={series} />
      <p className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
        &ldquo;Growing KV / token&rdquo; counts only the layers whose cache
        grows with every token; window, linear and state-space layers hold a
        fixed amount instead, which the cache total includes. The sparse variant
        stores an extra indexer key per token; its attention then reads only the
        2,048 cached tokens the indexer picks, while the indexer itself still
        scores every cached token with its smaller heads.
      </p>
    </WidgetFrame>
  );
}
