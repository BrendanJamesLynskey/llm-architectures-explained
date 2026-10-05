"use client";

/**
 * Chapter 5: depth against width. A Llama-style decoder of any depth and
 * width (depthWidthSpec, checked against reference/chapter_model.py),
 * costed by the cost model next to a few fixed shapes: parameters, the
 * embedding's share, the KV cache per token, and how much of a decode
 * step's FLOPs go to attention at the chosen context.
 */
import { useState } from "react";

import { Slider } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import { ScrollBox } from "@/components/viz/Legend";
import { decodeFlops, kvCache, params, type Spec } from "@/lib/arch/costModel";
import {
  formatBytes,
  formatCount,
  formatFlops,
  formatTokens,
} from "@/lib/arch/format";
import { depthWidthSpec } from "@/lib/chapters/model";

const FIXED: { label: string; layers: number; d: number }[] = [
  { label: "Deep and thin", layers: 80, d: 2048 },
  { label: "Llama 3 8B's shape", layers: 32, d: 4096 },
  { label: "Shallow and wide", layers: 12, d: 8192 },
];

function row(label: string, spec: Spec, ctx: number) {
  const p = params(spec);
  const kv = kvCache(spec, ctx, 2, 2);
  const f = decodeFlops(spec, ctx);
  const attn = f - 2 * p.matmul_active;
  return {
    label,
    layers: spec.layout[0]!.n,
    d: spec.d_model,
    total: p.total,
    nonEmb: p.non_embedding_total,
    embShare: (p.embedding + p.lm_head) / p.total,
    kvTok: kv.bytes_per_token_unbounded,
    kv: kv.total_bytes,
    flops: f,
    attnShare: attn / f,
  };
}

export default function DepthWidthWidget(): JSX.Element {
  const [layers, setLayers] = useState(48);
  const [k, setK] = useState(24);
  const [log2ctx, setLog2ctx] = useState(15);
  const ctx = 2 ** log2ctx;
  const d = 128 * k;
  const rows = [
    row("Yours", depthWidthSpec(layers, d), ctx),
    ...FIXED.map((x) => row(x.label, depthWidthSpec(x.layers, x.d), ctx)),
  ];

  return (
    <WidgetFrame
      testId="depth-width-widget"
      title="Trade layers for width"
      caption="A Llama-style decoder: 128-wide heads, 8 KV heads, a gated FFN 3.5× the width, a 128,256-token vocabulary. BF16 KV, batch 1."
    >
      <div className="grid items-end gap-4 sm:grid-cols-3">
        <Slider
          label="Layers"
          value={layers}
          min={4}
          max={128}
          onChange={setLayers}
        />
        <Slider
          label="Width (d_model)"
          value={k}
          min={4}
          max={128}
          onChange={setK}
          format={(v) => String(128 * v)}
        />
        <Slider
          label="Context length"
          value={log2ctx}
          min={10}
          max={20}
          onChange={setLog2ctx}
          format={(v) => `${formatTokens(2 ** v)} tokens`}
        />
      </div>
      <ScrollBox label="Depth and width table">
        <table
          className="w-full min-w-[640px] text-sm"
          data-testid="depth-width-table"
        >
          <thead className="bg-neutral-100 text-left dark:bg-neutral-800">
            <tr>
              <th scope="col" className="p-2">
                Shape
              </th>
              <th scope="col" className="p-2 text-right">
                Layers × width
              </th>
              <th scope="col" className="p-2 text-right">
                Parameters
              </th>
              <th scope="col" className="p-2 text-right">
                Non-embedding
              </th>
              <th scope="col" className="p-2 text-right">
                Embeddings
              </th>
              <th scope="col" className="p-2 text-right">
                KV / token
              </th>
              <th scope="col" className="p-2 text-right">
                KV at {formatTokens(ctx)}
              </th>
              <th scope="col" className="p-2 text-right">
                Attention share of decode
              </th>
            </tr>
          </thead>
          <tbody className="whitespace-nowrap font-mono text-xs">
            {rows.map((r) => (
              <tr
                key={r.label}
                className="border-t border-neutral-200 dark:border-neutral-800"
              >
                <th
                  scope="row"
                  className="p-2 text-left font-sans text-sm font-medium"
                >
                  {r.label}
                </th>
                <td className="p-2 text-right">
                  {r.layers} × {r.d}
                </td>
                <td className="p-2 text-right">{formatCount(r.total)}</td>
                <td className="p-2 text-right">{formatCount(r.nonEmb)}</td>
                <td className="p-2 text-right">
                  {(100 * r.embShare).toFixed(1)}%
                </td>
                <td className="p-2 text-right">{formatBytes(r.kvTok)}</td>
                <td className="p-2 text-right">{formatBytes(r.kv)}</td>
                <td className="p-2 text-right">
                  {(100 * r.attnShare).toFixed(1)}% of {formatFlops(r.flops)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollBox>
      <p className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
        With the KV-head count fixed, the cache per token grows with depth and
        not with width, so of two models with the same parameters the deeper one
        caches more. The embedding tables grow with width only, so they are a
        large share of a small, wide model.
      </p>
    </WidgetFrame>
  );
}
