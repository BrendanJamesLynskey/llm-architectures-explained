"use client";

/**
 * Chapter 4: dense against mixture of experts. Rebuilds Llama 3 8B with a
 * mixture of experts in place of its FFN (moeSpec, checked against
 * reference/chapter_model.py) and costs it: total and active parameters,
 * the memory the weights occupy and the bytes a batch-1 decode step reads.
 * Presets use the expert counts of real models on this body; the expert
 * widths are this body's FFN divided by the granularity, not theirs.
 */
import { useState } from "react";

import { Segmented, Slider, Stat } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import { decodeBytes, decodeFlops, params } from "@/lib/arch/costModel";
import { formatBytes, formatCount, formatFlops } from "@/lib/arch/format";
import { CHAPTERS } from "@/lib/chapters/data";
import { layerTotal, moeSpec } from "@/lib/chapters/model";

const EXPERTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512];
const GRAN = ["1", "2", "4", "8", "16"] as const;

type Setting = {
  e: number;
  k: number;
  s: number;
  g: (typeof GRAN)[number];
  p: number;
};

const PRESETS: { label: string; v: Setting }[] = [
  { label: "Dense (Llama 3 8B)", v: { e: 1, k: 1, s: 0, g: "1", p: 0 } },
  {
    label: "8 experts, top 2 (Mixtral's counts)",
    v: { e: 8, k: 2, s: 0, g: "1", p: 0 },
  },
  {
    label: "64 fine-grained, top 6 + 2 shared (DeepSeekMoE 16B's counts)",
    v: { e: 64, k: 6, s: 2, g: "8", p: 1 },
  },
  {
    label: "128, top 8 (Qwen3-30B-A3B's counts)",
    v: { e: 128, k: 8, s: 0, g: "8", p: 0 },
  },
  {
    label: "256, top 8 + 1 shared, 3 dense layers (DeepSeek-V3's counts)",
    v: { e: 256, k: 8, s: 1, g: "16", p: 3 },
  },
];

export default function MoeWidget(): JSX.Element {
  const base = CHAPTERS.moe.spec;
  const [v, setV] = useState<Setting>(PRESETS[2]!.v);
  const set = (patch: Partial<Setting>) =>
    setV((old) => {
      const n = { ...old, ...patch };
      n.k = Math.min(n.k, n.e);
      return n;
    });
  const spec = moeSpec(base, v.e, v.k, v.s, Number(v.g), v.p);
  const p = params(spec);
  const dense = params(base);
  const read = decodeBytes(spec, 8192, 2, 2).total;
  const denseRead = decodeBytes(base, 8192, 2, 2).total;
  const layers = layerTotal(base);
  const dExpert = Math.floor(base.ffns.dense!.d_ff! / Number(v.g));

  return (
    <WidgetFrame
      testId="moe-widget"
      title="Turn Llama 3 8B's FFN into a mixture of experts"
      caption="Same attention, width and depth. Costs from this site's cost model: BF16 weights, batch 1, an 8K-token context for the decode step."
    >
      <div className="flex flex-wrap gap-1.5" role="group" aria-label="Presets">
        {PRESETS.map((x) => (
          <button
            key={x.label}
            type="button"
            onClick={() => setV(x.v)}
            aria-pressed={JSON.stringify(x.v) === JSON.stringify(v)}
            className="focus-ring rounded border border-neutral-300 px-2 py-1 text-xs text-neutral-700 hover:bg-neutral-100 aria-pressed:border-accent aria-pressed:text-accent dark:border-neutral-700 dark:text-neutral-300 dark:hover:bg-neutral-800 dark:aria-pressed:text-indigo-300"
          >
            {x.label}
          </button>
        ))}
      </div>
      <div className="mt-4 grid items-end gap-4 sm:grid-cols-2">
        <Slider
          label="Routed experts per layer"
          value={EXPERTS.indexOf(v.e)}
          min={0}
          max={EXPERTS.length - 1}
          onChange={(i) => set({ e: EXPERTS[i]! })}
          format={(i) => String(EXPERTS[i])}
        />
        <Slider
          label="Active (routed) experts per token"
          value={v.k}
          min={1}
          max={Math.max(1, Math.min(16, v.e))}
          onChange={(k) => set({ k })}
        />
        <Slider
          label="Shared experts (always on)"
          value={v.s}
          min={0}
          max={2}
          onChange={(s) => set({ s })}
        />
        <Slider
          label="Dense FFN in the first layers"
          value={v.p}
          min={0}
          max={3}
          onChange={(p) => set({ p })}
        />
        <Segmented
          label={`Expert width = FFN (${base.ffns.dense!.d_ff}) ÷`}
          value={v.g}
          options={GRAN.map((g) => ({ value: g, label: g }))}
          onChange={(g) => set({ g })}
        />
        <p className="text-xs text-neutral-600 dark:text-neutral-400">
          {v.e <= 1
            ? "One expert is the dense model."
            : `Each expert is ${dExpert} wide; ${layers - v.p} of ${layers} layers use the experts.`}
        </p>
      </div>
      <div
        className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3"
        data-testid="moe-stats"
      >
        <Stat
          label="Total parameters"
          value={formatCount(p.total)}
          hint={`dense: ${formatCount(dense.total)}`}
        />
        <Stat
          label="Active per token"
          value={formatCount(p.active)}
          hint={`${((100 * p.active) / p.total).toFixed(1)}% of total`}
        />
        <Stat
          label="Weights in memory (BF16)"
          value={formatBytes(2 * p.total)}
        />
        <Stat
          label="Bytes read per decode step"
          value={formatBytes(read)}
          hint={`${(read / denseRead).toFixed(2)}× dense`}
        />
        <Stat
          label="Decode FLOPs per token"
          value={formatFlops(decodeFlops(spec, 8192))}
        />
        <Stat
          label="Router weights"
          value={formatCount(v.e > 1 ? (layers - v.p) * base.d_model * v.e : 0)}
        />
      </div>
      <p className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
        At batch 1 a decode step reads only the active experts. A server
        batching many requests routes them to different experts, so it reads far
        more of the total: the memory the weights occupy, not the active count,
        then sets the hardware.
      </p>
    </WidgetFrame>
  );
}
