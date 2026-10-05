"use client";

/**
 * Chapter 8: looping the layer stack. Ouro 2.6B's sourced shape run 1–4
 * times per token (loopedSpec, checked against reference/chapter_model.py):
 * the parameters stay put while FLOPs scale with the loops, and the KV
 * cache is either one per pass or shared (Ouro's decode-time reuse).
 */
import { useState } from "react";

import { Segmented, Slider, Stat } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import { decodeFlops, params, prefillFlops } from "@/lib/arch/costModel";
import {
  formatBytes,
  formatCount,
  formatFlops,
  formatTokens,
} from "@/lib/arch/format";
import { CHAPTERS } from "@/lib/chapters/data";
import { loopedKvBytes, loopedSpec } from "@/lib/chapters/model";

const CACHE = [
  { value: "per-loop", label: "One cache per pass" },
  { value: "shared", label: "One shared cache" },
] as const;

export default function LoopWidget(): JSX.Element {
  const [loops, setLoops] = useState(4);
  const [log2ctx, setLog2ctx] = useState(13);
  const [cache, setCache] =
    useState<(typeof CACHE)[number]["value"]>("per-loop");
  const ctx = 2 ** log2ctx;
  const spec = loopedSpec(CHAPTERS.loops.spec, loops);
  const one = loopedSpec(CHAPTERS.loops.spec, 1);
  const p = params(spec);
  const kv = loopedKvBytes(spec, ctx, 2, cache === "per-loop");
  const f = decodeFlops(spec, ctx);
  const f1 = decodeFlops(one, ctx);

  return (
    <WidgetFrame
      testId="loop-widget"
      title="Run Ouro 2.6B's stack more than once"
      caption="The same 48 layers, reused per token. Costs from this site's cost model: BF16 cache, batch 1."
    >
      <div className="grid items-end gap-4 sm:grid-cols-3">
        <Slider
          label="Passes through the stack"
          value={loops}
          min={1}
          max={4}
          onChange={setLoops}
        />
        <Slider
          label="Context length"
          value={log2ctx}
          min={10}
          max={16}
          onChange={setLog2ctx}
          format={(v) => `${formatTokens(2 ** v)} tokens`}
        />
        <Segmented
          label="KV cache"
          value={cache}
          options={CACHE}
          onChange={setCache}
        />
      </div>
      <div
        className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4"
        data-testid="loop-stats"
      >
        <Stat
          label="Parameters"
          value={formatCount(p.total)}
          hint="unchanged by looping"
        />
        <Stat
          label="Decode FLOPs / token"
          value={formatFlops(f)}
          hint={`${(f / f1).toFixed(2)}× one pass`}
        />
        <Stat
          label={`Prefill FLOPs, ${formatTokens(ctx)} prompt`}
          value={formatFlops(prefillFlops(spec, ctx))}
        />
        <Stat label={`KV at ${formatTokens(ctx)}`} value={formatBytes(kv)} />
      </div>
      <p className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
        Ouro&apos;s paper reports that every pass needs its own cache during
        prefill, and that during decoding keeping only the last pass&apos;s
        cache (or an average) loses little while using a quarter of the memory
        (arXiv:2510.25741, section 5.4.2, Table 14).
      </p>
    </WidgetFrame>
  );
}
