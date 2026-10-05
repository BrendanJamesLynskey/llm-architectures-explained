"use client";

/**
 * Chapter 3: where the norms go, in the textbook variance idealisation
 * (src/lib/chapters/model.ts normProfile, checked against
 * reference/chapter_model.py). The residual stream's RMS along the stack,
 * each sub-block's update relative to the stream, and how much of the
 * embedding survives to the top, for post-LN, pre-norm and a norm on the
 * sub-block's output (sandwich / OLMo 2-style). Plus the cost model's count
 * of what the norms weigh.
 */
import { useState } from "react";

import { Slider, Stat } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import { Legend, ScrollBox } from "@/components/viz/Legend";
import { LinearChart } from "@/components/viz/LinearChart";
import { PALETTE } from "@/components/viz/LineChart";
import { params } from "@/lib/arch/costModel";
import { formatCount } from "@/lib/arch/format";
import { CHAPTERS } from "@/lib/chapters/data";
import { normProfile, PLACEMENTS, type Placement } from "@/lib/chapters/model";

const LABELS: Record<Placement, string> = {
  "post-ln": "Post-LN (norm after the residual add)",
  "pre-norm": "Pre-norm (norm before each sub-block)",
  "output-norm": "Norm on the sub-block output (sandwich, OLMo 2)",
};

export default function NormWidget(): JSX.Element {
  const [layers, setLayers] = useState(32);
  const [gain, setGain] = useState(1.0);
  const profiles = PLACEMENTS.map((p, i) => ({
    p,
    colour: PALETTE[i]!,
    out: normProfile(p, layers, gain),
  }));
  const series = profiles.map(({ p, colour, out }) => ({
    id: p,
    label: LABELS[p],
    colour,
    // dashed, so it stays visible where it coincides with pre-norm (gain 1)
    dashed: p === "output-norm",
    points: out.rms.map((v, k) => [k + 1, v] as [number, number]),
  }));
  const base = params(CHAPTERS.moe.spec);

  return (
    <WidgetFrame
      testId="norm-widget"
      title="The residual stream under three norm placements"
      caption="Idealised: a unit-RMS embedding, and each sub-block adds an output independent of the stream, with RMS = gain × its input's (or the norm's scale, 1, when its output is normalised). Two sub-blocks per layer."
    >
      <div className="grid items-end gap-4 sm:grid-cols-2">
        <Slider
          label="Layers"
          value={layers}
          min={2}
          max={96}
          onChange={setLayers}
        />
        <Slider
          label="Sub-block gain"
          value={gain}
          min={0.25}
          max={4}
          step={0.25}
          onChange={setGain}
          format={(v) => `${v.toFixed(2)}×`}
        />
      </div>
      <div className="mt-4 grid gap-2 sm:grid-cols-3" data-testid="norm-stats">
        {profiles.map(({ p, out }) => (
          <Stat
            key={p}
            label={
              p === "post-ln"
                ? "Post-LN"
                : p === "pre-norm"
                  ? "Pre-norm"
                  : "Output norm"
            }
            value={`RMS ${out.rms[out.rms.length - 1]!.toFixed(2)}, last update ${(
              100 * out.update[out.update.length - 1]!
            ).toFixed(1)}%`}
            hint={`Embedding share at the top: ${(100 * out.embedding_share).toPrecision(3)}%`}
          />
        ))}
      </div>
      <ScrollBox label="Residual RMS chart">
        <LinearChart
          series={series}
          title="Residual-stream RMS after each sub-block"
          xLabel="Sub-block (two per layer)"
          yLabel="Residual RMS"
          formatY={(v) => v.toFixed(1)}
        />
      </ScrollBox>
      <Legend series={series} />
      <p className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
        What they weigh: in Llama 3 8B the per-layer norm scales are{" "}
        {formatCount(base.norms)} of {formatCount(base.total)} parameters (
        {((100 * base.norms) / base.total).toFixed(4)}%). Placement costs
        nothing to run; it changes how training behaves.
      </p>
    </WidgetFrame>
  );
}
