"use client";

/**
 * Chapter 7: what multi-token prediction modules weigh and what they can
 * buy at decode. For any model in the data set with MTP layers: the
 * modules' weights (cost model), the tokens a step emits when each drafted
 * token is accepted with probability a (1 + a + … + a^D), and the speed-up
 * if decode is memory-bound at batch 1 (mtpSpeedup, checked against
 * reference/chapter_model.py). An idealisation: acceptance is independent
 * and equal at every depth, and the modules' own KV reads are ignored.
 */
import { useState } from "react";

import { Slider, Stat } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import { params } from "@/lib/arch/costModel";
import { formatBytes, formatCount, formatTokens } from "@/lib/arch/format";
import { CHAPTERS } from "@/lib/chapters/data";
import { mtpModuleActive, mtpSpeedup } from "@/lib/chapters/model";

import { useSpecs } from "./useSpecs";

export default function MtpWidget(): JSX.Element {
  const { models, error } = useSpecs();
  const [id, setId] = useState("deepseek-v3");
  const [depth, setDepth] = useState(1);
  const [acc, setAcc] = useState(0.85);
  const [log2ctx, setLog2ctx] = useState(13);
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
  const list = CHAPTERS.mtp
    .map((mid) => models.find((x) => x.id === mid)!)
    .filter((x) => x.spec);
  const m = list.find((x) => x.id === id) ?? list[0]!;
  const spec = m.spec!;
  const ctx = 2 ** log2ctx;
  const p = params(spec);
  const mod = mtpModuleActive(spec);
  const s = mtpSpeedup(spec, ctx, depth, acc, 2, 2);

  return (
    <WidgetFrame
      testId="mtp-widget"
      title="Multi-token prediction: what a draft module weighs and buys"
      caption="Batch 1, memory-bound decode, BF16: a step reads the model's active weights and its KV cache once, plus each MTP module's weights. Each drafted token is accepted with probability a given the ones before it."
    >
      <div className="grid items-end gap-4 sm:grid-cols-2">
        <label className="flex min-w-0 flex-col gap-1 text-sm">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            Model ({list.length} in the data set have MTP layers)
          </span>
          <select
            value={m.id}
            onChange={(e) => setId(e.target.value)}
            className="focus-ring w-full min-w-0 rounded border border-neutral-300 bg-white px-2 py-1 dark:border-neutral-700 dark:bg-neutral-950"
          >
            {list.map((x) => (
              <option key={x.id} value={x.id}>
                {x.name} ({x.spec!.mtp_layers} MTP layer
                {x.spec!.mtp_layers === 1 ? "" : "s"})
              </option>
            ))}
          </select>
        </label>
        <Slider
          label="Drafted tokens per step (depth)"
          value={depth}
          min={1}
          max={4}
          onChange={setDepth}
        />
        <Slider
          label="Acceptance a"
          value={acc}
          min={0.5}
          max={0.95}
          step={0.05}
          onChange={setAcc}
          format={(v) => v.toFixed(2)}
        />
        <Slider
          label="Context length"
          value={log2ctx}
          min={10}
          max={17}
          onChange={setLog2ctx}
          format={(v) => `${formatTokens(2 ** v)} tokens`}
        />
      </div>
      <div
        className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3"
        data-testid="mtp-stats"
      >
        <Stat
          label="One MTP module, active"
          value={formatCount(mod)}
          hint={`${((100 * mod) / p.active).toFixed(1)}% of the model's ${formatCount(p.active)} active`}
        />
        <Stat
          label="MTP weights stored"
          value={formatCount(p.mtp)}
          hint="all modules, every expert"
        />
        <Stat
          label="Tokens per step"
          value={s.tokens_per_step.toFixed(3)}
          plain
          hint="1 + a + … + a^depth"
        />
        <Stat label="Bytes per step" value={formatBytes(s.bytes_per_step)} />
        <Stat label="Bytes per token" value={formatBytes(s.bytes_per_token)} />
        <Stat
          label="Speed-up (memory-bound)"
          value={`${s.speedup.toFixed(2)}×`}
        />
      </div>
      {depth > (spec.mtp_layers ?? 0) && (
        <p className="mt-2 text-xs text-amber-800 dark:text-amber-200">
          {m.name} ships {spec.mtp_layers} MTP layer
          {spec.mtp_layers === 1 ? "" : "s"}; deeper drafting here is
          hypothetical.
        </p>
      )}
    </WidgetFrame>
  );
}
