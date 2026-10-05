"use client";

/**
 * Chapter 2: what RoPE's base (θ) and partial rotation do. For a model's
 * settings (read from its pinned config.json) or your own, the wavelength
 * of every rotated pair of dimensions, and how many turn less than once
 * across the chosen context. Closed forms in src/lib/chapters/model.ts,
 * checked against reference/chapter_model.py.
 */
import { useState } from "react";

import { Segmented, Slider, Stat } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import { ScrollBox } from "@/components/viz/Legend";
import { formatTokens } from "@/lib/arch/format";
import { CHAPTERS } from "@/lib/chapters/data";
import {
  ropeLongPairs,
  ropeWavelengths,
  rotaryDims,
} from "@/lib/chapters/model";

const W = 640;
const H = 220;
const M = { l: 56, r: 12, t: 10, b: 36 };

const two = (v: number): string => String(Number(v.toPrecision(2)));

function fmtLen(x: number): string {
  if (x >= 1e9) return `${two(x / 1e9)}G`;
  if (x >= 1e6) return `${two(x / 1e6)}M`;
  if (x >= 1e3) return `${two(x / 1e3)}K`;
  return two(x);
}

export default function RopeWidget(): JSX.Element {
  const models = CHAPTERS.rope;
  const [id, setId] = useState(models[0]!.id);
  const [custom, setCustom] = useState(false);
  const [log10theta, setLog10theta] = useState(4);
  const [fraction, setFraction] = useState<"1" | "0.5" | "0.25">("1");
  const [log2ctx, setLog2ctx] = useState(15);
  const m = models.find((x) => x.id === id)!;
  const theta = custom ? 10 ** log10theta : m.theta;
  const frac = custom ? Number(fraction) : m.fraction;
  const head = custom ? 128 : m.head_dim;
  const ctx = 2 ** log2ctx;
  const waves = ropeWavelengths(head, frac, theta);
  const long = ropeLongPairs(head, frac, theta, ctx);
  const rot = rotaryDims(head, frac);

  const lo = Math.log10(Math.min(...waves, ctx) / 2);
  const hi = Math.log10(Math.max(...waves, ctx) * 2);
  const bw = (W - M.l - M.r) / waves.length;
  const sy = (v: number) =>
    H - M.b - ((Math.log10(v) - lo) / (hi - lo)) * (H - M.t - M.b);
  const ticks: number[] = [];
  for (let e = Math.ceil(lo); e <= Math.floor(hi); e++) ticks.push(10 ** e);

  return (
    <WidgetFrame
      testId="rope-widget"
      title="RoPE wavelengths against the context"
      caption="Each rotated pair of dimensions turns at its own rate; the pair's wavelength is 2π·θ^(2i/d). Pairs above the line turn less than once across the context."
    >
      <div className="grid items-end gap-4 sm:grid-cols-2">
        <label className="flex min-w-0 flex-col gap-1 text-sm">
          <span className="font-medium text-neutral-700 dark:text-neutral-300">
            Settings
          </span>
          <select
            value={custom ? "custom" : id}
            onChange={(e) => {
              if (e.target.value === "custom") setCustom(true);
              else {
                setCustom(false);
                setId(e.target.value);
              }
            }}
            className="focus-ring w-full min-w-0 rounded border border-neutral-300 bg-white px-2 py-1 dark:border-neutral-700 dark:bg-neutral-950"
          >
            {models.map((x) => (
              <option key={x.id} value={x.id}>
                {x.name} (θ = {x.theta.toLocaleString("en-GB")})
              </option>
            ))}
            <option value="custom">Your own θ and rotated fraction</option>
          </select>
        </label>
        <Slider
          label="Context length"
          value={log2ctx}
          min={10}
          max={20}
          onChange={setLog2ctx}
          format={(v) => `${formatTokens(2 ** v)} tokens`}
        />
        {custom && (
          <>
            <Slider
              label="Base θ"
              value={log10theta}
              min={3}
              max={8}
              step={0.5}
              onChange={setLog10theta}
              format={(v) => Math.round(10 ** v).toLocaleString("en-GB")}
            />
            <Segmented
              label="Rotated fraction of a 128-dim head"
              value={fraction}
              options={[
                { value: "1", label: "all" },
                { value: "0.5", label: "half" },
                { value: "0.25", label: "quarter" },
              ]}
              onChange={setFraction}
            />
          </>
        )}
      </div>
      <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="Rotated dims" value={`${rot} of ${head}`} />
        <Stat label="Pairs" value={String(waves.length)} />
        <Stat
          label="Slower than the context"
          value={`${long} pair${long === 1 ? "" : "s"}`}
        />
        <Stat
          label="Longest wavelength"
          value={`${fmtLen(waves[waves.length - 1]!)} tokens`}
        />
      </div>
      <ScrollBox label="Wavelength per pair chart">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label="Wavelength of each rotated pair, log scale, with the context length marked"
          className="h-auto w-full min-w-[480px]"
          data-long-pairs={long}
        >
          {ticks.map((t) => (
            <g key={t}>
              <line
                x1={M.l}
                x2={W - M.r}
                y1={sy(t)}
                y2={sy(t)}
                className="stroke-neutral-200 dark:stroke-neutral-800"
              />
              <text
                x={M.l - 6}
                y={sy(t) + 4}
                textAnchor="end"
                className="fill-neutral-600 text-[11px] dark:fill-neutral-400"
              >
                {fmtLen(t)}
              </text>
            </g>
          ))}
          {waves.map((w, i) => (
            <rect
              key={i}
              x={M.l + i * bw + bw * 0.15}
              width={bw * 0.7}
              y={sy(w)}
              height={H - M.b - sy(w)}
              className={
                w > ctx
                  ? "fill-amber-500 dark:fill-amber-400"
                  : "fill-indigo-500 dark:fill-indigo-400"
              }
            />
          ))}
          <line
            x1={M.l}
            x2={W - M.r}
            y1={sy(ctx)}
            y2={sy(ctx)}
            strokeDasharray="5 4"
            className="stroke-rose-600 dark:stroke-rose-400"
            strokeWidth={2}
          />
          <text
            x={W - M.r}
            y={sy(ctx) - 4}
            textAnchor="end"
            className="fill-rose-700 text-[11px] dark:fill-rose-300"
          >
            context {formatTokens(ctx)}
          </text>
          <text
            x={(W + M.l) / 2}
            y={H - 8}
            textAnchor="middle"
            className="fill-neutral-700 text-[11px] dark:fill-neutral-300"
          >
            Rotated pair i (fastest on the left)
          </text>
        </svg>
      </ScrollBox>
      <p className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
        {custom ? (
          "Your own settings, on a 128-dimension head."
        ) : (
          <>
            {m.name}: θ = {m.theta.toLocaleString("en-GB")}, {m.rotated}{" "}
            {m.head_dim}
            {m.fraction < 1 ? `, rotary fraction ${m.fraction}` : ""}, from its{" "}
            <a
              href={m.config}
              className="focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300"
            >
              config.json at a pinned revision
            </a>
            .
          </>
        )}
      </p>
    </WidgetFrame>
  );
}
