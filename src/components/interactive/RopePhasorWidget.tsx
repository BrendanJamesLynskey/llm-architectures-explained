"use client";

/**
 * Chapter 2's hero: RoPE as a bank of phasors. A 128-wide head is 64
 * rotating pairs; pair i turns by ω_i = θ^(−2i/d) radians per token, so
 * position is a clock that runs at 64 speeds at once. Eight pairs are drawn,
 * from the fastest to the slowest: the query's phasor (blue) and the key's
 * (orange), the angle between them shaded. The score below is the mean of
 * every pair's cos(Δφ): a sum of cosines, which only the distance m − n can
 * move. Every state is `ropeBank` (tested against reference/visuals.py).
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { useWidth } from "@/components/anim/useTween";
import { Segmented, Stat } from "@/components/ui/Controls";
import { CHAPTERS } from "@/lib/chapters/data";
import {
  ROPE_POSITIONS,
  ropeBank,
  ropeCaption,
  type RopeMode,
} from "@/lib/chapters/visuals";
import { OKABE_ITO } from "@/lib/viz/palette";

const MODELS = ["mistral-7b", "olmo-2-7b", "qwen3-8b"] as const;
type ModelId = (typeof MODELS)[number];

const theta = (t: number): string =>
  t >= 1e6 ? `${t / 1e6}M` : t >= 1e3 ? `${t / 1e3}K` : String(t);

const wave = (w: number): string =>
  w >= 1e9
    ? `${(w / 1e9).toFixed(1)}B`
    : w >= 1e6
      ? `${(w / 1e6).toFixed(1)}M`
      : w >= 1e4
        ? `${Math.round(w / 1e3)}K`
        : w.toFixed(w >= 100 ? 0 : 1);

export default function RopePhasorWidget({
  equation,
}: {
  equation?: ReactNode;
}): JSX.Element {
  const [model, setModel] = useState<ModelId>("mistral-7b");
  const [mode, setMode] = useState<RopeMode>("query");
  const r = CHAPTERS.rope.find((x) => x.id === model)!;
  const states = useMemo(
    () => ropeBank(r.head_dim, r.theta, mode),
    [r.head_dim, r.theta, mode],
  );
  const st = useStepper(states.length, {
    stepMs: 260,
    resetKey: `${model}|${mode}`,
  });
  const [ref, width] = useWidth(640);
  const s = states[st.step]!;
  const W = width;
  const phone = W < 560;
  const cols = phone ? 4 : 8;
  const dial = Math.min(78, (W - 16) / cols - 8);
  const rad = dial / 2 - 6;
  const rows = Math.ceil(s.pairs.length / cols);
  const dialsH = rows * (dial + 34);
  const plotY = dialsH + 26;
  const plotH = phone ? 110 : 130;
  const H = plotY + plotH + 30;
  const px = (t: number) => 34 + (t / (ROPE_POSITIONS - 1)) * (W - 46);
  const py = (v: number) => plotY + ((1 - v) / 2) * plotH;
  const curve = states
    .slice(0, st.step + 1)
    .map(
      (u, k) =>
        `${k ? "L" : "M"}${px(u.t).toFixed(1)},${py(u.score).toFixed(1)}`,
    )
    .join("");
  const hl = mode === "query" ? "a" : "p";
  return (
    <AnimationPanel
      testId="rope-phasors"
      title="RoPE: a bank of clocks running at 64 speeds"
      summary={
        <>
          {r.name}&apos;s heads (θ = {theta(r.theta)}, {r.head_dim} wide: 64
          rotating pairs). Each dial is one pair: the query&apos;s phasor in
          blue, the key&apos;s in orange, turning by ω<sub>i</sub> per token.
          Below, the attention score of equal query and key vectors, the mean of
          cos(Δφ) over all 64 pairs.
        </>
      }
      stepper={st}
      stepLabel="position"
      countFrom={0}
      caption={ropeCaption(s, mode)}
      equation={equation}
      hl={hl}
      visual={
        <div ref={ref} className="min-w-0">
          <svg
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            role="img"
            aria-label="Eight dials, one per rotary pair from fastest to slowest, each with a query and a key arrow and the angle between them; below, the attention score plotted against position"
            className="text-neutral-800 dark:text-neutral-200"
          >
            {s.pairs.map((p, k) => {
              const cx =
                8 + (k % cols) * ((W - 16) / cols) + (W - 16) / cols / 2;
              const cy = Math.floor(k / cols) * (dial + 34) + dial / 2 + 4;
              const qx = cx + rad * Math.cos(p.angQ);
              const qy = cy - rad * Math.sin(p.angQ);
              const kx = cx + rad * Math.cos(p.angK);
              const ky = cy - rad * Math.sin(p.angK);
              // the wedge between the two phasors (shorter way round)
              let d = (p.angQ - p.angK) % (2 * Math.PI);
              if (d > Math.PI) d -= 2 * Math.PI;
              if (d < -Math.PI) d += 2 * Math.PI;
              const r2 = rad * 0.55;
              const a0 = p.angK;
              const a1 = p.angK + d;
              const wedge = `M${cx},${cy} L${cx + r2 * Math.cos(a0)},${cy - r2 * Math.sin(a0)} A${r2},${r2} 0 0 ${d > 0 ? 0 : 1} ${cx + r2 * Math.cos(a1)},${cy - r2 * Math.sin(a1)} Z`;
              return (
                <g key={p.i} data-testid={`rope-pair-${p.i}`}>
                  <circle
                    cx={cx}
                    cy={cy}
                    r={rad}
                    fill="none"
                    stroke="#a3a3a3"
                    strokeOpacity={0.6}
                  />
                  <path d={wedge} fill={OKABE_ITO.green} fillOpacity={0.35} />
                  <line
                    x1={cx}
                    y1={cy}
                    x2={kx}
                    y2={ky}
                    stroke={OKABE_ITO.orange}
                    strokeWidth={3}
                    strokeLinecap="round"
                  />
                  <line
                    x1={cx}
                    y1={cy}
                    x2={qx}
                    y2={qy}
                    stroke={OKABE_ITO.blue}
                    strokeWidth={3}
                    strokeLinecap="round"
                  />
                  <text
                    x={cx}
                    y={cy + dial / 2 + 12}
                    fontSize={11}
                    textAnchor="middle"
                    fill="currentColor"
                  >
                    pair {p.i}
                  </text>
                  <text
                    x={cx}
                    y={cy + dial / 2 + 25}
                    fontSize={11}
                    textAnchor="middle"
                    fill="currentColor"
                  >
                    λ {wave(p.wavelength)}
                  </text>
                </g>
              );
            })}
            <text x={4} y={plotY - 10} fontSize={11} fill="currentColor">
              score = mean of cos(Δφ) over 64 pairs →
            </text>
            {[1, 0, -1].map((v) => (
              <g key={v}>
                <line
                  x1={34}
                  x2={W - 12}
                  y1={py(v)}
                  y2={py(v)}
                  stroke="#a3a3a3"
                  strokeOpacity={v === 0 ? 0.6 : 0.3}
                />
                <text
                  x={30}
                  y={py(v) + 4}
                  fontSize={11}
                  textAnchor="end"
                  fill="currentColor"
                >
                  {v}
                </text>
              </g>
            ))}
            <path
              d={curve}
              fill="none"
              stroke={OKABE_ITO.green}
              strokeWidth={2}
            />
            <circle
              cx={px(s.t)}
              cy={py(s.score)}
              r={5}
              fill={OKABE_ITO.green}
              stroke="currentColor"
            />
            <text x={34} y={H - 6} fontSize={11} fill="currentColor">
              {mode === "query"
                ? "query position m (key fixed at 0) →"
                : "both positions, 8 apart →"}
            </text>
          </svg>
        </div>
      }
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Query position" value={String(s.m)} />
          <Stat label="Key position" value={String(s.n)} />
          <Stat label="Distance" value={String(s.m - s.n)} />
          <Stat label="Score" value={s.score.toFixed(3)} />
        </div>
      }
      params={
        <>
          <Segmented
            label="Model (θ)"
            value={model}
            options={MODELS.map((id) => {
              const x = CHAPTERS.rope.find((y) => y.id === id)!;
              return { value: id, label: `${x.name} (${theta(x.theta)})` };
            })}
            onChange={setModel}
          />
          <Segmented
            label="What moves"
            value={mode}
            options={[
              { value: "query", label: "the query" },
              { value: "both", label: "both, 8 apart" },
            ]}
            onChange={setMode}
          />
        </>
      }
    />
  );
}
