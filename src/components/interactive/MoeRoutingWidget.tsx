"use client";

/**
 * Chapter 4's hero: tokens routed to experts. Each step, 64 tokens (left)
 * each pick their top 2 of 8 experts by sigmoid affinity; the lines show
 * the picks, the bars each expert's load against the mean (dashed). The
 * tokens cluster near three experts, so the load starts lopsided. With
 * DeepSeek-V3's auxiliary-loss-free balancing on, each expert's bias
 * (in brackets after its load) is nudged after every step, down when it was
 * over the mean and up when under, and the loads even out; off, the hot
 * experts stay hot. Every state is `moeRouting` (tested against
 * reference/visuals.py).
 */
import { useMemo, useState, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { useWidth } from "@/components/anim/useTween";
import { Segmented, Stat } from "@/components/ui/Controls";
import { MOE, moeCaption, moeRouting } from "@/lib/chapters/visuals";
import { OKABE_ITO } from "@/lib/viz/palette";

const EXPERT = [
  OKABE_ITO.blue,
  OKABE_ITO.orange,
  OKABE_ITO.green,
  OKABE_ITO.purple,
  OKABE_ITO.sky,
  OKABE_ITO.vermillion,
  "#737373",
  OKABE_ITO.yellow,
];

export default function MoeRoutingWidget({
  equation,
}: {
  equation?: ReactNode;
}): JSX.Element {
  const [balance, setBalance] = useState<"on" | "off">("on");
  const states = useMemo(
    () => moeRouting(balance === "on" ? MOE.gamma : 0),
    [balance],
  );
  const st = useStepper(states.length, { stepMs: 700, resetKey: balance });
  const [ref, width] = useWidth(640);
  const s = states[st.step]!;
  const W = width;
  const phone = W < 560;
  const E = MOE.experts;
  const mean = (MOE.tokens * MOE.top_k) / E;
  const maxLoad = Math.max(
    mean * 2.6,
    ...states.map((x) => Math.max(...x.loads)),
  );
  // tokens: a column of dots on the left; experts: a column of boxes
  const tokX = 14;
  const top = 24;
  const flowH = phone ? 230 : 280;
  const tokGap = flowH / MOE.tokens;
  const expX = phone ? W * 0.42 : W * 0.38;
  const expH = flowH / E;
  const barX = expX + 46;
  const barW = (phone ? W - 8 : W * 0.68) - barX;
  const lx = (l: number) => barX + (l / maxLoad) * barW;
  const histX = phone ? 34 : W * 0.72;
  const histY = phone ? top + flowH + 44 : top;
  const histW = phone ? W - 46 : W - histX - 10;
  const histH = phone ? 110 : 150;
  const H = phone ? histY + histH + 34 : top + flowH + 30;
  const hx = (k: number) => histX + (k / (states.length - 1)) * histW;
  const hy = (v: number) => histY + histH - ((v - 1) / 1.8) * histH;
  const hist = states
    .slice(0, st.step + 1)
    .map(
      (x, k) =>
        `${k ? "L" : "M"}${hx(k).toFixed(1)},${hy(Math.min(2.8, x.imbalance)).toFixed(1)}`,
    )
    .join("");
  return (
    <AnimationPanel
      testId="moe-routing"
      title="Routing tokens to experts, and balancing the load"
      summary={
        <>
          {MOE.tokens} tokens a step, each sent to its top {MOE.top_k} of {E}{" "}
          experts by sigmoid affinity, as DeepSeek-V3 routes. The tokens cluster
          near three experts, so the load starts uneven. Balancing on: a bias
          per expert, used only to choose, nudged by γ = {MOE.gamma} after every
          step.
        </>
      }
      stepper={st}
      stepLabel="step"
      countFrom={0}
      caption={moeCaption(s, MOE, balance === "on")}
      equation={equation}
      hl={balance === "on" ? "u" : "c"}
      visual={
        <div ref={ref} className="min-w-0">
          <svg
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            role="img"
            aria-label="Tokens on the left joined by lines to their two chosen experts; each expert's load as a bar against the mean, with its routing bias; and the busiest expert's load over the mean, step by step"
            className="text-neutral-800 dark:text-neutral-200"
          >
            <text x={tokX - 6} y={14} fontSize={11} fill="currentColor">
              tokens
            </text>
            <text x={expX - 8} y={14} fontSize={11} fill="currentColor">
              experts: load (bias)
            </text>
            {s.routes.map((r, t) => {
              const y = top + t * tokGap + tokGap / 2;
              return (
                <g key={t}>
                  {r.experts.map((e, j) => (
                    <line
                      key={j}
                      x1={tokX + 3}
                      y1={y}
                      x2={expX - 4}
                      y2={top + e * expH + expH / 2}
                      stroke={EXPERT[e]}
                      strokeOpacity={0.12 + 0.5 * r.gates[j]!}
                      strokeWidth={1}
                    />
                  ))}
                  <circle
                    cx={tokX}
                    cy={y}
                    r={Math.min(2.5, tokGap / 2)}
                    fill="currentColor"
                  />
                </g>
              );
            })}
            {s.loads.map((l, e) => {
              const y = top + e * expH;
              const b = s.bias[e]!;
              return (
                <g key={e} data-testid={`expert-${e}`} data-load={l}>
                  <rect
                    x={expX - 4}
                    y={y + 3}
                    width={40}
                    height={expH - 6}
                    rx={3}
                    fill={EXPERT[e]}
                    stroke={e === 7 ? "#404040" : "none"}
                  />
                  <text
                    x={expX + 16}
                    y={y + expH / 2 + 4}
                    fontSize={11}
                    textAnchor="middle"
                    fill={e === 1 || e === 4 || e === 7 ? "#000" : "#fff"}
                  >
                    {e}
                  </text>
                  <rect
                    x={barX}
                    y={y + 4}
                    width={Math.max(1, lx(l) - barX)}
                    height={expH * 0.45}
                    fill={l > mean ? OKABE_ITO.vermillion : OKABE_ITO.blue}
                  />
                  <text
                    x={lx(l) + 4}
                    y={y + 4 + expH * 0.45 - 1}
                    fontSize={11}
                    fill="currentColor"
                  >
                    {l} ({b >= 0 ? "+" : "−"}
                    {Math.abs(b).toFixed(2)})
                  </text>
                </g>
              );
            })}
            <line
              x1={lx(mean)}
              x2={lx(mean)}
              y1={top}
              y2={top + flowH}
              stroke="currentColor"
              strokeDasharray="3 3"
            />
            <text
              x={lx(mean) + 3}
              y={top + flowH + 14}
              fontSize={11}
              fill="currentColor"
            >
              mean {mean}
            </text>
            <text x={histX} y={histY - 8} fontSize={11} fill="currentColor">
              busiest ÷ mean, per step
            </text>
            {[1, 2].map((v) => (
              <g key={v}>
                <line
                  x1={histX}
                  x2={histX + histW}
                  y1={hy(v)}
                  y2={hy(v)}
                  stroke="#a3a3a3"
                  strokeOpacity={0.4}
                />
                <text
                  x={histX - 4}
                  y={hy(v) + 4}
                  fontSize={11}
                  textAnchor="end"
                  fill="currentColor"
                >
                  {v}×
                </text>
              </g>
            ))}
            <path
              d={hist}
              fill="none"
              stroke={OKABE_ITO.vermillion}
              strokeWidth={2}
            />
          </svg>
        </div>
      }
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Step" value={`${s.step + 1} of ${states.length}`} />
          <Stat label="Busiest ÷ mean" value={`${s.imbalance.toFixed(2)}×`} />
          <Stat label="Busiest load" value={String(Math.max(...s.loads))} />
          <Stat label="Quietest load" value={String(Math.min(...s.loads))} />
        </div>
      }
      params={
        <Segmented
          label="Auxiliary-loss-free balancing"
          value={balance}
          options={[
            { value: "on", label: "on" },
            { value: "off", label: "off" },
          ]}
          onChange={setBalance}
        />
      }
    />
  );
}
