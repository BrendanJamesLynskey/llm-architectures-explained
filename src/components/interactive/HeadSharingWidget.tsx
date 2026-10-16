"use client";

/**
 * Chapter 1's hero: how attention shares its keys and values, drawn to
 * scale. Llama 3 8B's body with four kinds of attention: 32 query heads
 * (top) read the K/V units below them (one per head, one per group, one for
 * all, or MLA's single latent), and the square on the right has the area of
 * one token's KV cache across all 32 layers. Each frame is a variant of
 * `headSharing` (the cost model's bytes, tested against reference/visuals.py).
 */
import { useMemo, type ReactNode } from "react";

import { AnimationPanel } from "@/components/anim/AnimationPanel";
import { useStepper } from "@/components/anim/useStepper";
import { useWidth } from "@/components/anim/useTween";
import { Stat } from "@/components/ui/Controls";
import { CHAPTERS } from "@/lib/chapters/data";
import { headSharing, sharingCaption } from "@/lib/chapters/visuals";
import { OKABE_ITO } from "@/lib/viz/palette";

const kib = (b: number): string =>
  `${(b / 1024).toFixed(b % 1024 ? 1 : 0)} KiB`;

export default function HeadSharingWidget({
  equation,
}: {
  equation?: ReactNode;
}): JSX.Element {
  const variants = useMemo(() => headSharing(CHAPTERS.attention.variants), []);
  const st = useStepper(variants.length, { stepMs: 1800 });
  const [ref, width] = useWidth(640);
  const v = variants[st.step]!;
  const base = variants[0]!;
  const W = width;
  const phone = W < 560;
  // left: heads and units; right (or below on a phone): the squares
  const leftW = phone ? W - 8 : W * 0.6;
  const hw = (leftW - 16) / v.heads;
  const qY = 26;
  const uY = 110;
  const unitW = (leftW - 16) / v.units;
  const maxSide = phone ? Math.min(150, W / 2 - 20) : Math.min(170, W * 0.36);
  const side = (b: number) => maxSide * Math.sqrt(b / base.bytesPerToken);
  const sqX = phone ? 8 : leftW + 24;
  const sqY = phone ? uY + 60 : 20;
  const H = phone ? sqY + maxSide + 60 : Math.max(uY + 70, sqY + maxSide + 40);
  const unitOf = (h: number) => Math.floor(h / (v.heads / v.units));
  return (
    <AnimationPanel
      testId="head-sharing"
      title="Sharing keys and values: MHA, GQA, MQA, MLA, to scale"
      summary={
        <>
          Llama 3 8B&apos;s shape (32 layers, 32 query heads of 128) with four
          kinds of attention. Lines join each query head to the keys and values
          it reads; the square&apos;s area is one token&apos;s KV cache in BF16,
          from the cost model.
        </>
      }
      stepper={st}
      stepLabel="variant"
      countFrom={0}
      caption={sharingCaption(v, base)}
      equation={equation}
      hl={v.id === "mla" ? "w" : "a"}
      visual={
        <div ref={ref} className="min-w-0">
          <svg
            width={W}
            height={H}
            viewBox={`0 0 ${W} ${H}`}
            role="img"
            aria-label="A row of 32 query heads joined by lines to the key and value units they read, and squares whose areas are the KV cache per token of each attention kind"
            className="text-neutral-800 dark:text-neutral-200"
          >
            <text x={8} y={14} fontSize={11} fill="currentColor">
              32 query heads
            </text>
            {Array.from({ length: v.heads }, (_, h) => {
              const x = 8 + h * hw + hw / 2;
              const u = unitOf(h);
              const ux = 8 + u * unitW + unitW / 2;
              return (
                <g key={h}>
                  <line
                    x1={x}
                    y1={qY + 10}
                    x2={ux}
                    y2={uY}
                    stroke="#a3a3a3"
                    strokeOpacity={0.7}
                  />
                  <rect
                    x={x - Math.min(5, hw / 2 - 1)}
                    y={qY}
                    width={Math.min(10, hw - 2)}
                    height={10}
                    rx={2}
                    fill={OKABE_ITO.blue}
                  />
                </g>
              );
            })}
            {Array.from({ length: v.units }, (_, u) => (
              <rect
                key={u}
                data-testid="kv-unit"
                x={8 + u * unitW + 1}
                y={uY}
                width={Math.max(2, unitW - 2)}
                height={22}
                rx={3}
                fill={v.id === "mla" ? OKABE_ITO.purple : OKABE_ITO.orange}
              />
            ))}
            <text x={8} y={uY + 38} fontSize={11} fill="currentColor">
              {v.id === "mla"
                ? `1 latent of ${v.unitElems} numbers per token per layer`
                : `${v.units} K/V head${v.units > 1 ? "s" : ""} × ${v.unitElems} numbers per token per layer`}
            </text>
            {variants.map((u, k) => {
              const sd = side(u.bytesPerToken);
              const on = k === st.step;
              return (
                <g key={u.id} data-testid={`kv-square-${u.id}`}>
                  <rect
                    x={sqX}
                    y={sqY + maxSide - sd}
                    width={sd}
                    height={sd}
                    fill={u.id === "mla" ? OKABE_ITO.purple : OKABE_ITO.orange}
                    fillOpacity={on ? 0.85 : 0.08}
                    stroke={on ? "currentColor" : "#a3a3a3"}
                    strokeDasharray={on ? undefined : "3 3"}
                  />
                </g>
              );
            })}
            <text
              x={sqX}
              y={sqY + maxSide + 16}
              fontSize={11}
              fill="currentColor"
            >
              {v.label}: {kib(v.bytesPerToken)} per token
            </text>
            <text
              x={sqX}
              y={sqY + maxSide + 30}
              fontSize={11}
              fill="currentColor"
            >
              (dashed: the other three)
            </text>
          </svg>
        </div>
      }
      stats={
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          <Stat label="Query heads" value={String(v.heads)} />
          <Stat label="K/V units per layer" value={String(v.units)} />
          <Stat label="KV per token" value={kib(v.bytesPerToken)} />
          <Stat
            label="Against MHA"
            value={
              v.id === "mha"
                ? "baseline"
                : `${(base.bytesPerToken / v.bytesPerToken).toFixed(1)}× less`
            }
          />
        </div>
      }
    />
  );
}
