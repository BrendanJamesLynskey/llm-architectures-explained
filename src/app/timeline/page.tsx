/**
 * /timeline: when each variation shows up across the data set. One row per
 * feature, one dot per model that has it, placed by release month. Every
 * dot is derived from the model's data (src/lib/arch/features.ts).
 *
 * Server Component, static SVG.
 */
import Link from "next/link";

import {
  FEATURES,
  type FeatureId,
  features,
  released,
} from "@/lib/arch/features";
import { byRelease } from "@/lib/data";

export const metadata = {
  title: "Timeline",
  description:
    "When each architectural variation appears across 160 language models, from 2017 to today.",
};

const ROWS: FeatureId[] = [
  "encoder-decoder",
  "learned",
  "rope",
  "partial-rope",
  "nope",
  "parallel",
  "gqa",
  "mqa",
  "sliding",
  "chunked",
  "moe",
  "shared-expert",
  "mla",
  "ssm",
  "linear",
  "deltanet",
  "hybrid",
  "sparse",
  "qk-norm",
  "post-norm",
  "sandwich",
  "mtp",
  "looped",
  "ced",
];

const X0 = 2017;
const X1 = 2027;
const W = 900;
const L = 150;
const ROW = 24;

const A =
  "focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300";

export default function TimelinePage(): JSX.Element {
  const models = byRelease().filter((m) => released(m) !== null);
  const feats = new Map(models.map((m) => [m.id, features(m)]));
  const sx = (t: number) => L + ((t - X0) / (X1 - X0)) * (W - L - 24);
  const H = ROWS.length * ROW + 40;
  const earliest = ROWS.map((f) => {
    const m = models.find((x) => feats.get(x.id)!.has(f));
    return { f, m };
  });
  return (
    <main className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        /timeline
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        How the variations spread
      </h1>
      <p className="mt-3 max-w-3xl text-neutral-700 dark:text-neutral-300">
        Each row is one design choice; each dot is a model in this data set that
        makes it, placed by release month (for open models, the month the
        Hugging Face repository was created unless a paper or announcement date
        is recorded). Hover or focus a dot for the model; follow it for the
        sources. Filled dots are open weights, hollow dots closed.
      </p>
      <figure
        tabIndex={0}
        className="mt-6 overflow-x-auto rounded-lg border border-neutral-200 bg-white p-3 dark:border-neutral-800 dark:bg-neutral-950"
      >
        <svg
          viewBox={`0 0 ${W} ${H}`}
          aria-label="Timeline of architectural features by model release date"
          className="h-auto w-full min-w-[720px]"
        >
          <title>
            Timeline of architectural features by model release date
          </title>
          {Array.from({ length: X1 - X0 + 1 }, (_, i) => X0 + i).map((y) => (
            <g key={y}>
              <line
                x1={sx(y)}
                x2={sx(y)}
                y1={8}
                y2={H - 28}
                className="stroke-neutral-200 dark:stroke-neutral-800"
              />
              <text
                x={sx(y)}
                y={H - 12}
                textAnchor="middle"
                className="fill-neutral-600 text-[11px] dark:fill-neutral-400"
              >
                {y}
              </text>
            </g>
          ))}
          {ROWS.map((f, r) => {
            const meta = FEATURES.find((x) => x.id === f)!;
            const y = 18 + r * ROW;
            const dots = models.filter((m) => feats.get(m.id)!.has(f));
            return (
              <g key={f}>
                <text
                  x={L - 8}
                  y={y + 4}
                  textAnchor="end"
                  className="fill-neutral-800 text-[11px] dark:fill-neutral-200"
                >
                  {meta.label}
                </text>
                <line
                  x1={L}
                  x2={W - 10}
                  y1={y}
                  y2={y}
                  className="stroke-neutral-100 dark:stroke-neutral-900"
                />
                {dots.map((m, i) => (
                  <a
                    key={m.id}
                    href={`/models/${m.id}`}
                    aria-label={`${m.name}, ${m.facts.released.v}: ${meta.label}`}
                  >
                    <circle
                      cx={sx(released(m)!)}
                      cy={y + ((i % 3) - 1) * 4}
                      r={3.5}
                      className={
                        m.open_weights
                          ? "fill-indigo-600 dark:fill-indigo-400"
                          : "fill-white stroke-indigo-600 dark:fill-neutral-950 dark:stroke-indigo-400"
                      }
                      strokeWidth={1.5}
                    >
                      <title>{`${m.name} (${m.facts.released.v})`}</title>
                    </circle>
                  </a>
                ))}
              </g>
            );
          })}
        </svg>
      </figure>
      <section className="mt-10">
        <h2 className="text-xl font-semibold tracking-tight">
          Earliest in this data set
        </h2>
        <p className="mt-1 text-sm text-neutral-700 dark:text-neutral-300">
          The first model here with each feature. This is a property of the data
          set, not a claim about who invented it: many ideas appeared first in
          papers or in models not listed.
        </p>
        <ul className="mt-3 grid gap-x-6 gap-y-1 text-sm sm:grid-cols-2">
          {earliest.map(({ f, m }) => (
            <li key={f}>
              <span className="font-medium">
                {FEATURES.find((x) => x.id === f)!.label}
              </span>
              :{" "}
              {m ? (
                <Link className={A} href={`/models/${m.id}`}>
                  {m.name}
                </Link>
              ) : (
                "—"
              )}
              {m && (
                <span className="text-neutral-600 dark:text-neutral-400">
                  {" "}
                  ({m.facts.released.v})
                </span>
              )}
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
