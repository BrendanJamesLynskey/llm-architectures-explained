/**
 * The landing page's hero: one dot per model, by release month and the KV
 * cache a token costs it at 32K context, coloured by attention family
 * (`kvLandscape`, from the cost model). Server Component: plain SVG, no
 * client JavaScript. Every dot has a <title> with its model and bytes.
 */
import {
  FAMILIES,
  LANDSCAPE_CONTEXT,
  type Family,
  type LandscapePoint,
} from "@/lib/arch/landscape";

const COLOUR: Record<Family, string> = {
  MHA: "#D55E00",
  GQA: "#0072B2",
  MQA: "#56B4E9",
  "sliding window": "#009E73",
  MLA: "#CC79A7",
  "linear or hybrid": "#E69F00",
};

const X0 = 2017;
const X1 = 2027;
const Y0 = Math.log2(1024 / 4); // 256 B
const Y1 = Math.log2(8 * 1024 * 1024); // 8 MiB

const size = (b: number): string =>
  b >= 1024 * 1024
    ? `${(b / 1024 / 1024).toFixed(1)} MiB`
    : b >= 1024
      ? `${(b / 1024).toFixed(0)} KiB`
      : `${b.toFixed(0)} B`;

/** A few dots named on the figure, with where their label sits. */
const LABELS: Record<
  string,
  { text: string; dx: number; dy: number; end?: boolean }
> = {
  "gpt-2-xl": { text: "GPT-2 XL", dx: 7, dy: -7 },
  "llama-2-70b": { text: "Llama 2 70B", dx: -7, dy: -8, end: true },
  "deepseek-v3": { text: "DeepSeek-V3 (MLA)", dx: -8, dy: 16, end: true },
};

/**
 * The figure twice: a wide drawing, and a narrower one with the same data
 * for phones (its labels stay readable); CSS shows one.
 */
export function KvLandscapeSvg({
  points,
}: {
  points: LandscapePoint[];
}): JSX.Element {
  return (
    <>
      <div className="hidden sm:block">
        <Plot points={points} phone={false} />
      </div>
      <div className="sm:hidden">
        <Plot points={points} phone />
      </div>
    </>
  );
}

function Plot({
  points,
  phone,
}: {
  points: LandscapePoint[];
  phone: boolean;
}): JSX.Element {
  const W = phone ? 340 : 640;
  const H = phone ? 300 : 330;
  const M = phone
    ? { l: 52, r: 8, t: 14, b: 30 }
    : { l: 58, r: 12, t: 16, b: 34 };
  const fs = phone ? 11 : 12;
  const x = (yr: number) => M.l + ((yr - X0) / (X1 - X0)) * (W - M.l - M.r);
  const y = (b: number) =>
    M.t +
    (1 - (Math.log2(Math.max(b, 256)) - Y0) / (Y1 - Y0)) * (H - M.t - M.b);
  const cols = phone ? 2 : 3;
  const colW = phone ? 150 : 190;
  const legendH = Math.ceil(FAMILIES.length / cols) * 18 + 6;
  const ticksY = [1024, 16 * 1024, 256 * 1024, 4 * 1024 * 1024];
  return (
    <svg
      viewBox={`0 0 ${W} ${H + legendH}`}
      role="img"
      aria-labelledby={`kv-landscape-title${phone ? "-phone" : ""}`}
      className="h-auto w-full"
      data-kv-landscape={phone ? "phone" : "wide"}
    >
      <title id={`kv-landscape-title${phone ? "-phone" : ""}`}>
        {`KV cache per token at a ${LANDSCAPE_CONTEXT / 1024}K context for ${points.length} models, by release month: multi-head attention models at the top, grouped-query attention in the middle, and MLA, sliding-window and linear or hybrid models lower, mostly since 2024.`}
      </title>
      {ticksY.map((t) => (
        <g key={t}>
          <line
            x1={M.l}
            x2={W - M.r}
            y1={y(t)}
            y2={y(t)}
            className="stroke-neutral-300 dark:stroke-neutral-700"
          />
          <text
            x={M.l - 6}
            y={y(t) + 4}
            fontSize={fs}
            textAnchor="end"
            className="fill-neutral-600 dark:fill-neutral-400"
          >
            {size(t)}
          </text>
        </g>
      ))}
      {(phone ? [2018, 2022, 2026] : [2018, 2020, 2022, 2024, 2026]).map(
        (yr) => (
          <text
            key={yr}
            x={x(yr)}
            y={H - M.b + 18}
            fontSize={fs}
            textAnchor="middle"
            className="fill-neutral-600 dark:fill-neutral-400"
          >
            {yr}
          </text>
        ),
      )}
      {points.map((p) => (
        <circle
          key={p.id}
          data-family={p.family}
          cx={x(p.year)}
          cy={y(p.bytesPerToken)}
          r={phone ? 3.5 : 4.5}
          fill={COLOUR[p.family]}
          fillOpacity={0.85}
          stroke="#000"
          strokeOpacity={0.25}
        >
          <title>{`${p.name}: ${size(p.bytesPerToken)} per token (${p.family})`}</title>
        </circle>
      ))}
      {points
        .filter((p) => LABELS[p.id] && (!phone || p.id !== "llama-2-70b"))
        .map((p) => {
          const l = LABELS[p.id]!;
          return (
            <text
              key={`l-${p.id}`}
              x={x(p.year) + l.dx}
              y={y(p.bytesPerToken) + l.dy}
              fontSize={fs}
              textAnchor={l.end ? "end" : "start"}
              className="fill-neutral-800 dark:fill-neutral-200"
            >
              {l.text}
            </text>
          );
        })}
      {FAMILIES.map((f, k) => {
        const col = k % cols;
        const row = Math.floor(k / cols);
        return (
          <g
            key={f}
            transform={`translate(${(phone ? 8 : M.l) + col * colW}, ${H + 6 + row * 18})`}
          >
            <circle cx={5} cy={-4} r={5} fill={COLOUR[f]} />
            <text
              x={14}
              y={0}
              fontSize={fs}
              className="fill-neutral-700 dark:fill-neutral-300"
            >
              {f}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
