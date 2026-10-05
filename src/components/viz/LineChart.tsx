/**
 * A small log-log line chart in plain SVG (no hooks, no D3), so it renders
 * on the server and in the browser. Used for KV cache against context
 * length. Series with `dashed` are drawn dashed (e.g. reported estimates).
 */

export type Series = {
  id: string;
  label: string;
  colour: string;
  points: [number, number][];
  dashed?: boolean;
};

const W = 640;
const H = 300;
const M = { l: 64, r: 36, t: 12, b: 40 };

function logTicks(lo: number, hi: number): number[] {
  const out: number[] = [];
  for (
    let e = Math.floor(Math.log10(lo));
    e <= Math.ceil(Math.log10(hi));
    e++
  ) {
    const v = 10 ** e;
    if (v >= lo / 1.0001 && v <= hi * 1.0001) out.push(v);
  }
  return out;
}

export function LineChart({
  series,
  xLabel,
  yLabel,
  formatX,
  formatY,
  title,
  yTicks = logTicks,
}: {
  series: Series[];
  xLabel: string;
  yLabel: string;
  formatX: (v: number) => string;
  formatY: (v: number) => string;
  title: string;
  /** Tick positions on the y axis (default: powers of ten). */
  yTicks?: (lo: number, hi: number) => number[];
}): JSX.Element {
  const pts = series
    .flatMap((s) => s.points)
    .filter(([x, y]) => x > 0 && y > 0);
  if (!pts.length) {
    return (
      <p className="text-sm text-neutral-600 dark:text-neutral-400">
        No data to plot.
      </p>
    );
  }
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  let y0 = Math.min(...ys);
  let y1 = Math.max(...ys);
  if (y0 === y1) {
    y0 /= 2;
    y1 *= 2;
  }
  const lx0 = Math.log10(x0);
  const lx1 = Math.log10(x1);
  const ly0 = Math.log10(y0) - 0.05;
  const ly1 = Math.log10(y1) + 0.05;
  const sx = (x: number) =>
    M.l + ((Math.log10(x) - lx0) / (lx1 - lx0 || 1)) * (W - M.l - M.r);
  const sy = (y: number) =>
    H - M.b - ((Math.log10(y) - ly0) / (ly1 - ly0)) * (H - M.t - M.b);
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={title}
      className="h-auto w-full min-w-[480px]"
    >
      <title>{title}</title>
      {logTicks(x0, x1).map((t) => (
        <g key={`x${t}`}>
          <line
            x1={sx(t)}
            x2={sx(t)}
            y1={M.t}
            y2={H - M.b}
            className="stroke-neutral-200 dark:stroke-neutral-800"
          />
          <text
            x={sx(t)}
            y={H - M.b + 14}
            textAnchor="middle"
            className="fill-neutral-600 text-[11px] dark:fill-neutral-400"
          >
            {formatX(t)}
          </text>
        </g>
      ))}
      {yTicks(10 ** ly0, 10 ** ly1).map((t) => (
        <g key={`y${t}`}>
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
            {formatY(t)}
          </text>
        </g>
      ))}
      <text
        x={(W + M.l) / 2}
        y={H - 6}
        textAnchor="middle"
        className="fill-neutral-700 text-[11px] dark:fill-neutral-300"
      >
        {xLabel}
      </text>
      <text
        x={12}
        y={(H - M.b) / 2}
        transform={`rotate(-90 12 ${(H - M.b) / 2})`}
        textAnchor="middle"
        className="fill-neutral-700 text-[11px] dark:fill-neutral-300"
      >
        {yLabel}
      </text>
      {series.map((s) => {
        const p = s.points.filter(([x, y]) => x > 0 && y > 0);
        return (
          <g key={s.id}>
            <polyline
              points={p
                .map(([x, y]) => `${sx(x).toFixed(1)},${sy(y).toFixed(1)}`)
                .join(" ")}
              fill="none"
              stroke={s.colour}
              strokeWidth={2}
              strokeDasharray={s.dashed ? "5 4" : undefined}
            >
              <title>{s.label}</title>
            </polyline>
            {p.length > 0 && (
              <circle
                cx={sx(p[p.length - 1]![0])}
                cy={sy(p[p.length - 1]![1])}
                r={3}
                fill={s.colour}
              />
            )}
          </g>
        );
      })}
    </svg>
  );
}

/** Powers of two between lo and hi, at most about six of them: byte axes. */
export function binaryTicks(lo: number, hi: number): number[] {
  const a = Math.ceil(Math.log2(lo));
  const b = Math.floor(Math.log2(hi));
  const step = Math.max(1, Math.ceil((b - a + 1) / 6));
  const out: number[] = [];
  for (let e = b; e >= a; e -= step) out.push(2 ** e);
  return out.reverse();
}

/** Colours for up to four compared models (distinct in light and dark). */
export const SERIES_COLOURS = [
  "#4f46e5",
  "#e11d48",
  "#059669",
  "#d97706",
] as const;

/** Colours for up to eight series (the chapter interactives); the first four are SERIES_COLOURS. */
export const PALETTE = [
  ...SERIES_COLOURS,
  "#0891b2",
  "#9333ea",
  "#65a30d",
  "#78716c",
] as const;
