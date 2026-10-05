/**
 * A small line chart with linear axes in plain SVG (no hooks, no D3), the
 * companion of the log-log LineChart: for profiles along a layer stack.
 */
import type { Series } from "./LineChart";

const W = 640;
const H = 260;
const M = { l: 56, r: 24, t: 12, b: 40 };

function niceTicks(lo: number, hi: number, n = 5): number[] {
  const span = hi - lo || 1;
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step =
    [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step)
    out.push(Number(v.toPrecision(12)));
  return out;
}

export function LinearChart({
  series,
  xLabel,
  yLabel,
  title,
  yMin = 0,
  formatY = (v) => String(v),
}: {
  series: Series[];
  xLabel: string;
  yLabel: string;
  title: string;
  yMin?: number;
  formatY?: (v: number) => string;
}): JSX.Element {
  const pts = series.flatMap((s) => s.points);
  const x0 = Math.min(...pts.map((p) => p[0]));
  const x1 = Math.max(...pts.map((p) => p[0]));
  const y0 = Math.min(yMin, ...pts.map((p) => p[1]));
  const y1 = Math.max(...pts.map((p) => p[1])) * 1.05 || 1;
  const sx = (x: number) => M.l + ((x - x0) / (x1 - x0 || 1)) * (W - M.l - M.r);
  const sy = (y: number) =>
    H - M.b - ((y - y0) / (y1 - y0 || 1)) * (H - M.t - M.b);
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={title}
      className="h-auto w-full min-w-[480px]"
    >
      <title>{title}</title>
      {niceTicks(x0, x1).map((t) => (
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
            {t}
          </text>
        </g>
      ))}
      {niceTicks(y0, y1).map((t) => (
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
      {series.map((s) => (
        <polyline
          key={s.id}
          points={s.points
            .map(([x, y]) => `${sx(x).toFixed(1)},${sy(y).toFixed(1)}`)
            .join(" ")}
          fill="none"
          stroke={s.colour}
          strokeWidth={2}
          strokeDasharray={s.dashed ? "5 4" : undefined}
        >
          <title>{s.label}</title>
        </polyline>
      ))}
    </svg>
  );
}
