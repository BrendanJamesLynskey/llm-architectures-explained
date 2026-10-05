/**
 * A legend for the charts: one swatch per series, on its own background
 * box (dashed swatches for dashed series). Server or client.
 */
import type { Series } from "./LineChart";

export function Legend({ series }: { series: Series[] }): JSX.Element {
  return (
    <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 rounded bg-white px-3 py-2 text-xs ring-1 ring-neutral-200 dark:bg-neutral-950 dark:ring-neutral-800">
      {series.map((s) => (
        <li
          key={s.id}
          className="flex items-center gap-1.5 text-neutral-700 dark:text-neutral-300"
        >
          <svg aria-hidden width="18" height="8">
            <line
              x1="1"
              x2="17"
              y1="4"
              y2="4"
              stroke={s.colour}
              strokeWidth="2.5"
              strokeDasharray={s.dashed ? "4 3" : undefined}
            />
          </svg>
          {s.label}
        </li>
      ))}
    </ul>
  );
}

/** A keyboard-focusable box that scrolls a wide chart or table on a phone. */
export function ScrollBox({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}): JSX.Element {
  return (
    <div
      tabIndex={0}
      role="region"
      aria-label={label}
      className="focus-ring mt-3 overflow-x-auto rounded"
    >
      {children}
    </div>
  );
}
