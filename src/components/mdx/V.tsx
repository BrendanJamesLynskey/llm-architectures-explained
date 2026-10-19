/**
 * <V v="ch01.gqa128k" fmt="gib" />: one of the learning layer's model values
 * (src/lib/learn/values.ts), formatted, in a self-check's reveal or a
 * "What to remember" box. Server Component.
 */
import { formatValue, learnValue, type Fmt } from "@/lib/learn/values";

export function V({ v, fmt = "num" }: { v: string; fmt?: Fmt }): JSX.Element {
  return <span data-learn-value={v}>{formatValue(learnValue(v), fmt)}</span>;
}
