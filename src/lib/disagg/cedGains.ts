/**
 * The landing page's CED line (brief 41): the capacity gain of the causal
 * encoder-decoder over the decoder-only model, in the simulator's capacity
 * search (results.md §17, recorded in public/disagg/ced-results.json):
 * the best split's sustainable rate with CED over the decoder-only model's,
 * for each workload and each place the replay runs.
 */
import raw from "../../../public/disagg/ced-results.json";

type Cell = { rate: number; text: string };
type Row = { cells: Record<string, Cell> };
type Workload = { workload: string; rows: Record<string, Row> };

const best = (r: Row): number =>
  Math.max(...Object.values(r.cells).map((c) => c.rate));

export type CedGain = { workload: string; variant: string; gain: number };

export function cedGains(): CedGain[] {
  const out: CedGain[] = [];
  for (const w of (raw as unknown as { s17: Workload[] }).s17) {
    const base = best(w.rows["decoder-only"]!);
    for (const [k, r] of Object.entries(w.rows))
      if (k !== "decoder-only")
        out.push({ workload: w.workload, variant: k, gain: best(r) / base });
  }
  return out;
}

/** The smallest and largest gain, as results.md prints them (two decimals). */
export function cedRange(): { lo: string; hi: string } {
  const g = cedGains().map((x) => x.gain);
  return {
    lo: Math.min(...g).toFixed(2),
    hi: Math.max(...g).toFixed(2),
  };
}
