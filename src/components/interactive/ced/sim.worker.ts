/**
 * Runs the vendored simulator's capacity search off the main thread, so the
 * chapter stays responsive while it re-measures a results.md table (up to
 * fifteen bisections of 1,000-request simulations). Receives
 * `{ id, prompt, output, draws, cells }`; posts `{ id, key, rate,
 * simulations }` as each cell finishes, then `{ id, done: true }` (or
 * `{ id, error }`). A newer request supersedes an older one between cells.
 */
import "./worker-global";

import {
  maxSustainableRate,
  type DrawRow,
  type SimConfig,
} from "@/lib/disagg/engine";

type Req = {
  id: number;
  prompt: number;
  output: number;
  draws: DrawRow[];
  cells: { key: string; cfg: SimConfig }[];
};

let latest = -1;

self.onmessage = (e: MessageEvent<Req>) => {
  const { id, prompt, output, draws, cells } = e.data;
  latest = id;
  let i = 0;
  const next = (): void => {
    if (id !== latest) return;
    if (i >= cells.length) {
      self.postMessage({ id, done: true });
      return;
    }
    const c = cells[i++]!;
    try {
      const r = maxSustainableRate(c.cfg, draws, prompt, output);
      self.postMessage({
        id,
        key: c.key,
        rate: r.rate,
        simulations: r.simulations,
      });
    } catch (err) {
      self.postMessage({
        id,
        error: err instanceof Error ? err.message : String(err),
      });
      return;
    }
    // yield between cells so a newer request can arrive
    setTimeout(next, 0);
  };
  next();
};
