/** Fixture readers shared by the CED simulation tests. */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import type { DrawRow, SimConfig } from "@/lib/disagg/engine";

export function readJson<T>(path: string): T {
  return JSON.parse(readFileSync(join(process.cwd(), path), "utf-8")) as T;
}

export type Cell = {
  rate: number;
  text: string;
  ceiling: number;
  cfg: SimConfig;
};
export type ResultsFixture = {
  commit: string;
  s16: {
    params: Record<string, { v: number; line: string; text: string }>;
    steps: Record<
      string,
      {
        line: string;
        cells: Record<
          "base" | "ced" | "enc",
          {
            flops: number;
            time: number;
            bound: string;
            pflop: string;
            ms: string;
            ratio?: string;
          }
        >;
      }
    >;
    room: Record<
      string,
      {
        line: string;
        base: { weights: number; kv_tokens: number | null };
        enc: { weights: number; kv_tokens: number | null };
      }
    >;
  };
  s17: {
    workload: string;
    prompt: number;
    output: number;
    ttft_slo: number;
    slo_text: string;
    rows: Record<
      string,
      { line: string; best: number; cells: Record<string, Cell> }
    >;
  }[];
  s18: {
    workload: string;
    prompt: number;
    output: number;
    ttft_slo: number;
    rows: Record<string, { line: string; cells: Record<string, Cell> }>;
    router_note: string;
  };
  quotes: Record<string, string>;
};

export const results = readJson<ResultsFixture>(
  "tests/unit/fixtures/disagg_ced_results.json",
);

export function draws(
  prompt: number,
  output: number,
): {
  commit: string;
  n: number;
  rows: DrawRow[];
} {
  return readJson(`public/disagg/workloads/ced-${prompt}-${output}.json`);
}
