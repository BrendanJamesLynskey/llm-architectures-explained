/**
 * Number formatting shared by the pages and the interactives. Bytes use
 * binary units (KiB, MiB, GiB, TiB); parameter and FLOP counts use SI
 * prefixes, as the labs quote them.
 */

function trim(x: number, digits: number): string {
  return Number(x.toPrecision(digits)).toString();
}

/** 235e9 -> "235B", 1.6e12 -> "1.6T", 760e6 -> "760M". */
export function formatCount(n: number | null | undefined, digits = 3): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  const a = Math.abs(n);
  if (a >= 1e12) return `${trim(n / 1e12, digits)}T`;
  if (a >= 1e9) return `${trim(n / 1e9, digits)}B`;
  if (a >= 1e6) return `${trim(n / 1e6, digits)}M`;
  if (a >= 1e3) return `${trim(n / 1e3, digits)}K`;
  return trim(n, digits);
}

const BYTE_UNITS = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"] as const;

/** 1536 -> "1.5 KiB". */
export function formatBytes(n: number | null | undefined, digits = 3): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  let v = n;
  let i = 0;
  while (Math.abs(v) >= 1024 && i < BYTE_UNITS.length - 1) {
    v /= 1024;
    i++;
  }
  return `${trim(v, digits)} ${BYTE_UNITS[i]}`;
}

/** 3.2e12 -> "3.2 TFLOP". */
export function formatFlops(n: number | null | undefined, digits = 3): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  const units: [number, string][] = [
    [1e18, "EFLOP"],
    [1e15, "PFLOP"],
    [1e12, "TFLOP"],
    [1e9, "GFLOP"],
    [1e6, "MFLOP"],
  ];
  for (const [s, u] of units) {
    if (Math.abs(n) >= s) return `${trim(n / s, digits)} ${u}`;
  }
  return `${trim(n, digits)} FLOP`;
}

/** 131072 -> "128K", 1048576 -> "1M" (context lengths, powers of two). */
export function formatTokens(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  if (n >= 1048576 && n % 1048576 === 0) return `${n / 1048576}M`;
  if (n >= 1024 && n % 1024 === 0) return `${n / 1024}K`;
  return n.toLocaleString("en-GB");
}
