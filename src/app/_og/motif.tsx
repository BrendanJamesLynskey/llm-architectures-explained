/**
 * This site's favicon glyph (src/app/icon.svg) and its card's graphic.
 * Glyph: the hexagon of the Inference track, the letter A, #0d9488.
 */
import { params } from "@/lib/arch/costModel";
import { specOf } from "@/lib/arch/features";
import { MODELS } from "@/lib/data";

export function Glyph({ size }: { size: number }): JSX.Element {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32">
      <path d="M16 0.5 29.5 8.25v15.5L16 31.5 2.5 23.75V8.25z" fill="#0d9488" />
      <path
        d="M19.57 23.03 18.55 19.7H13.45L12.43 23.03H9.24L14.18 8.97H17.8L22.76 23.03ZM16.82 13.95Q16.73 13.62 16.57 13.11Q16.41 12.6 16.25 12.07Q16.1 11.54 16 11.15Q15.9 11.54 15.74 12.12Q15.57 12.7 15.42 13.22Q15.27 13.74 15.2 13.95L14.2 17.21H17.84Z"
        fill="#fff"
      />
    </svg>
  );
}

/**
 * How the counts are checked, drawn: every open model whose published weights can be counted, its
 * modelled parameter count against the count of those weights (log scales). The dots sit on the
 * diagonal. From the site's data at build time.
 */
export function Motif(): JSX.Element {
  const pts: [number, number, string][] = [];
  for (const m of MODELS) {
    const hw = m.checks.hf_weight_count;
    const s = specOf(m);
    if (!s || !hw?.v || hw.packed || m.multimodal || m.arch_from) continue;
    pts.push([params(s).total, hw.v, m.kind]);
  }
  const lo =
    Math.log10(Math.min(...pts.map((p) => Math.min(p[0], p[1])))) - 0.1;
  const hi =
    Math.log10(Math.max(...pts.map((p) => Math.max(p[0], p[1])))) + 0.1;
  const t = (v: number) => (Math.log10(v) - lo) / (hi - lo);
  return (
    <svg width="440" height="420" viewBox="0 0 440 420">
      <line
        x1={20}
        y1={400}
        x2={420}
        y2={20}
        stroke="#475569"
        strokeWidth={3}
        strokeDasharray="8 8"
      />
      {pts.map(([a, b, k], i) => (
        <circle
          key={i}
          cx={20 + 400 * t(a)}
          cy={400 - 380 * t(b)}
          r={11}
          fill={k === "decoder" ? "#818cf8" : "#34d399"}
          fillOpacity={0.75}
          stroke="#0b1020"
          strokeWidth={2}
        />
      ))}
    </svg>
  );
}
