/**
 * A model's architecture drawn from its data: the layer stack (one column
 * per layer, token mixer on top, feed-forward block below) and a schematic
 * of each distinct block, with norms placed as the record says.
 *
 * Pure SVG with no hooks, so it renders on the server (model pages) and in
 * the browser (compare view) alike. Nothing here is drawn by hand: change
 * the data and the diagram follows.
 */
import type { Ffn, LayerRun, Mixer, Spec } from "@/lib/arch/costModel";
import { expandLayout } from "@/lib/arch/costModel";
import { MIXER_NAMES } from "@/lib/arch/features";
import { formatCount } from "@/lib/arch/format";

export const MIXER_COLOURS: Record<string, string> = {
  full: "#6366f1",
  sliding: "#0ea5e9",
  chunked: "#06b6d4",
  mla: "#8b5cf6",
  csa: "#d946ef",
  deltanet: "#10b981",
  kda: "#10b981",
  linear: "#14b8a6",
  mamba1: "#f59e0b",
  mamba2: "#f59e0b",
  conv: "#84cc16",
  rwkv: "#f97316",
  mlstm: "#f43f5e",
  sparse: "#a855f7",
};
const FFN_COLOURS: Record<string, string> = {
  dense: "#64748b",
  moe: "#ec4899",
};

/** The colour key for a mixer: windowed and sparse attention get their own. */
export function mixerKey(m: Mixer): string {
  if (m.type === "attn") {
    if (m.indexer) return "sparse";
    if (m.chunk) return "chunked";
    return m.window ? "sliding" : "full";
  }
  return m.type;
}

export function mixerLabel(m: Mixer): string {
  if (m.type === "attn") {
    const kind =
      m.kv_heads === 1 ? "MQA" : m.kv_heads! < m.heads! ? "GQA" : "MHA";
    const parts = [`${kind}: ${m.heads} query / ${m.kv_heads} KV heads`];
    parts.push(
      `head ${m.head_dim}${m.v_head_dim && m.v_head_dim !== m.head_dim ? ` (V ${m.v_head_dim})` : ""}`,
    );
    if (m.window) parts.push(`window ${m.window.toLocaleString("en-GB")}`);
    if (m.chunk) parts.push(`chunks of ${m.chunk.toLocaleString("en-GB")}`);
    if (m.indexer) parts.push(`sparse top-${m.indexer.topk}`);
    if (m.gate) parts.push(`${m.gate} output gate`);
    if (m.k_eq_v) parts.push("K = V");
    return parts.join(" · ");
  }
  if (m.type === "mla") {
    const parts = [
      `MLA: ${m.heads} heads`,
      `KV latent ${m.kv_lora_rank} + RoPE ${m.qk_rope}`,
    ];
    if (m.q_lora_rank) parts.push(`Q latent ${m.q_lora_rank}`);
    if (m.indexer) parts.push(`sparse top-${m.indexer.topk}`);
    if (m.window) parts.push(`window ${m.window}`);
    return parts.join(" · ");
  }
  if (m.type === "csa") {
    return `Compressed attention: ${m.heads} heads · latent ${m.head_dim} · window ${m.window} · indexer top-${m.indexer?.topk ?? "—"}`;
  }
  if (m.type === "deltanet" || m.type === "kda") {
    return `${MIXER_NAMES[m.type]}: ${m.v_heads} heads · state ${m.k_head_dim}×${m.v_head_dim} per head`;
  }
  if (m.type === "mamba2") {
    return `Mamba-2: ${m.heads} heads × ${m.head_dim} · state ${m.state}`;
  }
  if (m.type === "mamba1")
    return `Mamba: inner ${m.d_inner} · state ${m.state}`;
  if (m.type === "conv") return `Gated short convolution · kernel ${m.kernel}`;
  if (m.type === "linear")
    return `Linear attention: ${m.heads} heads × ${m.head_dim}`;
  if (m.type === "rwkv") return `RWKV time mixing · head ${m.head_dim}`;
  if (m.type === "mlstm") return `mLSTM: ${m.heads} heads`;
  return m.type;
}

export function ffnLabel(f: Ffn): string {
  if (f.type === "moe") {
    const parts = [
      `MoE: ${f.experts} experts, ${f.active} active`,
      `expert ${f.d_expert}`,
    ];
    if (f.shared) parts.push(`${f.shared} shared`);
    if (f.latent) parts.push(`latent ${f.latent}`);
    if (f.dense_parallel_d_ff) parts.push(`+ dense ${f.dense_parallel_d_ff}`);
    return parts.join(" · ");
  }
  return `${f.gated === false ? "MLP" : "Gated MLP"}: ${f.d_ff}`;
}

type Placement = "pre" | "post" | "sandwich" | "parallel" | "post-ln" | string;

const W = 720;

function Stack({
  spec,
  runs,
  y,
  label,
  split,
}: {
  spec: Spec;
  runs: LayerRun[];
  y: number;
  label: string;
  split?: number;
}): JSX.Element {
  const n = runs.length;
  const x0 = 70;
  const width = W - x0 - 10;
  const step = width / n;
  const colW = Math.max(step - (step > 4 ? 1 : 0), 0.6);
  return (
    <g>
      <text
        x={0}
        y={y + 14}
        className="fill-neutral-700 text-[11px] dark:fill-neutral-300"
      >
        {label}
      </text>
      <text
        x={0}
        y={y + 40}
        className="fill-neutral-500 text-[10px] dark:fill-neutral-400"
      >
        mixer / FFN
      </text>
      {runs.map((r, i) => {
        const m = r.mixer === "none" ? null : spec.mixers[r.mixer]!;
        const f = r.ffn === "none" ? null : spec.ffns[r.ffn]!;
        const x = x0 + i * step;
        return (
          <g key={i}>
            <rect
              x={x}
              y={y}
              width={colW}
              height={22}
              fill={m ? MIXER_COLOURS[mixerKey(m)] : "transparent"}
              opacity={r.kv_shared || r.kv_source === false ? 0.45 : 1}
            >
              <title>{`layer ${i}: ${m ? mixerLabel(m) : "no mixer"}${r.kv_shared || r.kv_source === false ? " (reuses another layer's KV)" : ""}${r.ratio ? ` · compression ${r.ratio}` : ""}`}</title>
            </rect>
            <rect
              x={x}
              y={y + 26}
              width={colW}
              height={14}
              fill={f ? FFN_COLOURS[f.type] : "transparent"}
            >
              <title>{`layer ${i}: ${f ? ffnLabel(f) : "no FFN"}`}</title>
            </rect>
          </g>
        );
      })}
      {[0, Math.floor(n / 2), n - 1].map((i) => (
        <text
          key={i}
          x={x0 + i * step + colW / 2}
          y={y + 54}
          textAnchor="middle"
          className="fill-neutral-500 text-[10px] dark:fill-neutral-400"
        >
          {i}
        </text>
      ))}
      {split !== undefined && (
        <g>
          <line
            x1={x0 + split * step - 0.5}
            x2={x0 + split * step - 0.5}
            y1={y - 6}
            y2={y + 44}
            className="stroke-neutral-900 dark:stroke-neutral-100"
            strokeWidth={1.5}
            strokeDasharray="3 2"
          />
          <text
            x={x0 + (split * step) / 2}
            y={y - 9}
            textAnchor="middle"
            className="fill-neutral-700 text-[10px] dark:fill-neutral-300"
          >
            encoder: runs at prefill
          </text>
          <text
            x={x0 + split * step + ((n - split) * step) / 2}
            y={y - 9}
            textAnchor="middle"
            className="fill-neutral-700 text-[10px] dark:fill-neutral-300"
          >
            decoder: replays the last tokens
          </text>
        </g>
      )}
    </g>
  );
}

function Box({
  x,
  y,
  w,
  h,
  fill,
  text,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  fill: string;
  text: string;
}): JSX.Element {
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={5}
        fill={fill}
        opacity={0.18}
      />
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={5}
        fill="none"
        stroke={fill}
        strokeWidth={1.5}
      />
      <text
        x={x + 8}
        y={y + h / 2 + 4}
        className="fill-neutral-900 text-[11px] dark:fill-neutral-100"
      >
        {text}
      </text>
    </g>
  );
}

function Norm({ x, y }: { x: number; y: number }): JSX.Element {
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={44}
        height={16}
        rx={3}
        className="fill-neutral-200 dark:fill-neutral-700"
      />
      <text
        x={x + 22}
        y={y + 12}
        textAnchor="middle"
        className="fill-neutral-700 text-[9px] dark:fill-neutral-200"
      >
        norm
      </text>
    </g>
  );
}

/** One block: residual stream on the left, sub-blocks to its right. */
function Block({
  spec,
  run,
  y,
  placement,
  count,
}: {
  spec: Spec;
  run: LayerRun;
  y: number;
  placement: Placement;
  count: number;
}): JSX.Element {
  const m = run.mixer === "none" ? null : spec.mixers[run.mixer]!;
  const f = run.ffn === "none" ? null : spec.ffns[run.ffn]!;
  const rx = 14;
  const subs = [
    m && { text: mixerLabel(m), fill: MIXER_COLOURS[mixerKey(m)]! },
    f && { text: ffnLabel(f), fill: FFN_COLOURS[f.type]! },
  ].filter(Boolean) as { text: string; fill: string }[];
  const rowH = 30;
  const h =
    placement === "parallel" ? rowH + 26 : subs.length * (rowH + 8) + 18;
  const pre = placement === "pre" || placement === "sandwich";
  const post =
    placement === "post" || placement === "sandwich" || placement === "post-ln";
  return (
    <g>
      <text
        x={0}
        y={y + 10}
        className="fill-neutral-600 text-[10px] dark:fill-neutral-400"
      >
        {`× ${count}`}
      </text>
      <line
        x1={rx + 30}
        x2={rx + 30}
        y1={y}
        y2={y + h}
        className="stroke-neutral-400"
        strokeWidth={2}
      />
      {placement === "parallel" ? (
        <g>
          <Norm x={rx + 52} y={y + 18} />
          {subs.map((s, i) => (
            <Box
              key={i}
              x={rx + 104 + i * 290}
              y={y + 12}
              w={280}
              h={rowH}
              fill={s.fill}
              text={s.text}
            />
          ))}
          <text
            x={rx + 30}
            y={y + h - 2}
            textAnchor="middle"
            className="fill-neutral-600 text-[10px] dark:fill-neutral-300"
          >
            +
          </text>
        </g>
      ) : (
        subs.map((s, i) => {
          const yy = y + 8 + i * (rowH + 8);
          let x = rx + 52;
          const parts: JSX.Element[] = [];
          if (pre) {
            parts.push(<Norm key="pre" x={x} y={yy + 7} />);
            x += 52;
          }
          const bw = W - x - (post ? 60 : 8);
          parts.push(
            <Box
              key="box"
              x={x}
              y={yy}
              w={bw}
              h={rowH}
              fill={s.fill}
              text={s.text}
            />,
          );
          if (post) parts.push(<Norm key="post" x={x + bw + 8} y={yy + 7} />);
          return (
            <g key={i}>
              {parts}
              <circle
                cx={rx + 30}
                cy={yy + rowH + 3}
                r={5}
                className="fill-white stroke-neutral-500 dark:fill-neutral-950"
              />
              <text
                x={rx + 30}
                y={yy + rowH + 6.5}
                textAnchor="middle"
                className="fill-neutral-600 text-[9px] dark:fill-neutral-300"
              >
                +
              </text>
            </g>
          );
        })
      )}
    </g>
  );
}

export type DiagramProps = {
  spec: Spec;
  placement?: Placement;
  title: string;
};

function blockHeight(placement: Placement, subs: number): number {
  return placement === "parallel" ? 30 + 26 : subs * 38 + 18;
}

export function BlockDiagram({
  spec,
  placement = "pre",
  title,
}: DiagramProps): JSX.Element {
  const runs = expandLayout(spec);
  const enc =
    spec.kind === "encoder-decoder" ? expandLayout(spec, "encoder_layout") : [];
  // Distinct blocks, most common first.
  const kinds = new Map<string, { run: LayerRun; count: number }>();
  for (const r of runs) {
    const k = `${r.mixer}|${r.ffn}`;
    const e = kinds.get(k);
    if (e) e.count++;
    else kinds.set(k, { run: r, count: 1 });
  }
  const blocks = [...kinds.values()].sort((a, b) => b.count - a.count);
  let y = 16;
  const parts: JSX.Element[] = [];
  if (enc.length) {
    parts.push(
      <Stack
        key="enc"
        spec={spec}
        runs={enc}
        y={y}
        label={`encoder (${enc.length})`}
      />,
    );
    y += 66;
  }
  parts.push(
    <Stack
      key="dec"
      spec={spec}
      runs={runs}
      y={y + (spec.kind === "ced" ? 12 : 0)}
      label={`${enc.length ? "decoder" : "layers"} (${runs.length})`}
      split={spec.kind === "ced" ? spec.ced_encoder_layers : undefined}
    />,
  );
  y += 70 + (spec.kind === "ced" ? 12 : 0);
  for (const [i, b] of blocks.entries()) {
    const subs =
      (b.run.mixer !== "none" ? 1 : 0) + (b.run.ffn !== "none" ? 1 : 0);
    parts.push(
      <Block
        key={`b${i}`}
        spec={spec}
        run={b.run}
        y={y}
        placement={placement}
        count={b.count}
      />,
    );
    y += blockHeight(placement, subs) + 12;
  }
  const mixers = [
    ...new Set(
      runs
        .filter((r) => r.mixer !== "none")
        .map((r) => mixerKey(spec.mixers[r.mixer]!)),
    ),
  ];
  const ffns = [
    ...new Set(
      runs.filter((r) => r.ffn !== "none").map((r) => spec.ffns[r.ffn]!.type),
    ),
  ];
  const legend = [
    ...mixers.map((k) => ({ k, c: MIXER_COLOURS[k]!, l: LEGEND[k] ?? k })),
    ...ffns.map((k) => ({
      k: `f${k}`,
      c: FFN_COLOURS[k]!,
      l: k === "moe" ? "MoE FFN" : "dense FFN",
    })),
  ];
  const ly = y + 4;
  const rows = Math.ceil(legend.length / 4);
  return (
    <svg
      viewBox={`0 0 ${W} ${ly + rows * 18 + 6}`}
      role="img"
      aria-label={title}
      className="h-auto w-full min-w-[640px]"
    >
      <title>{title}</title>
      {parts}
      {legend.map((e, i) => (
        <g
          key={e.k}
          transform={`translate(${(i % 4) * 178}, ${ly + Math.floor(i / 4) * 18})`}
        >
          <rect x={0} y={0} width={12} height={12} rx={2} fill={e.c} />
          <text
            x={17}
            y={10}
            className="fill-neutral-700 text-[11px] dark:fill-neutral-300"
          >
            {e.l}
          </text>
        </g>
      ))}
    </svg>
  );
}

const LEGEND: Record<string, string> = {
  full: "full attention",
  sliding: "sliding window",
  chunked: "chunked attention",
  sparse: "sparse attention",
  mla: "MLA",
  csa: "compressed attention",
  deltanet: "Gated DeltaNet",
  kda: "Kimi Delta Attention",
  linear: "linear attention",
  mamba1: "Mamba",
  mamba2: "Mamba-2",
  conv: "short conv",
  rwkv: "RWKV",
  mlstm: "mLSTM",
};

/** Parameter breakdown line used under the diagram. */
export function paramLine(total: number, active: number): string {
  return total === active
    ? `${formatCount(total)} parameters (modelled)`
    : `${formatCount(total)} parameters, ${formatCount(active)} active per token (modelled)`;
}
