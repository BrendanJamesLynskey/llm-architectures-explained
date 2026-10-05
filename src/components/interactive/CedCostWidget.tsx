"use client";

/**
 * Chapter 9: what encoder-only prefill saves on the real model. DeepSeek
 * V4.1-Flash's sourced architecture, costed by this site's cost model as
 * published (a causal encoder-decoder: a prompt token runs the encoder, the
 * decoder's KV projection, and only the last W tokens replay the decoder)
 * and as the same stack run as an ordinary decoder (asDecoder, checked
 * against reference/chapter_model.py).
 */
import { useState } from "react";

import { Slider, Stat } from "@/components/ui/Controls";
import { WidgetFrame } from "@/components/ui/WidgetFrame";
import { decodeFlops, params, prefillFlops } from "@/lib/arch/costModel";
import { formatCount, formatFlops, formatTokens } from "@/lib/arch/format";
import { CHAPTERS } from "@/lib/chapters/data";
import { asDecoder } from "@/lib/chapters/model";

export default function CedCostWidget(): JSX.Element {
  const [log2n, setLog2n] = useState(15);
  const n = 2 ** log2n;
  const ced = CHAPTERS.ced.spec;
  const dec = asDecoder(ced);
  const pc = prefillFlops(ced, n);
  const pd = prefillFlops(dec, n);
  const p = params(ced);

  return (
    <WidgetFrame
      testId="ced-cost-widget"
      title="DeepSeek-V4.1-Flash's prefill, with and without the split"
      caption={`Its ${ced.layout.reduce((k, r) => k + r.n, 0)} layers: the first ${ced.ced_encoder_layers} are the causal encoder; the decoder replays the last ${ced.ced_window} prompt tokens. Cost model conventions as on /about.`}
    >
      <div className="max-w-md">
        <Slider
          label="Prompt length"
          value={log2n}
          min={10}
          max={20}
          onChange={setLog2n}
          format={(v) => `${formatTokens(2 ** v)} tokens`}
        />
      </div>
      <div
        className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4"
        data-testid="ced-cost-stats"
      >
        <Stat label="Prefill, as published (CED)" value={formatFlops(pc)} />
        <Stat
          label="Prefill, same stack as a decoder"
          value={formatFlops(pd)}
        />
        <Stat label="CED / decoder" value={(pc / pd).toFixed(3)} />
        <Stat
          label="Decode FLOPs / token"
          value={formatFlops(decodeFlops(ced, n))}
          hint="the same either way: decode runs every layer"
        />
      </div>
      <p className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
        Modelled parameters: {formatCount(p.total)} total,{" "}
        {formatCount(p.active)} active per decoded token. The paper&apos;s own
        figures are 8B activated per token at prefill and 16B at decode.
      </p>
    </WidgetFrame>
  );
}
