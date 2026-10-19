import type { Route } from "next";
import Link from "next/link";

import { CheckedStrip, StartButton } from "@/components/ui/Landing";
import { KvLandscapeSvg } from "@/components/viz/KvLandscapeSvg";
import { LANDSCAPE_CONTEXT, kvLandscape } from "@/lib/arch/landscape";
import { cedRange } from "@/lib/disagg/cedGains";
import { SECTIONS } from "@/lib/mdx/sections";
import { stats, weightChecks } from "@/lib/stats";

/**
 * Landing page: what the site is, one figure built from all of its data
 * (brief 41: every model's KV cache per token by release month), the site's
 * own CED result, and the ways in. Server Component with no client
 * JavaScript of its own.
 */
export default function HomePage(): JSX.Element {
  const s = stats();
  const w = weightChecks();
  const points = kvLandscape();
  const ced = cedRange();
  const ENTRY = [
    {
      href: "/learn",
      title: "Nine ways models differ",
      blurb:
        "One chapter per axis of variation, from attention to the causal encoder-decoder, each with a live interactive and the models that use it.",
      cta: "Start learning →",
    },
    {
      href: "/models",
      title: "Every model",
      blurb: `${s.models} models in one table: size, depth, width, attention, KV cache per token. Filter by any design choice.`,
      cta: "Open the table →",
    },
    {
      href: "/compare",
      title: "Compare and calculate",
      blurb:
        "Pick up to four models: diagrams drawn from their data, KV cache against context, prefill and decode cost.",
      cta: "Compare models →",
    },
    {
      href: "/timeline",
      title: "How the variations spread",
      blurb:
        "From the 2017 Transformer to this year: when GQA, MoE, MLA, sliding windows and linear attention took hold.",
      cta: "See the timeline →",
    },
  ] as const;
  return (
    <main className="mx-auto max-w-5xl px-6 py-12 sm:py-20">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        LLM Architectures Explained
      </p>
      <h1 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
        How language-model architectures differ, model by model.
      </h1>
      <div className="mt-6">
        <p className="max-w-2xl text-lg text-neutral-600 dark:text-neutral-300">
          An interactive guide built from the recorded design choices of{" "}
          {s.models} models, every value traced to its source, with a tested
          cost model.
        </p>
        <div className="mt-6">
          <StartButton href={`/learn/${SECTIONS[0]!.slug}` as Route} />
        </div>
      </div>
      <figure
        data-hero-figure
        className="mt-8 rounded-lg border border-neutral-200 p-4 dark:border-neutral-800"
      >
        <p className="mb-2 text-sm font-medium text-neutral-800 dark:text-neutral-200">
          KV cache per token at a {LANDSCAPE_CONTEXT / 1024}K context,{" "}
          {points.length} models
        </p>
        <KvLandscapeSvg points={points} />
        <figcaption className="mt-3 text-sm text-neutral-600 dark:text-neutral-400">
          Each dot is a model, placed by its release month and what one token of
          its context costs to cache in BF16, computed from its recorded
          configuration by the site&rsquo;s cost model. Grouped-query attention
          cut it several-fold; MLA, sliding windows and linear-attention hybrids
          cut it again.{" "}
          <Link
            href="/learn/01-attention"
            className="focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300"
          >
            Why, chapter 1
          </Link>
        </figcaption>
      </figure>
      <p
        data-ced-result
        className="mt-4 max-w-3xl rounded-lg bg-neutral-50 p-4 text-sm text-neutral-700 dark:bg-neutral-900 dark:text-neutral-300"
      >
        <strong>Original result.</strong> A causal encoder-decoder, whose prompt
        tokens touch about half the parameters, serves{" "}
        {`${ced.lo}× to ${ced.hi}×`}
        the load of the same-sized decoder-only model at the same SLOs, in this
        site&rsquo;s run of a disaggregated-serving simulator&rsquo;s capacity
        search (four prompt-to-output mixes, six 4×H100 instances).{" "}
        <Link
          href="/learn/09-encoder-decoder-and-ced"
          className="focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300"
        >
          Chapter 9
        </Link>{" "}
        reruns the search in your browser.
      </p>
      <CheckedStrip href="/about#checks">
        Every value links to its source, and the modelled parameter count of{" "}
        {w.within} of the {w.compared} open models whose published weights can
        be counted is within 0.5% of those weights at a pinned commit.
      </CheckedStrip>
      <section aria-labelledby="learn-heading" className="mt-12">
        <h2 id="learn-heading" className="text-xl font-semibold tracking-tight">
          What you&apos;ll learn
        </h2>
        <p className="mt-4 max-w-3xl text-neutral-600 dark:text-neutral-300">
          Every large language model is a stack of the same few parts, chosen
          differently: how attention shares its keys and values, how positions
          are encoded, where the norms sit, whether the feed-forward block is
          one network or many experts, how deep and how wide. This site records
          those choices for {s.models} models, each value traced to the
          model&rsquo;s own configuration, paper or announcement, and computes
          what they cost.
        </p>
      </section>
      <nav
        aria-label="Ways in"
        className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-4"
      >
        {ENTRY.map((e) => (
          <Link
            key={e.href}
            href={e.href}
            className="focus-ring group rounded-lg border border-neutral-200 p-5 hover:border-accent dark:border-neutral-800 dark:hover:border-indigo-400"
          >
            <h2 className="font-semibold">{e.title}</h2>
            <p className="mt-2 text-sm text-neutral-600 dark:text-neutral-400">
              {e.blurb}
            </p>
            <p className="mt-4 text-sm font-medium text-accent dark:text-indigo-300">
              {e.cta}
            </p>
          </Link>
        ))}
      </nav>
      <p
        data-track-start
        className="mt-12 text-sm text-neutral-600 dark:text-neutral-400"
      >
        New to the Inference track?{" "}
        <a
          href="https://llm-inference-explained.vercel.app/inference"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Start here
        </a>
        : the four sites in reading order, what you need first, and a glossary
        of the terms they define.
      </p>
      <p className="mt-12 text-sm text-neutral-600 dark:text-neutral-400">
        Part of a family of sixteen companion sites in four tracks, all in the
        site menu above: Foundations (
        <a
          href="https://transformer-decoder-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Decoder
        </a>
        ,{" "}
        <a
          href="https://transformer-training-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Training
        </a>
        ), Hardware (
        <a
          href="https://gpu-kernels-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Kernels
        </a>
        ,{" "}
        <a
          href="https://systolic-arrays-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Systolic
        </a>
        ,{" "}
        <a
          href="https://ai-accelerators-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Accelerators
        </a>
        ,{" "}
        <a
          href="https://ai-racks-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Racks
        </a>
        ), Inference (
        <a
          href="https://llm-inference-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Serving
        </a>
        , <strong className="font-medium">Architectures</strong>,{" "}
        <a
          href="https://numerics-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Numerics
        </a>
        ,{" "}
        <a
          href="https://inference-tradeoffs-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Trade-offs
        </a>
        ) and Agents (
        <a
          href="https://agent-harnesses-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Harnesses
        </a>
        ,{" "}
        <a
          href="https://agent-protocols-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Protocols
        </a>
        ,{" "}
        <a
          href="https://agent-context-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Context
        </a>
        ,{" "}
        <a
          href="https://agent-orchestration-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Orchestration
        </a>
        ,{" "}
        <a
          href="https://agent-evals-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Evals
        </a>
        ,{" "}
        <a
          href="https://agent-security-explained.vercel.app"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Security
        </a>
        ). This one is about how the models differ. How this site was built, and
        how to check its data:{" "}
        <Link
          href="/about"
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          about
        </Link>
        .
      </p>
    </main>
  );
}
