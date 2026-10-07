import Link from "next/link";

import { stats } from "@/lib/stats";
import {
  DECODER_URL,
  INFERENCE_URL,
  KERNELS_URL,
  NUMERICS_URL,
  SILICON_URL,
  TRADEOFFS_URL,
} from "@/lib/site";

/**
 * Landing page: what the site is, how much it covers, and the ways in.
 * Server Component with no client JavaScript of its own.
 */
export default function HomePage(): JSX.Element {
  const s = stats();
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
      <p className="mt-6 max-w-3xl text-lg text-neutral-600 dark:text-neutral-300">
        Every large language model is a stack of the same few parts, chosen
        differently: how attention shares its keys and values, how positions are
        encoded, where the norms sit, whether the feed-forward block is one
        network or many experts, how deep and how wide. This site records those
        choices for {s.models} models, each value traced to the model&rsquo;s
        own configuration, paper or announcement, and computes what they cost.
      </p>
      <dl
        className="mt-10 grid grid-cols-2 gap-4 sm:grid-cols-4"
        data-testid="stats"
      >
        {[
          [String(s.models), "models"],
          [`${s.gallery}/${s.gallery}`, "gallery checklist names covered"],
          [String(s.pinned), "configs pinned to a Hugging Face commit"],
          [
            String(s.closed),
            "closed models, sizes not disclosed unless stated",
          ],
        ].map(([v, l]) => (
          <div
            key={l}
            className="rounded-lg border border-neutral-200 p-4 dark:border-neutral-800"
          >
            <dt className="text-xs text-neutral-600 dark:text-neutral-400">
              {l}
            </dt>
            <dd className="mt-1 font-mono text-2xl">{v}</dd>
          </div>
        ))}
      </dl>
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
      <p className="mt-12 text-sm text-neutral-600 dark:text-neutral-400">
        Part of a family of companion sites: the{" "}
        <a
          href={DECODER_URL}
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Transformer Decoder Explainer
        </a>{" "}
        (one forward pass),{" "}
        <a
          href={INFERENCE_URL}
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          LLM Inference Explained
        </a>{" "}
        (serving it),{" "}
        <a
          href={KERNELS_URL}
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          GPU Kernels Explained
        </a>{" "}
        (how a GPU executes it),{" "}
        <a
          href={NUMERICS_URL}
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Numerics Explained
        </a>{" "}
        (the number formats it runs in),{" "}
        <a
          href={SILICON_URL}
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Systolic Arrays Explained
        </a>{" "}
        (the matrix hardware of TPUs) and{" "}
        <a
          href={TRADEOFFS_URL}
          className="focus-ring rounded underline decoration-accent/40 underline-offset-4 hover:decoration-accent"
        >
          Inference Trade-offs Explained
        </a>{" "}
        (which serving lever helps which metric). How this site was built, and
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
