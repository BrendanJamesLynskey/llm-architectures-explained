/**
 * /learn — index of chapters.
 *
 * Server Component, statically rendered. Same layout as LLM Inference
 * Explained's /learn (transformer-explainer's, minus the per-user progress
 * badges: this site has no accounts).
 */
import Link from "next/link";

import { SECTIONS } from "@/lib/mdx/sections";
import { DECODER_URL, INFERENCE_URL } from "@/lib/site";

export const metadata = {
  title: "Learn",
  description:
    "Nine chapters, one per way LLM architectures vary, each with a live interactive driven by the tested cost model.",
};

export default function LearnIndex(): JSX.Element {
  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        /learn
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        Architectures, axis by axis
      </h1>
      <p className="mt-4 text-neutral-600 dark:text-neutral-300">
        One chapter per way LLM architectures differ. Each has a live
        interactive driven by this site&apos;s cost model (checked against its
        Python reference) and links to the models in the data set that use the
        feature. Toggle layers (Concept / Maths / Code) inside any chapter to
        choose how deep to go. New to transformers? Start with the{" "}
        <a
          href={DECODER_URL}
          className="focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300"
        >
          Transformer Decoder Explainer
        </a>
        ; for what happens when a model is served, read{" "}
        <a
          href={INFERENCE_URL}
          className="focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300"
        >
          LLM Inference Explained
        </a>
        .
      </p>

      <ol className="mt-10 divide-y divide-neutral-200 dark:divide-neutral-800">
        {SECTIONS.map((s, i) => (
          <li key={s.slug} className="py-5">
            <div className="flex items-baseline gap-3">
              <span className="font-mono text-xs text-neutral-500 dark:text-neutral-400">
                {String(i + 1).padStart(2, "0")}
              </span>
              <Link
                href={`/learn/${s.slug}`}
                className="focus-ring rounded text-lg font-medium text-neutral-900 hover:text-accent dark:text-neutral-100"
              >
                {s.title}
              </Link>
            </div>
            <p className="mt-1 pl-9 text-sm text-neutral-600 dark:text-neutral-400">
              {s.summary}
            </p>
          </li>
        ))}
      </ol>
    </main>
  );
}
