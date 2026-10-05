/**
 * /compare: two to four models side by side, with a calculator.
 * Server Component shell; the tool is a code-split client component.
 */
import Link from "next/link";

import { CompareTool } from "@/components/interactive/lazy";

export const metadata = {
  title: "Compare",
  description:
    "Compare up to four LLM architectures: diagrams, KV cache against context, and prefill and decode cost.",
};

const A =
  "focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300";

export default function ComparePage(): JSX.Element {
  return (
    <main className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        /compare
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        Compare and calculate
      </h1>
      <p className="mt-3 max-w-3xl text-neutral-700 dark:text-neutral-300">
        Pick up to four models. The numbers come from the{" "}
        <Link className={A} href="/about#cost-model">
          cost model
        </Link>
        , run in your browser on each model&rsquo;s sourced configuration:
        memory for the weights and the KV cache, and the arithmetic and memory
        traffic of prefill and of each decode step. Closed models whose sizes
        are not disclosed stay blank unless you include reported estimates,
        which are labelled wherever they appear.
      </p>
      <div className="mt-6">
        <CompareTool />
      </div>
    </main>
  );
}
