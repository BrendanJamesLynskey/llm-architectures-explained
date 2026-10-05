/**
 * /models: every model in the data set, filterable and sortable.
 * Server Component; the table itself is a client component.
 */
import { ModelTable } from "@/components/interactive/ModelTable";
import { toRow } from "@/lib/arch/rows";
import { MODELS } from "@/lib/data";

export const metadata = {
  title: "Models",
  description:
    "Every model in the data set: size, depth, width, attention, KV cache per token and context, with filters.",
};

export default function ModelsPage(): JSX.Element {
  const rows = MODELS.map(toRow);
  return (
    <main className="mx-auto max-w-6xl px-4 py-10 sm:px-6">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        /models
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">Models</h1>
      <p className="mt-3 max-w-3xl text-neutral-700 dark:text-neutral-300">
        {rows.length} models, from the original Transformer to this year&rsquo;s
        releases. Open a model to see every value with its source. Totals are
        the figures the labs state; an asterisk marks a count modelled from the
        configuration where no figure is stated.
      </p>
      <div className="mt-6">
        <ModelTable rows={rows} />
      </div>
    </main>
  );
}
