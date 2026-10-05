/**
 * `<ModelsWith feature="mla" />` in a chapter: the models in this site's
 * data set that use a feature, newest first, each linking to its page. The
 * features are derived from the sourced records (src/lib/arch/features.ts),
 * the same derivation the /models filters and the timeline use, so the list
 * always agrees with the data. Server Component.
 */
import Link from "next/link";

import { FEATURES, features, type FeatureId } from "@/lib/arch/features";
import { byRelease } from "@/lib/data";

export function modelsWith(
  feature: FeatureId,
): { id: string; name: string; released: string | null }[] {
  return byRelease()
    .filter((m) => features(m).has(feature))
    .reverse()
    .map((m) => ({ id: m.id, name: m.name, released: m.facts.released.v }));
}

export function ModelsWith({ feature }: { feature: FeatureId }): JSX.Element {
  const f = FEATURES.find((x) => x.id === feature)!;
  const list = modelsWith(feature);
  return (
    <details
      data-models-with={feature}
      className="my-4 rounded-lg border border-neutral-200 px-4 py-2 text-sm dark:border-neutral-800"
    >
      <summary className="focus-ring cursor-pointer rounded font-medium text-neutral-800 dark:text-neutral-200">
        {f.label}: {list.length} model{list.length === 1 ? "" : "s"} in the data
        set
      </summary>
      <p className="mt-2 text-xs text-neutral-600 dark:text-neutral-400">
        {f.description} Newest first; each links to its sourced page.
      </p>
      <ul className="mt-2 flex flex-wrap gap-1.5">
        {list.map((m) => (
          <li key={m.id}>
            <Link
              href={`/models/${m.id}`}
              className="focus-ring inline-block rounded border border-neutral-300 px-2 py-0.5 text-xs text-neutral-800 hover:border-accent hover:text-accent dark:border-neutral-700 dark:text-neutral-200 dark:hover:text-indigo-300"
            >
              {m.name}
            </Link>
          </li>
        ))}
      </ul>
    </details>
  );
}
