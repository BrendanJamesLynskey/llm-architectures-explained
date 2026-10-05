"use client";

/**
 * Code-split client widgets: each loads its own chunk after the page shell,
 * so pages stay light (the pattern of LLM Inference Explained's lazy.tsx).
 */
import dynamic from "next/dynamic";

function Placeholder(): JSX.Element {
  return (
    <p
      data-pending-widget
      className="text-sm text-neutral-600 dark:text-neutral-400"
    >
      Loading the comparison…
    </p>
  );
}

export const CompareTool = dynamic(() => import("./CompareTool"), {
  ssr: false,
  loading: Placeholder,
});
