"use client";

/**
 * Code-split client widgets: each loads its own chunk after the page shell,
 * so pages stay light (the pattern of LLM Inference Explained's lazy.tsx).
 */
import dynamic from "next/dynamic";

function Placeholder({ what }: { what: string }): JSX.Element {
  return (
    <p
      data-pending-widget
      className="text-sm text-neutral-600 dark:text-neutral-400"
    >
      Loading the {what}…
    </p>
  );
}

const loading = (what: string) =>
  function Loading(): JSX.Element {
    return <Placeholder what={what} />;
  };

export const CompareTool = dynamic(() => import("./CompareTool"), {
  ssr: false,
  loading: loading("comparison"),
});

// The chapter interactives (/learn).
export const AttentionWidget = dynamic(() => import("./AttentionWidget"), {
  ssr: false,
  loading: loading("interactive"),
});
export const RopeWidget = dynamic(() => import("./RopeWidget"), {
  ssr: false,
  loading: loading("interactive"),
});
export const NormWidget = dynamic(() => import("./NormWidget"), {
  ssr: false,
  loading: loading("interactive"),
});
export const MoeWidget = dynamic(() => import("./MoeWidget"), {
  ssr: false,
  loading: loading("interactive"),
});
export const DepthWidthWidget = dynamic(() => import("./DepthWidthWidget"), {
  ssr: false,
  loading: loading("interactive"),
});
export const ContextWidget = dynamic(() => import("./ContextWidget"), {
  ssr: false,
  loading: loading("interactive"),
});
export const MtpWidget = dynamic(() => import("./MtpWidget"), {
  ssr: false,
  loading: loading("interactive"),
});
export const LoopWidget = dynamic(() => import("./LoopWidget"), {
  ssr: false,
  loading: loading("interactive"),
});
export const CedCostWidget = dynamic(() => import("./CedCostWidget"), {
  ssr: false,
  loading: loading("interactive"),
});
export const CedSimulatorWidget = dynamic(
  () => import("./CedSimulatorWidget"),
  { ssr: false, loading: loading("simulator") },
);
