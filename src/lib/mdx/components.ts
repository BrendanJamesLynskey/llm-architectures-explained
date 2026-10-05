/**
 * MDX components map (as in LLM Inference Explained). The interactives come
 * through `lazy.tsx`, so each chapter loads only its own widgets' code;
 * `ModelsWith` lists the data set's models that use a feature.
 */
import type { MDXRemoteProps } from "next-mdx-remote/rsc";

import { Layer } from "@/components/interactive/Layer";
import {
  AttentionWidget,
  CedCostWidget,
  CedSimulatorWidget,
  ContextWidget,
  DepthWidthWidget,
  LoopWidget,
  MoeWidget,
  MtpWidget,
  NormWidget,
  RopeWidget,
} from "@/components/interactive/lazy";
import { ModelsWith } from "@/components/mdx/ModelsWith";
import { Callout } from "@/components/ui/Callout";
import { MdxTable } from "@/components/ui/MdxTable";

export const mdxComponents: NonNullable<MDXRemoteProps["components"]> = {
  table: MdxTable,
  Layer,
  Callout,
  ModelsWith,
  AttentionWidget,
  RopeWidget,
  NormWidget,
  MoeWidget,
  DepthWidthWidget,
  ContextWidget,
  MtpWidget,
  LoopWidget,
  CedCostWidget,
  CedSimulatorWidget,
};
