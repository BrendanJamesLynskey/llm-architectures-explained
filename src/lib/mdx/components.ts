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
import {
  HeadSharing,
  MoeRouting,
  RopePhasors,
} from "@/components/interactive/Visuals";
import { CheckYourself } from "@/components/learn/CheckYourself";
import { Keep, Remember } from "@/components/learn/Remember";
import { Check } from "@/components/mdx/Check";
import { V } from "@/components/mdx/V";
import { ModelsWith } from "@/components/mdx/ModelsWith";
import { Callout } from "@/components/ui/Callout";
import { MdxTable } from "@/components/ui/MdxTable";

export const mdxComponents: NonNullable<MDXRemoteProps["components"]> = {
  table: MdxTable,
  Layer,
  Callout,
  CheckYourself,
  Check,
  Remember,
  Keep,
  V,
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
  // brief 41: the chapter heroes
  HeadSharing,
  RopePhasors,
  MoeRouting,
};
