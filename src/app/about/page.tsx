/**
 * /about: what the site is, where every value comes from, how estimates are
 * handled, the cost model's conventions and how it is checked.
 * Server Component, static.
 */
import Link from "next/link";

import vendored from "@/lib/disagg/vendor/VENDORED.json";

import { StatusChip } from "@/components/ui/Provenance";
import {
  DECODER_URL,
  GALLERY_URL,
  GITHUB_URL,
  INFERENCE_URL,
  KERNELS_URL,
  NUMERICS_URL,
  SILICON_URL,
  TRADEOFFS_URL,
  repoFile,
} from "@/lib/site";
import { stats, weightChecks } from "@/lib/stats";

export const metadata = {
  title: "About",
  description:
    "Where every value on LLM Architectures Explained comes from, how estimates are labelled, and how the cost model is checked.",
};

const A =
  "focus-ring rounded text-accent underline underline-offset-2 dark:text-indigo-300";

export default function AboutPage(): JSX.Element {
  const s = stats();
  const w = weightChecks();
  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <p className="font-mono text-xs uppercase tracking-widest text-accent dark:text-indigo-300">
        /about
      </p>
      <h1 className="mt-2 text-3xl font-semibold tracking-tight">
        About this site
      </h1>
      <div className="mdx-content mt-6">
        <p>
          <strong>LLM Architectures Explained</strong> records how {s.models}{" "}
          language models are built and what that costs. It is the third of a
          family of companion sites: the{" "}
          <a className={A} href={DECODER_URL}>
            Transformer Decoder Explainer
          </a>{" "}
          shows one forward pass,{" "}
          <a className={A} href={INFERENCE_URL}>
            LLM Inference Explained
          </a>{" "}
          shows how a model is served,{" "}
          <a className={A} href={KERNELS_URL}>
            GPU Kernels Explained
          </a>{" "}
          shows how a GPU executes it,{" "}
          <a className={A} href={NUMERICS_URL}>
            Numerics Explained
          </a>{" "}
          shows the number formats it runs in,{" "}
          <a className={A} href={SILICON_URL}>
            Systolic Arrays Explained
          </a>{" "}
          shows the matrix hardware of TPUs, and{" "}
          <a className={A} href={TRADEOFFS_URL}>
            Inference Trade-offs Explained
          </a>{" "}
          measures which serving lever helps which metric. This one shows how
          the models themselves differ.
        </p>

        <h2>Where every value comes from</h2>
        <p>
          Every value carries a status and a link to its source, at the place it
          was read:
        </p>
        <ul>
          <li>
            <StatusChip st="config" /> read from the model&rsquo;s{" "}
            <code>config.json</code> at a pinned Hugging Face commit ({s.pinned}{" "}
            models). A test re-reads each such value from the stored snapshot.
          </li>
          <li>
            <StatusChip st="disclosed" /> stated by the lab: its model card at
            the same pinned commit, its GitHub, its documentation or its
            announcement.
          </li>
          <li>
            <StatusChip st="paper" /> from the model&rsquo;s paper, by section
            or table. Every arXiv identifier was checked against arXiv.
          </li>
          <li>
            <StatusChip st="code" /> a rule from the model&rsquo;s modelling
            code (a default the configuration leaves out, or where the norms
            sit), citing the file.
          </li>
          <li>
            <StatusChip st="not-disclosed" /> nothing credible is published.
          </li>
          <li>
            <StatusChip st="reported-estimate" /> a third-party estimate for a
            closed model; see below.
          </li>
        </ul>
        <p>
          Where a configuration is gated on Hugging Face, the values come
          instead from the lab&rsquo;s own GitHub (Meta&rsquo;s
          <code>llama-models</code>, Google DeepMind&rsquo;s <code>gemma</code>,
          xAI&rsquo;s <code>grok-1</code>) or from the paper, transcribed into a
          file that names the line or table for every value. The data, the
          snapshots and the transcriptions are all in the{" "}
          <a className={A} href={GITHUB_URL}>
            repository
          </a>
          , with a{" "}
          <a className={A} href={repoFile("data/schema/model.schema.json")}>
            JSON Schema
          </a>{" "}
          that CI validates.
        </p>

        <h2 id="gallery">The gallery checklist</h2>
        <p>
          Sebastian Raschka&rsquo;s{" "}
          <a className={A} href={GALLERY_URL}>
            LLM Architecture Gallery
          </a>{" "}
          is good related reading. This site uses it only as a checklist: its{" "}
          {s.gallery} model names are stored, and a test checks that each one
          maps to a model here. No figure, diagram or text is taken from it;
          every diagram here is generated from the data, and every fact comes
          from the model&rsquo;s own sources. The other {s.extras} models add
          the history the gallery skips (the 2017 Transformer, BERT, T5, GPT-3,
          Switch, GLaM, PaLM, Chinchilla, BLOOM, Llama, Mistral and Mixtral,
          Mamba, RWKV, Jamba and more) and closed frontier models.
        </p>

        <h2 id="estimates">Closed models and reported estimates</h2>
        <p>
          Closed models are listed with what their labs disclose: usually the
          context window and the release date, and sometimes that the model is a
          mixture of experts. Where a third party has published a size estimate,
          it is shown as a <em>reported estimate</em>: never as a fact, always
          with its source, its date and the confidence its source gives. There
          are {s.estimates} such values, all from one source whose authors say
          they cannot vouch for them. The calculator ignores them unless you
          switch them on. Where nothing credible exists, the site says
          &ldquo;not disclosed&rdquo;.
        </p>

        <h2 id="cost-model">The cost model</h2>
        <p>
          For each model with published dimensions ({s.withArch} of {s.models}),
          the cost model computes parameter counts, KV-cache and recurrent-state
          bytes, prefill and decode FLOPs, and the bytes a decode step reads. It
          is a Python reference (
          <a className={A} href={repoFile("reference/arch_model.py")}>
            reference/arch_model.py
          </a>
          ) with unit tests for every closed form, and a TypeScript port that
          this site runs; CI checks that the port reproduces every number in the
          reference&rsquo;s fixtures exactly, for every model. Conventions:
        </p>
        <ul>
          <li>
            a matrix-vector product of an m-vector with an m × n matrix costs
            2mn FLOPs; norms, activations, softmax and routing are not counted;
          </li>
          <li>
            attention is counted in its expanded form for every type
            (MLA&rsquo;s absorbed decode trades these FLOPs for latent-space
            ones);
          </li>
          <li>
            sliding-window layers keep only their window; chunked layers (Llama
            4) keep one chunk, and each query reads only its own chunk so far;
            sparse attention reads only the selected entries, plus its indexer
            keys in full;
          </li>
          <li>
            linear attention, Mamba and short convolutions keep a fixed-size
            state, not a cache;
          </li>
          <li>
            a causal encoder-decoder (DeepSeek-V4.1-Flash) runs only its encoder
            half over the prompt, projects the decoder&rsquo;s keys and values
            from the encoder output, and replays the last tokens through the
            decoder half;
          </li>
          <li>
            batch size 1; serving at scale changes the balance between weights
            and cache (see LLM Inference Explained).
          </li>
        </ul>

        <h2 id="checks">How the counts are checked</h2>
        <p>
          The modelled parameter count of {w.within} of the {w.compared} open
          text models whose published weights can be counted (unpacked, no
          vision tower) is within 0.5% of the number of parameters in those
          weights, at the pinned commit. Every stated total is also checked
          against the model (within 5%, or between the counts with and without
          embeddings, since labs differ on that), with each exception named in
          the tests. Some parts are approximated and say so on their model
          pages: compressed convolutional attention (ZAYA1), Kimi Delta
          Attention&rsquo;s gate projections, n-gram embedding tables, GDLA
          (Motif) and the weights of multi-token-prediction layers.
        </p>

        <h2 id="chapters">The chapters and the simulator</h2>
        <p>
          The nine{" "}
          <Link className={A} href="/learn">
            chapters
          </Link>{" "}
          each drive an interactive with this cost model, or with a closed form
          from the same Python reference (
          <code>reference/chapter_model.py</code>, ported line by line and
          checked against its fixtures). The encoder-decoder chapter also runs{" "}
          <a
            className={A}
            href="https://github.com/BrendanJamesLynskey/Disaggregated_Inference_Sim"
          >
            Disaggregated_Inference_Sim
          </a>
          &rsquo;s own JavaScript engine, copied byte for byte from commit{" "}
          <code>{vendored.commit.slice(0, 7)}</code> (the one that added the
          causal encoder-decoder option), with parity tests against the Python
          package at that commit, and reruns its capacity search in the browser,
          rate for rate. Its numbers are for an illustrative dense 70B-shaped
          proxy, not for any lab&rsquo;s model.
        </p>

        <h2>Freshness</h2>
        <p>
          A weekly workflow re-fetches every pinned configuration and reports
          any model whose repository has moved on, so the data can be re-pinned
          deliberately rather than drift.
        </p>

        <h2>How it is built</h2>
        <p>
          Next.js 14 with strict TypeScript and Tailwind, the design system of
          the companion sites, and diagrams and charts in plain SVG generated
          from the data. There is no database and no sign-in. Vitest checks the
          cost model and the data helpers, pytest the reference and the data,
          and Playwright every page at desktop and phone widths in light and
          dark mode. Start at{" "}
          <Link className={A} href="/models">
            the models
          </Link>
          .
        </p>
      </div>
    </main>
  );
}
