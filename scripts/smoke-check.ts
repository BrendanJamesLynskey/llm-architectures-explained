/**
 * scripts/smoke-check.ts
 *
 * Post-deploy smoke check, adapted from LLM Inference Explained's. Fetches
 * every page and exits non-zero if any fails:
 *
 *     pnpm smoke https://llm-architectures-explained.vercel.app
 *
 * With no argument it checks http://localhost:3000. For a protected
 * preview deployment, pass the bypass token as VERCEL_BYPASS (sent as the
 * `x-vercel-protection-bypass` header; never printed).
 *
 * Fails when a page is not a 200 (redirects count as failures) or lacks
 * the content that proves it rendered real data: every model page must
 * show its name and provenance chips, and the compare tool's data file must
 * hold every model with specs the cost model can evaluate. Every chapter
 * must render its MDX (a layer and a pending interactive), and the CED
 * simulator's recorded results and workloads must be served at the vendored
 * engine's commit.
 */
import vendored from "@/lib/disagg/vendor/VENDORED.json";
import { params, type Spec } from "@/lib/arch/costModel";
import { MODELS } from "@/lib/data";
import { SECTIONS } from "@/lib/mdx/sections";

type Result = { path: string; ok: boolean; detail: string };

const headers: Record<string, string> = process.env.VERCEL_BYPASS
  ? { "x-vercel-protection-bypass": process.env.VERCEL_BYPASS }
  : {};

async function checkPage(
  base: string,
  path: string,
  mustContain: string[],
): Promise<Result> {
  try {
    const res = await fetch(base + path, { redirect: "manual", headers });
    if (res.status !== 200)
      return { path, ok: false, detail: String(res.status) };
    const html = await res.text();
    const missing = mustContain.filter((s) => !html.includes(s));
    if (missing.length)
      return {
        path,
        ok: false,
        detail: `200 but missing ${missing.join(", ")}`,
      };
    return { path, ok: true, detail: "200" };
  } catch (err) {
    return { path, ok: false, detail: (err as Error).message };
  }
}

async function checkSpecs(base: string): Promise<Result> {
  const path = "/data/specs.json";
  try {
    const res = await fetch(base + path, { redirect: "manual", headers });
    if (res.status !== 200)
      return { path, ok: false, detail: String(res.status) };
    const d = (await res.json()) as { id: string; spec: Spec | null }[];
    if (d.length !== MODELS.length)
      return { path, ok: false, detail: `200 but ${d.length} models` };
    const v3 = d.find((m) => m.id === "deepseek-v3");
    const total = v3?.spec ? params(v3.spec).total : 0;
    if (Math.abs(total / 671e9 - 1) > 0.01)
      return {
        path,
        ok: false,
        detail: `200 but DeepSeek-V3 models to ${total}`,
      };
    return {
      path,
      ok: true,
      detail: `200, ${d.length} models, DeepSeek-V3 = ${(total / 1e9).toFixed(1)}B`,
    };
  } catch (err) {
    return { path, ok: false, detail: (err as Error).message };
  }
}

async function checkJson(
  base: string,
  path: string,
  test: (d: Record<string, unknown>) => string | null,
): Promise<Result> {
  try {
    const res = await fetch(base + path, { redirect: "manual", headers });
    if (res.status !== 200)
      return { path, ok: false, detail: String(res.status) };
    const bad = test((await res.json()) as Record<string, unknown>);
    return bad
      ? { path, ok: false, detail: `200 but ${bad}` }
      : { path, ok: true, detail: "200" };
  } catch (err) {
    return { path, ok: false, detail: (err as Error).message };
  }
}

async function main(): Promise<void> {
  const base = (process.argv[2] ?? "http://localhost:3000").replace(/\/$/, "");
  const checks: Promise<Result>[] = [
    checkPage(base, "/", [
      "LLM Architectures Explained",
      'data-testid="stats"',
    ]),
    checkPage(base, "/models", ['data-testid="model-table"', "DeepSeek-V3"]),
    checkPage(base, "/compare", ["Compare and calculate"]),
    checkPage(base, "/timeline", ["How the variations spread"]),
    checkPage(base, "/about", ["Where every value comes from"]),
    checkSpecs(base),
    checkPage(
      base,
      "/learn",
      SECTIONS.map((x) => x.slug),
    ),
    ...SECTIONS.map((x) =>
      checkPage(base, `/learn/${x.slug}`, [
        'data-layer="concept"',
        "data-pending-widget",
      ]),
    ),
    checkJson(base, "/disagg/ced-results.json", (d) =>
      d.commit !== vendored.commit
        ? `commit ${String(d.commit)}`
        : (d.s17 as unknown[]).length !== 4
          ? "not four workloads"
          : null,
    ),
    ...[
      [2048, 512],
      [4096, 256],
      [8192, 128],
      [16384, 64],
    ].map(([p, o]) =>
      checkJson(base, `/disagg/workloads/ced-${p}-${o}.json`, (d) =>
        d.commit !== vendored.commit
          ? `commit ${String(d.commit)}`
          : (d.rows as unknown[]).length !== 1000
            ? "not 1,000 requests"
            : null,
      ),
    ),
  ];
  for (const m of MODELS) {
    const must = [m.name.replace(/&/g, "&amp;"), "data-status="];
    if (m.arch) must.push('data-testid="diagram"');
    if (m.facts.estimates?.length) must.push('data-testid="estimates"');
    checks.push(checkPage(base, `/models/${m.id}`, must));
  }
  const results = await Promise.all(checks);
  let failed = 0;
  for (const r of results) {
    if (!r.ok) failed++;
    if (!r.ok || results.length < 20)
      console.log(`${r.ok ? "ok  " : "FAIL"} ${r.path} ${r.detail}`);
  }
  console.log(
    `${results.length - failed}/${results.length} checks passed against ${base}`,
  );
  if (failed) process.exit(1);
}

void main();
