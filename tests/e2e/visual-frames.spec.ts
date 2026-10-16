/**
 * Frame tests on the page (brief 41): set key frames with the scrub bar and
 * require the caption on screen to be the caption built from
 * reference/visuals.py's state (tests/fixtures/visual_fixtures.json); and
 * the landing page's hero and CED result.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { expect, test, type Locator } from "@playwright/test";

import {
  MOE,
  moeCaption,
  ropeCaption,
  sharingCaption,
  type MoeState,
  type RopeState,
  type Sharing,
} from "@/lib/chapters/visuals";

const fx = JSON.parse(
  readFileSync(
    join(process.cwd(), "tests/fixtures/visual_fixtures.json"),
    "utf-8",
  ),
) as {
  rope: { banks: Record<string, { states: RopeState[] }> };
  sharing: Sharing[];
  moe: { on: MoeState[]; off: MoeState[] };
};

test.use({ contextOptions: { reducedMotion: "reduce" } });

async function open(
  page: import("@playwright/test").Page,
  path: string,
  id: string,
) {
  await page.goto(path);
  await expect(page.locator("[data-pending-widget]")).toHaveCount(0, {
    timeout: 30_000,
  });
  const fig = page.getByTestId(id);
  await expect(fig).toBeVisible({ timeout: 30_000 });
  return fig;
}

async function show(fig: Locator, s: number): Promise<void> {
  await fig.getByTestId("scrub").fill(String(s));
  await expect(fig).toHaveAttribute("data-step", String(s));
}

test("chapter 1: head sharing's frames are the reference's", async ({
  page,
}) => {
  const fig = await open(page, "/learn/01-attention", "head-sharing");
  for (const [k, v] of fx.sharing.entries()) {
    await show(fig, k);
    await expect(fig.getByTestId("caption")).toHaveText(
      sharingCaption(v, fx.sharing[0]!),
    );
    await expect(fig.getByTestId("kv-unit")).toHaveCount(v.units);
  }
});

test("chapter 2: the RoPE bank's frames are the reference's", async ({
  page,
}) => {
  const fig = await open(page, "/learn/02-positional-encoding", "rope-phasors");
  const st = fx.rope.banks["mistral-7b|query"]!.states;
  for (const k of [0, 1, 40, st.length - 1]) {
    await show(fig, k);
    await expect(fig.getByTestId("caption")).toHaveText(
      ropeCaption(st[k]!, "query"),
    );
  }
});

test("chapter 4: MoE routing's frames are the reference's", async ({
  page,
}) => {
  const fig = await open(page, "/learn/04-dense-and-moe", "moe-routing");
  const st = fx.moe.on;
  for (const k of [0, 1, 16, st.length - 1]) {
    await show(fig, k);
    await expect(fig.getByTestId("caption")).toHaveText(
      moeCaption(st[k]!, MOE, true),
    );
    for (const e of [0, 3, 7])
      await expect(fig.getByTestId(`expert-${e}`)).toHaveAttribute(
        "data-load",
        String(st[k]!.loads[e]),
      );
  }
});

test("landing: the KV landscape and the CED result", async ({ page }) => {
  await page.goto("/");
  const fig = page.locator("[data-hero-figure] [data-kv-landscape=wide]");
  await expect(fig).toBeVisible();
  expect(await fig.locator("circle[data-family]").count()).toBeGreaterThan(130);
  await expect(page.locator("[data-ced-result]")).toContainText(
    "1.52× to 2.14×",
  );
});
