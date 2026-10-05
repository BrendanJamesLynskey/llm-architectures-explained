/**
 * Captures the README screenshots. Manual run, output committed.
 *
 *   pnpm build && pnpm start   # in another shell
 *   pnpm screenshots           # headless Chromium writes docs/screenshots/*.png
 *
 * Light theme, fixed viewport, and the interactives' default (deterministic)
 * settings, as transformer-explainer's script does.
 */
import { mkdirSync } from "node:fs";
import path from "node:path";

import { chromium } from "@playwright/test";

const OUT = path.join(process.cwd(), "docs", "screenshots");
const BASE = process.env.SCREENSHOT_BASE_URL ?? "http://localhost:3000";

type Shot = {
  name: string;
  path: string;
  widget?: string;
  click?: RegExp[];
  /** Wait until the widget's text contains this (e.g. a finished simulation). */
  waitText?: string;
};

const SHOTS: Shot[] = [
  { name: "01-landing", path: "/" },
  { name: "02-models", path: "/models" },
  {
    name: "03-model-page",
    path: "/models/deepseek-v4.1-flash",
    widget: "diagram",
  },
  {
    name: "04-compare",
    path: "/compare?m=llama-3-8b,qwen3-next-80b-a3b,deepseek-v3",
    widget: "compare-tool",
  },
  { name: "05-timeline", path: "/timeline" },
  {
    name: "06-attention-chapter",
    path: "/learn/01-attention",
    widget: "attention-widget",
  },
  {
    name: "07-ced-simulator",
    path: "/learn/09-encoder-decoder-and-ced",
    widget: "ced-simulator",
    click: [/Re-measure live/],
    waitText: "Live: 15 of 15 cells",
  },
];

async function main(): Promise<void> {
  mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    colorScheme: "light",
  });
  const page = await context.newPage();
  for (const s of SHOTS) {
    await page.goto(BASE + s.path);
    await page.waitForLoadState("networkidle");
    const target = s.widget ? page.getByTestId(s.widget).first() : null;
    for (const name of s.click ?? []) {
      await target!.getByRole("button", { name }).first().click();
    }
    if (s.waitText) {
      const want = s.waitText;
      await page.waitForFunction(
        ([id, t]) =>
          document
            .querySelector(`[data-testid="${id}"]`)
            ?.textContent?.includes(t) ?? false,
        [s.widget!, want] as const,
        { timeout: 180_000 },
      );
    }
    const file = path.join(OUT, `${s.name}.png`);
    if (target) await target.screenshot({ path: file });
    else await page.screenshot({ path: file });
    console.log("wrote", path.relative(process.cwd(), file));
  }
  await browser.close();
}

void main();
