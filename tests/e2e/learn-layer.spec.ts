/**
 * Brief 44, the learning layer: a self-check waits for a committed answer,
 * then reveals the model's value and says whether the guess was right; a
 * choice and a slider work from the keyboard; with no JavaScript the answer
 * is still one click away (a native <details>); axe finds nothing serious
 * with every answer of a chapter revealed, at 390 px in dark mode.
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

import { numericAnswer } from "@/lib/learn/answer";

async function check(page: Page, id: string) {
  const box = page.locator(`[data-self-check="${id}"]`);
  await expect(box).toBeVisible();
  return box;
}

test("ch 2: the reveal waits for an answer, then shows the model's", async ({
  page,
}) => {
  await page.goto("/learn/02-positional-encoding");
  await expect(page.locator("[data-check-yourself-link]")).toHaveAttribute(
    "href",
    "#check-yourself",
  );
  const box = await check(page, "h02-q1");
  const summary = box.locator("summary");
  const details = box.locator("details");
  await summary.click();
  await expect(details).not.toHaveAttribute("open", "");
  await expect(box).toContainText("Commit to an answer first");
  const want = numericAnswer(undefined, "ch02.mistral32k", "int");
  await box.getByRole("spinbutton").fill("32");
  await summary.click();
  await expect(details).toHaveAttribute("open", "");
  await expect(box.locator("[data-answer]")).toHaveText(want.display);
  await expect(box.locator("[data-verdict]")).toHaveAttribute(
    "data-verdict",
    "wrong",
  );
  await box.getByRole("spinbutton").fill(String(want.value));
  await expect(box.locator("[data-verdict]")).toHaveAttribute(
    "data-verdict",
    "right",
  );
  const q3 = await check(page, "h02-q3");
  await q3.getByRole("radio").nth(1).check();
  await q3.locator("summary").press("Enter");
  await expect(q3.locator("[data-verdict]")).toHaveAttribute(
    "data-verdict",
    "right",
  );
});

test("ch 2: the RoPE equation fits its box at 390 px", async ({ browser }) => {
  const ctx = await browser.newContext({
    viewport: { width: 390, height: 900 },
  });
  const page = await ctx.newPage();
  await page.goto("/learn/02-positional-encoding");
  const eq = page.locator(".eq-panel .katex-display").first();
  await expect(eq).toBeVisible();
  expect(await eq.evaluate((e) => e.scrollWidth <= e.clientWidth)).toBe(true);
  await ctx.close();
});

test("ch 5: the attention-share slider, by keyboard", async ({ page }) => {
  await page.goto("/learn/05-depth-and-width");
  const box = await check(page, "h05-q3");
  const want = numericAnswer(undefined, "ch05.deepAttn", "pct1");
  const slider = box.getByRole("slider");
  await slider.focus();
  await page.keyboard.press("Home");
  const steps = Math.round(want.value);
  for (let i = 0; i < steps; i++) await page.keyboard.press("ArrowRight");
  await box.locator("summary").press("Enter");
  await expect(box.locator("[data-answer]")).toHaveText(`${want.display}%`);
  await expect(box.locator("[data-verdict]")).toHaveAttribute(
    "data-verdict",
    "right",
  );
});

test("axe: nothing serious with every answer revealed (390 px, dark)", async ({
  browser,
}) => {
  const ctx = await browser.newContext({
    colorScheme: "dark",
    viewport: { width: 390, height: 900 },
  });
  const page = await ctx.newPage();
  await page.goto("/learn/09-encoder-decoder-and-ced");
  for (const id of ["h09-q1", "h09-q2", "h09-q3"]) {
    const box = await check(page, id);
    const slider = box.getByRole("slider");
    if (await slider.count()) {
      await slider.focus();
      await page.keyboard.press("ArrowRight");
    } else if (await box.getByRole("spinbutton").count()) {
      await box.getByRole("spinbutton").fill("4");
    } else {
      await box.getByRole("radio").first().check();
    }
    await box.locator("summary").click();
    await expect(box.locator("details")).toHaveAttribute("open", "");
  }
  const overflow = await page.evaluate(
    () =>
      document.documentElement.scrollWidth -
      document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
  const results = await new AxeBuilder({ page })
    .include("#check-yourself")
    .include("[data-remember]")
    .analyze();
  const serious = results.violations.filter(
    (v) => v.impact === "serious" || v.impact === "critical",
  );
  expect(serious, JSON.stringify(serious, null, 2)).toEqual([]);
  await ctx.close();
});

test("with no JavaScript the question and the answer are in the page", async ({
  browser,
}) => {
  const ctx = await browser.newContext({ javaScriptEnabled: false });
  const page = await ctx.newPage();
  await page.goto("/learn/01-attention");
  const box = page.locator('[data-self-check="h01-q3"]');
  await expect(box).toContainText("What does multi-head latent attention");
  await box.locator("summary").click();
  await expect(box.locator("[data-answer]")).toContainText(
    "Memory, not arithmetic",
  );
  await expect(page.locator("[data-remember]")).toContainText(
    "What to remember",
  );
  await ctx.close();
});

test("the landing page links the Inference track start page", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.locator("[data-track-start] a")).toHaveAttribute(
    "href",
    "https://llm-inference-explained.vercel.app/inference",
  );
});
