/**
 * The interactives work: table filters and sorting, the compare tool's
 * controls, and the estimates toggle, which must keep estimates out of the
 * defaults and label them whenever they are shown.
 */
import { expect, test } from "@playwright/test";

test("model table filters and sorts", async ({ page }) => {
  await page.goto("/models");
  await page.waitForLoadState("networkidle");
  const table = page.getByTestId("model-table");
  const rows = table.locator("tbody tr");
  const all = await rows.count();
  expect(all).toBeGreaterThan(150);
  await page.getByRole("button", { name: "MLA", exact: true }).click();
  const mla = await rows.count();
  expect(mla).toBeLessThan(all);
  await expect(table).toContainText("DeepSeek-V3");
  await expect(table).not.toContainText("Llama 3 8B");
  await page.getByRole("button", { name: "MLA", exact: true }).click();
  await page.getByRole("searchbox").fill("gemma");
  expect(await rows.count()).toBeLessThan(15);
  await page.getByRole("searchbox").fill("");
  await page.getByRole("button", { name: /^Total/ }).click();
  const first = await rows.first().locator("td").first().innerText();
  expect(first.length).toBeGreaterThan(0);
});

test("estimates stay hidden until switched on, then are labelled", async ({
  page,
}) => {
  await page.goto("/models");
  await page.waitForLoadState("networkidle");
  const table = page.getByTestId("model-table");
  expect(await table.locator("[data-estimate]").count()).toBe(0);
  await page.getByLabel("Show reported estimates for closed models").check();
  const est = table.locator("[data-estimate]");
  expect(await est.count()).toBeGreaterThan(3);
  for (const e of await est.all())
    await expect(e).toContainText("reported estimate");
});

test("compare tool computes and labels estimates", async ({ page }) => {
  await page.goto("/compare?m=llama-3-8b,deepseek-v3,gpt-4");
  await page.waitForLoadState("networkidle");
  const t = page.getByTestId("compare-table");
  await expect(t).toBeVisible();
  await expect(t).toContainText("Llama 3 8B");
  await expect(t).toContainText("DeepSeek-V3");
  expect(await t.locator("[data-estimate]").count()).toBe(0);
  const kvBefore = await t.locator("tr", { hasText: "KV cache" }).innerText();
  await page.getByRole("slider", { name: /Context length/ }).focus();
  await page.keyboard.press("End");
  await expect(t.locator("tr", { hasText: "KV cache" })).not.toHaveText(
    kvBefore,
  );
  await page.getByRole("radio", { name: "FP8" }).first().click();
  await page.getByLabel("Include reported estimates").check();
  const est = t.locator("[data-estimate]");
  expect(await est.count()).toBeGreaterThan(0);
  for (const e of await est.all())
    await expect(e).toContainText("reported estimate");
});
