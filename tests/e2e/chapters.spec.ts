/**
 * The chapter interactives respond to their controls with the cost model's
 * numbers, and the encoder-decoder chapter's simulator re-measures results.md
 * in the browser, rate for rate.
 */
import { expect, test } from "@playwright/test";

test("attention widget: GQA's cache at 128K is 16 GiB, MHA's 4×", async ({
  page,
}) => {
  await page.goto("/learn/01-attention");
  await page.waitForLoadState("networkidle");
  const w = page.getByTestId("attention-widget");
  const slider = w.getByRole("slider", { name: /Context length/ });
  await slider.fill("17");
  await expect(w.locator('[data-variant="gqa"] [data-cell="kv"]')).toHaveText(
    "16 GiB",
  );
  await expect(w.locator('[data-variant="mha"] [data-cell="kv"]')).toHaveText(
    "64 GiB",
  );
});

test("RoPE widget: Qwen3 8B leaves 24 pairs slower than 32K", async ({
  page,
}) => {
  await page.goto("/learn/02-positional-encoding");
  await page.waitForLoadState("networkidle");
  const w = page.getByTestId("rope-widget");
  await w.getByRole("combobox").selectOption("qwen3-8b");
  await expect(w.locator("svg[data-long-pairs]")).toHaveAttribute(
    "data-long-pairs",
    "24",
  );
});

test("MoE widget: presets rebuild the model", async ({ page }) => {
  await page.goto("/learn/04-dense-and-moe");
  await page.waitForLoadState("networkidle");
  const w = page.getByTestId("moe-widget");
  await w.getByRole("button", { name: /^Dense/ }).click();
  await expect(w.getByTestId("moe-stats")).toContainText("8.03B");
  await w.getByRole("button", { name: /DeepSeek-V3's counts/ }).click();
  await expect(w.getByTestId("moe-stats")).toContainText("85B");
  await expect(w.getByTestId("moe-stats")).toContainText("5.83B");
});

test("CED simulator: section 16 matches, and section 17 re-measures exactly", async ({
  page,
}) => {
  test.setTimeout(180_000);
  await page.goto("/learn/09-encoder-decoder-and-ced");
  await page.waitForLoadState("networkidle");
  const w = page.getByTestId("ced-simulator");
  await expect(w.locator('[data-step-check="ok"]')).toHaveCount(3);
  await w.getByRole("radio", { name: "2048 : 512" }).click();
  await w.getByRole("button", { name: "Re-measure live" }).click();
  await expect(w.getByTestId("ced-progress")).toContainText(
    "Live: 15 of 15 cells",
    { timeout: 150_000 },
  );
  await expect(w.getByTestId("ced-progress")).toContainText(
    "15 identical to the recorded rate",
  );
  await expect(w.locator('[data-match="true"]')).toHaveCount(15);
  await expect(
    w.locator('[data-cell="CED, replay on decode|3"]'),
  ).toHaveAttribute("data-live", "41.07");
});
