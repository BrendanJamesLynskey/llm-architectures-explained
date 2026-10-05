/**
 * axe (serious and critical violations) on the main pages, in light and
 * dark mode.
 */
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

const PAGES = [
  "/",
  "/models",
  "/compare",
  "/timeline",
  "/about",
  "/models/deepseek-v3",
  "/models/gpt-4",
  "/learn",
  "/learn/01-attention",
  "/learn/02-positional-encoding",
  "/learn/03-normalisation",
  "/learn/04-dense-and-moe",
  "/learn/05-depth-and-width",
  "/learn/06-long-context",
  "/learn/07-multi-token-prediction",
  "/learn/08-looped-and-parallel-blocks",
  "/learn/09-encoder-decoder-and-ced",
];

for (const scheme of ["light", "dark"] as const) {
  test.describe(`axe, ${scheme}`, () => {
    test.use({ colorScheme: scheme });
    // scan the chapters with every layer (Concept, Maths, Code) shown
    test.beforeEach(async ({ page }) => {
      await page.addInitScript(() =>
        window.localStorage.setItem(
          "te:layers",
          JSON.stringify({ concept: true, maths: true, code: true }),
        ),
      );
    });
    for (const path of PAGES) {
      test(`${path} has no serious or critical violations`, async ({
        page,
      }) => {
        await page.goto(path);
        await page.waitForLoadState("networkidle");
        if (path === "/compare")
          await expect(page.getByTestId("compare-table")).toBeVisible();
        await expect(page.locator("[data-pending-widget]")).toHaveCount(0);
        // open the model lists so their links are scanned too
        for (const d of await page.locator("details[data-models-with]").all())
          await d.evaluate((el) => ((el as HTMLDetailsElement).open = true));
        const r = await new AxeBuilder({ page }).analyze();
        const bad = r.violations.filter(
          (v) => v.impact === "serious" || v.impact === "critical",
        );
        expect(
          bad.map(
            (v) =>
              `${v.id}: ${v.nodes
                .map((n) => n.target.join(" "))
                .slice(0, 3)
                .join(", ")}`,
          ),
        ).toEqual([]);
      });
    }
  });
}
