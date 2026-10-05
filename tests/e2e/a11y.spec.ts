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
];

for (const scheme of ["light", "dark"] as const) {
  test.describe(`axe, ${scheme}`, () => {
    test.use({ colorScheme: scheme });
    for (const path of PAGES) {
      test(`${path} has no serious or critical violations`, async ({
        page,
      }) => {
        await page.goto(path);
        await page.waitForLoadState("networkidle");
        if (path === "/compare")
          await expect(page.getByTestId("compare-table")).toBeVisible();
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
