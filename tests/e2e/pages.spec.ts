/**
 * Every page renders at desktop and phone widths, in light and dark mode,
 * with no page errors, no console errors and no horizontal overflow.
 */
import { expect, test, type Page } from "@playwright/test";

const PAGES = [
  "/",
  "/models",
  "/compare",
  "/timeline",
  "/about",
  "/models/deepseek-v3",
  "/models/deepseek-v4.1-flash",
  "/models/gemma-3-27b",
  "/models/qwen3-next-80b-a3b",
  "/models/nemotron-3-super",
  "/models/t5-11b",
  "/models/olmo-2-7b",
  "/models/gpt-4",
  "/models/claude-opus-5.5",
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

async function collectErrors(page: Page): Promise<string[]> {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(`console: ${m.text()}`);
  });
  return errors;
}

for (const scheme of ["light", "dark"] as const) {
  for (const width of [1280, 390]) {
    test.describe(`${scheme} @ ${width}px`, () => {
      test.use({ colorScheme: scheme, viewport: { width, height: 900 } });
      for (const path of PAGES) {
        test(`${path} renders cleanly`, async ({ page }) => {
          const errors = await collectErrors(page);
          const res = await page.goto(path);
          expect(res?.status()).toBe(200);
          await expect(page.locator("h1").first()).toBeVisible();
          await page.waitForLoadState("networkidle");
          if (path === "/compare") {
            await expect(page.getByTestId("compare-table")).toBeVisible();
          }
          // chapter interactives load after the page: wait for every one
          await expect(page.locator("[data-pending-widget]")).toHaveCount(0);
          const overflow = await page.evaluate(() => {
            const el = document.scrollingElement!;
            return el.scrollWidth - el.clientWidth;
          });
          expect(overflow, "horizontal overflow (px)").toBeLessThanOrEqual(0);
          const bg = await page.evaluate(
            () => getComputedStyle(document.body).backgroundColor,
          );
          expect(bg).toBe(
            scheme === "dark" ? "rgb(10, 10, 10)" : "rgb(255, 255, 255)",
          );
          expect(errors).toEqual([]);
        });
      }
    });
  }
}

test("every model page renders with provenance", async ({ page }) => {
  test.setTimeout(240_000);
  const res = await page.request.get("/data/specs.json");
  const models = (await res.json()) as { id: string; spec: unknown }[];
  expect(models.length).toBeGreaterThan(150);
  for (const m of models) {
    const r = await page.request.get(`/models/${m.id}`);
    expect(r.status(), m.id).toBe(200);
    const html = await r.text();
    expect(html, m.id).toContain("data-status=");
    if (m.spec) expect(html, m.id).toContain('data-testid="diagram"');
    else expect(html, m.id).toContain('data-testid="no-arch"');
  }
});

test("the six-way site switch: a row on desktop, a dropdown on phones", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto("/");
  const nav = page.getByRole("navigation", { name: "Companion sites" });
  await expect(
    nav.getByRole("link", { name: "Architectures" }),
  ).toHaveAttribute("aria-current", "true");
  await expect(nav.getByRole("link", { name: "Kernels" })).toHaveAttribute(
    "href",
    "https://gpu-kernels-explained.vercel.app",
  );
  await expect(nav.getByRole("link", { name: "Numerics" })).toHaveAttribute(
    "href",
    "https://numerics-explained.vercel.app",
  );
  await expect(nav.getByRole("link", { name: "Silicon" })).toHaveAttribute(
    "href",
    "https://systolic-arrays-explained.vercel.app",
  );
  await page.setViewportSize({ width: 390, height: 800 });
  const compact = page.locator("[data-site-switch='compact']");
  await expect(compact).toBeVisible();
  await compact.locator("summary").click();
  const box = await compact
    .getByRole("link", { name: "Decoder" })
    .boundingBox();
  expect(box!.height).toBeGreaterThanOrEqual(44);
  const overflow = await page.evaluate(
    () =>
      document.scrollingElement!.scrollWidth -
      document.scrollingElement!.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
