/**
 * /sitemap.xml: every page, every chapter and every model page, at this site's production
 * address.
 */
import type { MetadataRoute } from "next";

import { MODELS } from "@/lib/data";
import { SECTIONS } from "@/lib/mdx/sections";
import { SITE_URL } from "@/lib/site";

/** The site's other pages (the chapters are added from the catalogue). */
const PAGES = ["/", "/learn", "/models", "/compare", "/timeline", "/about"];

export default function sitemap(): MetadataRoute.Sitemap {
  return [
    ...PAGES,
    ...SECTIONS.map((s) => `/learn/${s.slug}`),
    ...MODELS.map((m) => `/models/${m.id}`),
  ].map((p) => ({ url: p === "/" ? SITE_URL : `${SITE_URL}${p}` }));
}
