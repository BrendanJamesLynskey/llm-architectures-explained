/**
 * Site-wide constants: this site's URL, its two companion sites, and the
 * external pages the site links to.
 */

/** This site (production). */
export const SITE_URL = "https://llm-architectures-explained.vercel.app";

/** The companion sites. */
export const DECODER_URL = "https://transformer-decoder-explained.vercel.app";
export const INFERENCE_URL = "https://llm-inference-explained.vercel.app";

export const GITHUB_URL =
  "https://github.com/BrendanJamesLynskey/llm-architectures-explained";

/** Used as a checklist of model names only, and cited as related reading. */
export const GALLERY_URL =
  "https://sebastianraschka.com/llm-architecture-gallery/";

/** A file in this site's repository on GitHub. */
export function repoFile(path: string): string {
  return `${GITHUB_URL}/blob/main/${path}`;
}
