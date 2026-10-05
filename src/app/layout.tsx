/**
 * Root layout for the App Router.
 *
 * Server Component, copied from transformer-explainer's layout (via LLM Inference Explained): HTML
 * scaffold, the global stylesheet and the site header. Dark mode follows
 * the system setting (`darkMode: "media"` in tailwind.config.ts).
 */
import type { Metadata } from "next";
import type { ReactNode } from "react";

import { SiteHeader } from "@/components/ui/SiteHeader";
import { SITE_URL } from "@/lib/site";

import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "LLM Architectures Explained",
    template: "%s · LLM Architectures Explained",
  },
  description:
    "How LLM architectures differ: attention, positions, normalisation, mixture-of-experts, depth and width, and more, across 160 models with every fact sourced and a tested cost model.",
};

export default function RootLayout({
  children,
}: {
  children: ReactNode;
}): JSX.Element {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen font-sans antialiased">
        <SiteHeader />
        {children}
      </body>
    </html>
  );
}
