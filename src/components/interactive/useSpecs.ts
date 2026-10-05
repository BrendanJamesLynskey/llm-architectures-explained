"use client";

/**
 * Fetch the compact model data (/data/specs.json) once per page and share it
 * between the chapter interactives that use real models.
 */
import { useEffect, useState } from "react";

import type { CompactModel } from "@/lib/arch/types";

let cache: Promise<CompactModel[]> | null = null;

function load(): Promise<CompactModel[]> {
  cache ??= fetch("/data/specs.json").then((r) =>
    r.ok
      ? (r.json() as Promise<CompactModel[]>)
      : Promise.reject(new Error(`HTTP ${r.status}`)),
  );
  return cache;
}

export function useSpecs(): {
  models: CompactModel[] | null;
  error: string | null;
} {
  const [models, setModels] = useState<CompactModel[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    load()
      .then(setModels)
      .catch((e: Error) => {
        cache = null;
        setError(e.message);
      });
  }, []);
  return { models, error };
}
