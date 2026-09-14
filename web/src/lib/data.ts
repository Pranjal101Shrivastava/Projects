/**
 * Artifact loading.
 *
 * Pipeline artifacts are copied into `public/data/<project>/` by `tools/sync_artifacts.py`
 * and fetched at runtime rather than imported into the bundle. That keeps the initial page
 * weight small: a visitor who opens the fraud project never downloads the 470 KB of NYC
 * exploratory data.
 *
 * Everything rendered anywhere in this application comes through here. No component
 * hard-codes a metric, so a figure on screen and a figure in the committed JSON cannot
 * drift apart.
 */

import { useEffect, useState } from "react";

const BASE = import.meta.env.BASE_URL;

export type LoadState<T> =
  | { status: "loading" }
  | { status: "error"; error: string }
  | { status: "ready"; data: T };

const cache = new Map<string, unknown>();

export async function loadArtifact<T = unknown>(project: string, name: string): Promise<T> {
  const key = `${project}/${name}`;
  if (cache.has(key)) return cache.get(key) as T;

  const response = await fetch(`${BASE}data/${key}.json`);
  if (!response.ok) {
    throw new Error(`Could not load ${key}.json (HTTP ${response.status})`);
  }
  const payload = (await response.json()) as T;
  cache.set(key, payload);
  return payload;
}

/** Load several artifacts for one project, keyed by name. */
export function useArtifacts<T extends Record<string, unknown>>(
  project: string,
  names: string[]
): LoadState<T> {
  const [state, setState] = useState<LoadState<T>>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading" });

    Promise.all(names.map((n) => loadArtifact(project, n)))
      .then((results) => {
        if (cancelled) return;
        const merged = Object.fromEntries(
          names.map((n, i) => [n, results[i]])
        ) as T;
        setState({ status: "ready", data: merged });
      })
      .catch((error: Error) => {
        if (!cancelled) setState({ status: "error", error: error.message });
      });

    return () => {
      cancelled = true;
    };
    // names is a literal array at every call site, so joining is a stable key.
  }, [project, names.join(",")]);

  return state;
}

/* ------------------------------------------------------------------ */
/* Shared artifact shapes                                              */
/* ------------------------------------------------------------------ */

export type RunBlock = {
  project: string;
  seed: number;
  git_commit: string;
  libraries: Record<string, string>;
  duration_seconds: number;
};

export type Decision = {
  question: string;
  choice: string;
  rationale: string;
  alternatives_rejected: string[];
};

export type Phase = {
  name: string;
  title: string;
  summary: string;
  decisions: Decision[];
  evidence: Record<string, unknown>;
  risks: string[];
};

export type CrispDmDoc = {
  project: string;
  business_question: string;
  phases: Phase[];
  phases_recorded: number;
  complete: boolean;
  _run?: RunBlock;
};

export type ProvenanceRecord = {
  id: string;
  title: string;
  kind: "REAL" | "SIMULATED";
  origin: string;
  license: string;
  url: string | null;
  mirror_note: string;
  rows: string;
  notes: string;
  cached_files: { file: string; sha256: string }[];
};

export type ProvenanceDoc = { datasets: ProvenanceRecord[]; _run?: RunBlock };
