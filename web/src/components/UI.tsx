/**
 * Shared presentational components.
 *
 * These exist so that the eight project views present the same kinds of information the
 * same way. The most load-bearing of them are `Caveat` and `PhaseTimeline`: this portfolio
 * argues that stating limitations is part of the work, so limitations get a dedicated,
 * visually prominent component rather than a paragraph at the bottom of a page.
 */

import { useState, type ReactNode } from "react";
import type { CrispDmDoc, Phase, ProvenanceRecord, LoadState } from "../lib/data";
import { fmt } from "./Charts";

/* ------------------------------------------------------------------ */

export function Stat({
  label, value, sub, tone,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  tone?: "good" | "warn" | "bad" | "accent";
}) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className={`stat-value${tone ? ` ${tone}` : ""}`}>{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

export function Callout({
  kind = "info", title, children,
}: {
  kind?: "info" | "warn" | "bad" | "good";
  title: string;
  children: ReactNode;
}) {
  const icon = { info: "ℹ", warn: "▲", bad: "✕", good: "✓" }[kind];
  return (
    <div className={`callout ${kind}`}>
      <div className="callout-title">
        <span aria-hidden="true">{icon}</span>
        {title}
      </div>
      <div>{children}</div>
    </div>
  );
}

/**
 * A limitation of the work, rendered prominently.
 *
 * Deliberately styled to draw the eye rather than to be skimmed past. A portfolio that
 * claims methodological care has to make its caveats as visible as its results.
 */
export function Caveat({ children }: { children: ReactNode }) {
  return (
    <div className="callout warn">
      <div className="callout-title">
        <span aria-hidden="true">▲</span> Limitation
      </div>
      <div>{children}</div>
    </div>
  );
}

export function Section({
  title, note, children, id,
}: {
  title: string;
  note?: ReactNode;
  children: ReactNode;
  id?: string;
}) {
  return (
    <section className="section" id={id}>
      <div className="section-head">
        <h2>{title}</h2>
        {note && <span className="section-note">{note}</span>}
      </div>
      {children}
    </section>
  );
}

export function Tabs({
  tabs, active, onChange,
}: {
  tabs: { key: string; label: string }[];
  active: string;
  onChange: (key: string) => void;
}) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map((t) => (
        <button key={t.key} role="tab" aria-selected={active === t.key}
                className={`tab${active === t.key ? " active" : ""}`}
                onClick={() => onChange(t.key)}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

export function Loading({ what }: { what: string }) {
  return <div className="card faint">Loading {what}…</div>;
}

export function LoadError({ error }: { error: string }) {
  return (
    <Callout kind="bad" title="Could not load artifacts">
      <p className="small" style={{ margin: 0 }}>{error}</p>
      <p className="small dim" style={{ marginTop: 8, marginBottom: 0 }}>
        Artifacts are produced by the project pipeline and synced into the site by{" "}
        <code>tools/sync_artifacts.py</code>. Run the pipeline, then the sync script, then
        rebuild.
      </p>
    </Callout>
  );
}

/** Render the three load states without every page repeating the branch. */
export function Resolved<T>({
  state, what, children,
}: {
  state: LoadState<T>;
  what: string;
  children: (data: T) => ReactNode;
}) {
  if (state.status === "loading") return <Loading what={what} />;
  if (state.status === "error") return <LoadError error={state.error} />;
  return <>{children(state.data)}</>;
}

/* ------------------------------------------------------------------ */
/* Provenance                                                          */
/* ------------------------------------------------------------------ */

export function ProvenanceTable({ datasets }: { datasets: ProvenanceRecord[] }) {
  const [open, setOpen] = useState<string | null>(null);
  return (
    <div className="stack">
      {datasets.map((d) => (
        <div className="card" key={d.id}>
          <div className="row">
            <span className={`badge ${d.kind === "REAL" ? "badge-real" : "badge-sim"}`}>
              {d.kind === "REAL" ? "● REAL DATA" : "● SIMULATED"}
            </span>
            <strong>{d.title}</strong>
            <span className="spacer" />
            <span className="tiny faint mono">{d.rows}</span>
          </div>
          <p className="small dim" style={{ margin: "10px 0 0" }}>{d.origin}</p>
          <div className="row tiny faint" style={{ marginTop: 10 }}>
            <span>Licence: {d.license}</span>
            <span className="spacer" />
            <button className="chip" onClick={() => setOpen(open === d.id ? null : d.id)}>
              {open === d.id ? "Hide" : "Show"} source & integrity
            </button>
          </div>
          {open === d.id && (
            <div style={{ marginTop: 12, borderTop: "1px solid var(--border)", paddingTop: 12 }}>
              {d.url && (
                <p className="tiny mono" style={{ wordBreak: "break-all", marginBottom: 8 }}>
                  <span className="faint">URL </span>
                  <a href={d.url} target="_blank" rel="noreferrer noopener">{d.url}</a>
                </p>
              )}
              {d.mirror_note && (
                <p className="tiny dim" style={{ marginBottom: 8 }}>
                  <strong>Why this host: </strong>{d.mirror_note}
                </p>
              )}
              {d.notes && (
                <p className="tiny dim" style={{ marginBottom: 8 }}>
                  <strong>Notes: </strong>{d.notes}
                </p>
              )}
              {d.cached_files?.length > 0 && (
                <div className="tiny faint mono">
                  {d.cached_files.map((f) => (
                    <div key={f.file} style={{ wordBreak: "break-all" }}>
                      sha256({f.file}) = {f.sha256.slice(0, 32)}…
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* CRISP-DM timeline                                                   */
/* ------------------------------------------------------------------ */

function PhaseBlock({ phase }: { phase: Phase }) {
  const [showEvidence, setShowEvidence] = useState(false);
  return (
    <div className="phase">
      <div className="phase-name">{phase.title}</div>
      <p className="phase-summary">{phase.summary}</p>

      {phase.decisions?.map((d, i) => (
        <div className="decision" key={i}>
          <div className="decision-q">{d.question}</div>
          <div className="decision-a">→ {d.choice}</div>
          <div className="decision-why">{d.rationale}</div>
          {d.alternatives_rejected?.length > 0 && (
            <div className="rejected">
              <strong>Rejected:</strong>
              <ul style={{ margin: "4px 0 0", paddingLeft: 18 }}>
                {d.alternatives_rejected.map((a, j) => <li key={j}>{a}</li>)}
              </ul>
            </div>
          )}
        </div>
      ))}

      {phase.risks?.length > 0 && (
        <div className="callout warn" style={{ marginTop: 10 }}>
          <div className="callout-title">
            <span aria-hidden="true">▲</span>
            {phase.risks.length === 1 ? "Limitation" : `Limitations (${phase.risks.length})`}
          </div>
          <ul style={{ margin: "4px 0 0", paddingLeft: 18 }}>
            {phase.risks.map((r, i) => <li key={i} style={{ marginBottom: 4 }}>{r}</li>)}
          </ul>
        </div>
      )}

      {phase.evidence && Object.keys(phase.evidence).length > 0 && (
        <>
          <button className="chip" style={{ marginTop: 8 }}
                  onClick={() => setShowEvidence((v) => !v)}>
            {showEvidence ? "Hide" : "Show"} recorded evidence
          </button>
          {showEvidence && (
            <pre className="sample" style={{ marginTop: 8 }}>
              {JSON.stringify(phase.evidence, null, 2)}
            </pre>
          )}
        </>
      )}
    </div>
  );
}

export function PhaseTimeline({ doc }: { doc: CrispDmDoc }) {
  return (
    <div>
      <div className="card" style={{ marginBottom: 18 }}>
        <div className="stat-label">Business question</div>
        <p style={{ margin: "6px 0 0", fontSize: "1.02rem" }}>{doc.business_question}</p>
      </div>
      {doc.phases.map((p) => <PhaseBlock key={p.name} phase={p} />)}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Run provenance footer                                               */
/* ------------------------------------------------------------------ */

export function RunStamp({ run }: { run?: { git_commit: string; seed: number; duration_seconds: number; libraries: Record<string, string> } }) {
  if (!run) return null;
  return (
    <div className="tiny faint mono" style={{ marginTop: 20 }}>
      Generated at commit <code>{run.git_commit}</code> · seed {run.seed} ·{" "}
      {fmt(run.duration_seconds, 1)}s ·{" "}
      {Object.entries(run.libraries)
        .filter(([k]) => k !== "python")
        .map(([k, v]) => `${k} ${v}`)
        .join(" · ")}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Metric table                                                        */
/* ------------------------------------------------------------------ */

export function MetricTable<T extends Record<string, unknown>>({
  rows, columns, highlight,
}: {
  rows: T[];
  columns: { key: string; label: string; num?: boolean; render?: (row: T) => ReactNode }[];
  highlight?: (row: T) => boolean;
}) {
  return (
    <div className="table-scroll">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} className={c.num ? "num" : undefined}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className={highlight?.(row) ? "highlight" : undefined}>
              {columns.map((c) => (
                <td key={c.key} className={c.num ? "num" : undefined}>
                  {c.render
                    ? c.render(row)
                    : typeof row[c.key] === "number"
                      ? fmt(row[c.key] as number, 4)
                      : String(row[c.key] ?? "—")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
