import { useState } from "react";
import type { ProjectMeta } from "../lib/projects";
import { useArtifacts, type CrispDmDoc, type ProvenanceDoc } from "../lib/data";
import {
  PhaseTimeline, ProvenanceTable, Resolved, RunStamp, Section, Tabs,
} from "../components/UI";
import { PROJECT_VIEWS } from "./views";

/**
 * Shell shared by all eight project pages.
 *
 * Three tabs are identical everywhere — Findings, Method, Data — because they answer the
 * same three questions for every project: what was found, how was it decided, and where
 * did the data come from. Only the Findings tab differs per project, supplied by
 * PROJECT_VIEWS.
 */
export default function ProjectPage({ meta }: { meta: ProjectMeta }) {
  const [tab, setTab] = useState("findings");
  const state = useArtifacts<{ crispdm: CrispDmDoc; provenance: ProvenanceDoc }>(
    meta.id, ["crispdm", "provenance"]
  );
  const View = PROJECT_VIEWS[meta.id];

  return (
    <div className="wrap">
      <header className="hero">
        <div className="row" style={{ marginBottom: 10 }}>
          <span className="project-num">PROJECT {meta.number}</span>
          <span className="badge badge-accent">{meta.domain}</span>
          <span className="badge badge-real">● REAL DATA</span>
        </div>
        <h1 style={{ marginBottom: 10 }}>{meta.title}</h1>
        <p className="lead">{meta.description}</p>
        <div className="row tiny faint mono" style={{ marginTop: 6 }}>
          <span>{meta.dataset}</span>
          <span>·</span>
          <span>{meta.rows}</span>
        </div>
      </header>

      <Tabs
        tabs={[
          { key: "findings", label: "Findings" },
          { key: "method", label: "Method (CRISP-DM)" },
          { key: "data", label: "Data provenance" },
        ]}
        active={tab}
        onChange={setTab}
      />

      {tab === "findings" && (
        <div style={{ paddingBottom: 30 }}>
          {View ? <View /> : <p className="dim">No view registered for this project.</p>}
        </div>
      )}

      {tab === "method" && (
        <Section
          title="Decisions, evidence and limitations"
          note="recorded by the pipeline as it ran, not written afterwards"
        >
          <Resolved state={state} what="the CRISP-DM record">
            {(d) => (
              <>
                <PhaseTimeline doc={d.crispdm} />
                <RunStamp run={d.crispdm._run} />
              </>
            )}
          </Resolved>
        </Section>
      )}

      {tab === "data" && (
        <Section
          title="Where this data came from"
          note="origin, licence and content hash for every source"
        >
          <Resolved state={state} what="provenance records">
            {(d) => (
              <>
                <ProvenanceTable datasets={d.provenance.datasets} note={d.provenance.note} />
                {d.provenance.datasets.length > 0 && (
                  <p className="small dim" style={{ marginTop: 16 }}>
                    Sources are declared once in <code>lib/dsx/data.py</code> and cached under{" "}
                    <code>.data/</code>. On first download the SHA-256 of the payload is
                    recorded; later runs verify the cached copy against it, so an upstream file
                    that changes underneath the work raises an error instead of silently
                    shifting every metric downstream.
                  </p>
                )}
                <RunStamp run={d.provenance._run} />
              </>
            )}
          </Resolved>
        </Section>
      )}
    </div>
  );
}
