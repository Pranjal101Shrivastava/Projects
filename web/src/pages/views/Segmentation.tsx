
import { useArtifacts } from "../../lib/data";
import { LineChart, ScatterChart, fmt, seriesColor } from "../../components/Charts";
import { Callout, Caveat, MetricTable, Resolved, Section, Stat } from "../../components/UI";

type Selection = {
  sweeps: Record<string, { k: number; silhouette: number; calinski_harabasz: number; davies_bouldin: number; inertia: number }[]>;
  consensus: Record<string, { votes: Record<string, number>; unanimous: boolean }>;
  stability: Record<string, { k: number; mean_ari: number; std_ari: number; min_ari: number; stable: boolean }[]>;
  stability_threshold: number;
  n_bootstrap: number;
  chosen_k: number;
  chosen_feature_set: string;
  stability_bar_met: boolean;
  algorithms: Record<string, { label: string; assumption: string; silhouette: number; davies_bouldin: number; calinski_harabasz: number; bic?: number }>;
  cross_algorithm_agreement: { kmeans_vs_gmm: number; kmeans_vs_agglomerative: number; interpretation: string };
};

type Segments = {
  external_validation: {
    overall_conversion: number;
    segment_profiles: {
      segment: number; size: number; share: number; conversion_rate: number;
      lift_vs_overall: number; median_age: number; top_job: string;
      top_education: string; top_contact: string; mean_campaign_calls: number;
      prior_contact_share: number;
    }[];
    conversion_spread: number; chi2: number; p_value: number; significant: boolean;
    interpretation: string;
  };
  projection: {
    explained_variance_ratio: number[];
    points: { x: number; y: number; segment: number; outcome: number }[];
    caveat: string;
  };
};

export default function Segmentation() {
  const state = useArtifacts<{ selection: Selection; segments: Segments }>(
    "02_customer_segmentation", ["selection", "segments"]
  );
  return (
    <Resolved state={state} what="segmentation artifacts">
      {(d) => <Body selection={d.selection} segments={d.segments} />}
    </Resolved>
  );
}

function Body({ selection, segments }: { selection: Selection; segments: Segments }) {
  const ev = segments.external_validation;
  const fs = selection.chosen_feature_set;
  const sweep = selection.sweeps[fs];
  const stability = selection.stability[fs];
  const votes = selection.consensus[fs].votes;
  const stableKs = stability.filter((s) => s.stable).map((s) => s.k);
  const best = ev.segment_profiles[0];
  const worst = ev.segment_profiles[ev.segment_profiles.length - 1];

  return (
    <>
      <div className="grid grid-4" style={{ marginBottom: 8 }}>
        <Stat label="Segments" value={selection.chosen_k} tone="accent"
              sub={`chosen for reproducibility, not geometry`} />
        <Stat label="Conversion spread" value={`${(ev.conversion_spread * 100).toFixed(1)}pp`}
              tone="good" sub={`${(worst.conversion_rate * 100).toFixed(2)}% → ${(best.conversion_rate * 100).toFixed(2)}%`} />
        <Stat label="χ² across segments" value={fmt(ev.chi2, 0)}
              sub={`p ${ev.p_value < 1e-10 ? "< 1e-10" : ev.p_value.toExponential(1)}`} />
        <Stat label="Bootstrap resamples" value={selection.n_bootstrap}
              sub={`ARI bar ${selection.stability_threshold}`} />
      </div>

      <Callout kind="warn" title="Geometry and reproducibility disagreed — and reproducibility won">
        All three validity indices unanimously prefer <strong>k = {votes.silhouette}</strong>{" "}
        (silhouette, Calinski-Harabasz and Davies-Bouldin all vote the same way). But only{" "}
        <strong>k ∈ {`{${stableKs.join(", ")}}`}</strong> survives bootstrap resampling at
        ARI ≥ {selection.stability_threshold}. A k=2 split would be the cleanest
        geometrically and close to useless commercially. Internal indices optimise a
        mathematical objective; they say nothing about whether a partition reproduces or
        whether anyone can act on it.
      </Callout>

      <Section title="Choosing k" note="three indices, then a stability stress test">
        <div className="grid grid-2">
          <div className="card">
            <h3>Validity indices by k</h3>
            <LineChart
              height={250}
              xLabel="k"
              series={[
                { name: "silhouette", points: sweep.map((s) => ({ x: s.k, y: s.silhouette })) },
                {
                  name: "Davies-Bouldin (lower better)", color: seriesColor(2),
                  points: sweep.map((s) => ({ x: s.k, y: s.davies_bouldin })),
                },
              ]}
            />
          </div>
          <div className="card">
            <h3>Bootstrap stability (adjusted Rand index)</h3>
            <LineChart
              height={250}
              xLabel="k"
              yLabel="mean ARI"
              showLegend={false}
              series={[{ name: "mean ARI", points: stability.map((s) => ({ x: s.k, y: s.mean_ari })) }]}
              markers={[{
                y: selection.stability_threshold,
                label: `bar ${selection.stability_threshold}`,
                color: "var(--warn)",
              }]}
            />
            <p className="tiny dim" style={{ marginTop: 6 }}>
              Each k is refitted on {selection.n_bootstrap} resamples and compared to the
              reference partition. A segmentation that changes when you resample the same
              population describes the sample, not the customers. This check is what almost
              no segmentation write-up performs.
            </p>
          </div>
        </div>
      </Section>

      <Section title="Do the segments matter?"
               note="validated against an outcome the clustering never saw">
        <MetricTable
          rows={ev.segment_profiles}
          highlight={(r) => r.segment === best.segment}
          columns={[
            { key: "segment", label: "Seg", num: true },
            { key: "size", label: "Customers", num: true, render: (r) => r.size.toLocaleString() },
            { key: "share", label: "Share", num: true, render: (r) => `${(r.share * 100).toFixed(1)}%` },
            {
              key: "conversion_rate", label: "Conversion", num: true,
              render: (r) => (
                <strong className={r.lift_vs_overall > 1.2 ? "good-text" : r.lift_vs_overall < 0.6 ? "bad-text" : undefined}>
                  {(r.conversion_rate * 100).toFixed(2)}%
                </strong>
              ),
            },
            { key: "lift_vs_overall", label: "Lift", num: true, render: (r) => `${fmt(r.lift_vs_overall, 2)}×` },
            { key: "top_job", label: "Modal job" },
            { key: "top_contact", label: "Contact" },
            { key: "mean_campaign_calls", label: "Mean calls", num: true, render: (r) => fmt(r.mean_campaign_calls, 1) },
            {
              key: "prior_contact_share", label: "Prior contact", num: true,
              render: (r) => `${(r.prior_contact_share * 100).toFixed(0)}%`,
            },
          ]}
        />

        <div className="grid grid-2" style={{ marginTop: 16 }}>
          <Callout kind="good" title={`Segment ${best.segment}: 63.8% conversion`}>
            {best.size.toLocaleString()} customers, {(best.prior_contact_share * 100).toFixed(0)}%
            of whom were contacted in a previous campaign. Converting at{" "}
            {fmt(best.lift_vs_overall, 2)}× the overall rate. Legitimate as a predictor — prior
            contact is known before dialling — but this segment largely re-identifies
            already-engaged customers rather than revealing a latent group.
          </Callout>
          <Callout kind="bad" title={`Segment ${worst.segment}: budget being actively wasted`}>
            {worst.size.toLocaleString()} customers averaging{" "}
            <strong>{fmt(worst.mean_campaign_calls, 1)} campaign calls</strong> and converting
            at {(worst.conversion_rate * 100).toFixed(2)}% — about a third of baseline. This
            is the actionable finding: a saturated cohort absorbing calls that would convert
            better elsewhere.
          </Callout>
        </div>
      </Section>

      <Section title="Algorithm agreement" note="how much structure is real vs imposed">
        <div className="grid grid-3">
          {Object.entries(selection.algorithms).map(([key, a]) => (
            <div className="card" key={key}>
              <strong>{a.label}</strong>
              <p className="tiny dim" style={{ margin: "4px 0 10px" }}>{a.assumption}</p>
              <div className="row tiny mono">
                <span>silhouette {fmt(a.silhouette, 3)}</span>
                <span className="spacer" />
                <span>DB {fmt(a.davies_bouldin, 2)}</span>
              </div>
            </div>
          ))}
        </div>
        <Caveat>
          K-means agrees with the Gaussian mixture at ARI{" "}
          <strong>{fmt(selection.cross_algorithm_agreement.kmeans_vs_gmm, 2)}</strong> and
          with Ward linkage at{" "}
          <strong>{fmt(selection.cross_algorithm_agreement.kmeans_vs_agglomerative, 2)}</strong>{" "}
          — moderate, not strong. Roughly half the partition structure is therefore imposed
          by K-means's spherical assumption rather than present in the data. These segment
          boundaries should be read as one defensible partition among several, not as
          discovered natural kinds.
        </Caveat>
      </Section>

      <Section title="Segment projection">
        <div className="card">
          <ScatterChart
            height={360}
            xLabel={`PC1 (${(segments.projection.explained_variance_ratio[0] * 100).toFixed(1)}% variance)`}
            yLabel={`PC2 (${(segments.projection.explained_variance_ratio[1] * 100).toFixed(1)}% variance)`}
            radius={2.2}
            opacity={0.55}
            points={segments.projection.points.map((p) => ({
              x: p.x, y: p.y, c: p.segment,
              label: `segment ${p.segment}\n${p.outcome ? "subscribed" : "declined"}`,
            }))}
            categories={ev.segment_profiles
              .slice()
              .sort((a, b) => a.segment - b.segment)
              .map((s) => `Segment ${s.segment} — ${(s.conversion_rate * 100).toFixed(1)}%`)}
          />
          <Caveat>{segments.projection.caveat}</Caveat>
        </div>
      </Section>

      <Section title="Why 'duration' was excluded from the feature space">
        <Callout kind="info" title="The same leak that inflates Project 06 by 40%">
          The source dataset contains a <code>duration</code> column recording how long the
          sales call lasted. It is only known once the call has happened and the outcome is
          effectively decided. Clustering on it would build segments that partly encode the
          answer, so it is excluded here entirely — and the subscription outcome itself is
          held out as the external validation signal.{" "}
          <a href="#/p/automl">Project 06 measures what including it costs.</a>
        </Callout>
      </Section>
    </>
  );
}
