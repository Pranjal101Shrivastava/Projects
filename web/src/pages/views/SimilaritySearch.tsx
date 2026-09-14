import { useState } from "react";
import { useArtifacts } from "../../lib/data";
import { LineChart, ScatterChart, fmt } from "../../components/Charts";
import { Callout, Caveat, MetricTable, Resolved, Section, Stat } from "../../components/UI";

type Config = {
  bands: number; rows: number; threshold_estimate: number;
  candidate_pairs: number; candidate_fraction_of_exact: number;
  verified_pairs: number; true_pairs_found: number; true_pairs_missed: number;
  recall: number; precision: number; f1: number;
  banding_seconds: number; verify_seconds: number; total_seconds: number;
  speedup_vs_exact: number;
  s_curve: { similarity: number; p_candidate: number }[];
};

type Results = {
  exact: { pairs_compared: number; pairs_found: number; seconds: number; threshold: number };
  minhash: {
    n_permutations: number; seconds: number;
    accuracy: {
      mean_error: number; std_error_aggregate: number; max_theoretical_std: number;
      share_of_pairs_below_0_1: number; interpretation: string;
      stratified_by_similarity: {
        similarity_band: string; n_pairs: number; mean_true_similarity: number;
        observed_std: number; theoretical_std: number;
      }[];
      scatter: { true: number; estimated: number }[];
    };
  };
  lsh_configurations: Config[];
  best_by_f1: Config;
  examples: { a: string; b: string; jaccard: number; minhash_estimate: number }[];
};

export default function SimilaritySearch() {
  const state = useArtifacts<{ results: Results }>("09_similarity_search", ["results"]);
  return (
    <Resolved state={state} what="similarity search artifacts">
      {(d) => <Body r={d.results} />}
    </Resolved>
  );
}

function Body({ r }: { r: Results }) {
  const [selected, setSelected] = useState(
    r.lsh_configurations.findIndex((c) => c.bands === r.best_by_f1.bands)
  );
  const config = r.lsh_configurations[Math.max(0, selected)];
  const perfect = r.lsh_configurations.filter((c) => c.recall >= 0.999);
  const acc = r.minhash.accuracy;

  return (
    <>
      <div className="grid grid-4" style={{ marginBottom: 8 }}>
        <Stat label="Exact comparison" value={r.exact.pairs_compared.toLocaleString()}
              sub={`pairs, ${r.exact.seconds.toFixed(1)}s`} />
        <Stat label="True pairs found" value={r.exact.pairs_found.toLocaleString()}
              sub={`Jaccard ≥ ${r.exact.threshold}`} />
        <Stat label="Best recall × speed" value={`${(r.best_by_f1.recall * 100).toFixed(1)}%`}
              tone="accent" sub={`at ${r.best_by_f1.speedup_vs_exact}× — b=${r.best_by_f1.bands}, r=${r.best_by_f1.rows}`} />
        <Stat label="100% recall costs"
              value={perfect.length ? `${perfect[perfect.length - 1].speedup_vs_exact}×` : "—"}
              tone="warn" sub="speedup collapses at full recall" />
      </div>

      <Callout kind="info" title="The point of this project is the trade, not the speed">
        LSH is an <strong>approximation</strong>. Reporting only its speedup reports half of
        a trade. At {r.exact.pairs_compared.toLocaleString()} pairs the exact answer is
        computable, so the recall and precision of the approximation are measured here
        rather than assumed — and the table below is the curve you are choosing a point on.
      </Callout>

      <Section title="The recall/speed trade-off" note="every configuration verified exactly">
        <MetricTable
          rows={r.lsh_configurations}
          highlight={(c) => c.bands === config.bands && c.rows === config.rows}
          columns={[
            { key: "bands", label: "Bands", num: true },
            { key: "rows", label: "Rows", num: true },
            {
              key: "threshold_estimate", label: "S-curve threshold", num: true,
              render: (c) => fmt(c.threshold_estimate, 3),
            },
            {
              key: "candidate_pairs", label: "Candidates", num: true,
              render: (c) => c.candidate_pairs.toLocaleString(),
            },
            {
              key: "recall", label: "Recall", num: true,
              render: (c) => (
                <strong className={c.recall >= 0.9 ? "good-text" : c.recall < 0.3 ? "bad-text" : "warn-text"}>
                  {(c.recall * 100).toFixed(1)}%
                </strong>
              ),
            },
            { key: "true_pairs_missed", label: "Missed", num: true },
            {
              key: "speedup_vs_exact", label: "Speedup", num: true,
              render: (c) => `${c.speedup_vs_exact}×`,
            },
          ]}
        />
        <p className="small dim" style={{ marginTop: 12 }}>
          Precision is 1.000 in every row <em>by construction</em>: LSH is used as a filter
          and every candidate it proposes is then verified exactly. That is the standard
          two-stage pattern — LSH proposes, exact comparison disposes — and it is why the
          verification time is included in the speedup rather than excluded from it.
        </p>
        <Caveat>
          Reaching 100% recall costs almost all of the speed advantage: it requires
          examining {perfect.length ? perfect[perfect.length - 1].candidate_pairs.toLocaleString() : "—"} candidate
          pairs, which then all have to be verified. That is not a tuning failure — it is
          the guarantee LSH offers. It is probabilistic, not exhaustive, and any application
          where a missed pair is unacceptable needs exact search.
        </Caveat>
      </Section>

      <Section title="The S-curve" note="select a configuration to see where its threshold sits">
        <div className="card">
          <div className="row" style={{ marginBottom: 14 }}>
            {r.lsh_configurations.map((c, i) => (
              <button key={`${c.bands}-${c.rows}`}
                      className={`chip${selected === i ? " active" : ""}`}
                      onClick={() => setSelected(i)}>
                b={c.bands}, r={c.rows}
              </button>
            ))}
          </div>
          <LineChart
            height={260}
            xLabel="true Jaccard similarity"
            yLabel="P(becomes a candidate)"
            xMin={0} xMax={1} yMin={0} yMax={1}
            series={[{
              name: `b=${config.bands}, r=${config.rows}`,
              points: config.s_curve.map((p) => ({ x: p.similarity, y: p.p_candidate })),
            }]}
            markers={[{
              x: r.exact.threshold,
              label: `threshold ${r.exact.threshold}`,
              color: "var(--warn)",
            }]}
            showLegend={false}
          />
          <p className="tiny dim" style={{ marginTop: 8 }}>
            P(candidate) = 1 − (1 − s<sup>r</sup>)<sup>b</sup>. The steep region sits near
            (1/b)<sup>1/r</sup> = {fmt(config.threshold_estimate, 3)}. Choosing bands and
            rows is choosing where that step falls — and therefore choosing the
            recall/precision trade explicitly rather than by accident.
          </p>
        </div>
      </Section>

      <Section title="Is MinHash itself accurate?"
               note="error stratified by true similarity — the only comparison that means anything">
        <div className="grid grid-2">
          <div className="card">
            <MetricTable
              rows={acc.stratified_by_similarity}
              columns={[
                { key: "similarity_band", label: "True similarity" },
                { key: "n_pairs", label: "Pairs", num: true },
                { key: "observed_std", label: "Observed sd", num: true, render: (x) => fmt(x.observed_std, 4) },
                { key: "theoretical_std", label: "Theory", num: true, render: (x) => fmt(x.theoretical_std, 4) },
              ]}
            />
            <Callout kind="warn" title="Why an aggregate error figure would mislead here">
              {acc.interpretation}
            </Callout>
          </div>
          <div className="card">
            <h3>Estimated vs true Jaccard</h3>
            <ScatterChart
              height={300}
              xLabel="true Jaccard"
              yLabel="MinHash estimate"
              radius={2}
              opacity={0.45}
              points={acc.scatter.map((p) => ({ x: p.true, y: p.estimated }))}
            />
            <p className="tiny dim" style={{ marginTop: 6 }}>
              {r.minhash.n_permutations} permutations. Points hug the diagonal; the spread
              widens toward the middle because the estimator's variance is s(1−s)/k, maximal
              at s = 0.5 and near zero at the extremes.
            </p>
          </div>
        </div>
      </Section>

      <Section title="What it found" note="top exact matches, with the MinHash estimate beside each">
        <MetricTable
          rows={r.examples.slice(0, 15)}
          columns={[
            { key: "a", label: "Name A" },
            { key: "b", label: "Name B" },
            { key: "jaccard", label: "Exact Jaccard", num: true, render: (x) => <strong>{fmt(x.jaccard, 3)}</strong> },
            { key: "minhash_estimate", label: "MinHash est.", num: true, render: (x) => fmt(x.minhash_estimate, 3) },
          ]}
        />
        <Caveat>
          Ground truth here is "Jaccard above a threshold", not "genuinely the same company".
          String similarity is a proxy: two distinct subsidiaries of one group can score
          highly, and a company that rebranded entirely will score low. The recall figures
          measure agreement with exact string search, not with reality.
        </Caveat>
      </Section>
    </>
  );
}
