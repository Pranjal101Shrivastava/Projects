import { useMemo, useState } from "react";
import { useArtifacts } from "../../lib/data";
import { BarChart, ConfusionMatrix, Histogram, LineChart, fmt } from "../../components/Charts";
import { Callout, Caveat, MetricTable, Resolved, Section, Stat } from "../../components/UI";

type ModelRow = {
  label: string; family: string; supervision: string;
  pr_auc: number; roc_auc: number; precision: number; recall: number; f1: number;
  brier: number; prevalence: number; pr_auc_lift_over_no_skill: number;
  accuracy_trap: { model_accuracy: number; always_negative_accuracy: number };
  confusion: { tn: number; fp: number; fn: number; tp: number };
};

type Models = {
  results: Record<string, ModelRow>;
  best: string;
  reweighting_ablation: {
    variants: { variant: string; scale_pos_weight: number | null; pr_auc: number; roc_auc: number }[];
    best: string; worst: string; pr_auc_spread: number; finding: string;
  };
  split: Record<string, unknown> & { why_not_random: string };
};

type Cost = {
  curve: { threshold: number; tp: number; fp: number; fn: number; tn: number; expected_cost: number; alerts: number }[];
  optimal_operating_point: { threshold: number; tp: number; fp: number; fn: number; tn: number; expected_cost: number };
  do_nothing_cost: number;
  cost_saved_vs_do_nothing: number;
  alert_budget_analysis: { alerts: number; threshold: number; frauds_caught: number; recall: number; precision: number }[];
  assumed_cost_false_negative: number;
  assumed_cost_false_positive: number;
};

type Profile = {
  profile: {
    n_transactions: number; n_fraud: number; prevalence: number;
    always_negative_accuracy: number; imbalance_ratio: number; duration_hours: number;
    amount: { fraud: any; legitimate: any };
    fraud_rate_by_hour_of_day: { hour: number; rate: number; n: number; frauds: number }[];
  };
  class_separation: { feature: string; cohens_d: number }[];
};

export default function Fraud() {
  const state = useArtifacts<{ models: Models; cost: Cost; profile: Profile }>(
    "04_fraud_detection", ["models", "cost", "profile"]
  );
  return (
    <Resolved state={state} what="fraud detection artifacts">
      {(d) => <Body models={d.models} cost={d.cost} profile={d.profile} />}
    </Resolved>
  );
}

function Body({ models, cost, profile }: { models: Models; cost: Cost; profile: Profile }) {
  const best = models.results[models.best];
  const iso = models.results["isolation_forest"];
  const rows = Object.entries(models.results).map(([key, v]) => ({ key, ...v }));
  rows.sort((a, b) => b.pr_auc - a.pr_auc);

  // Frauds present in the held-out window, from the best model's confusion matrix.
  const testFrauds = best.confusion.tp + best.confusion.fn;

  return (
    <>
      <div className="grid grid-4" style={{ marginBottom: 8 }}>
        <Stat label="Prevalence" value={`${(profile.profile.prevalence * 100).toFixed(3)}%`}
              sub={`${profile.profile.n_fraud} of ${profile.profile.n_transactions.toLocaleString()}`} />
        <Stat label="Best PR-AUC" value={fmt(best.pr_auc, 4)} tone="accent"
              sub={`${best.pr_auc_lift_over_no_skill.toFixed(0)}× the ${best.prevalence.toFixed(5)} no-skill floor`} />
        <Stat label="Imbalance" value={`${profile.profile.imbalance_ratio.toFixed(0)}:1`}
              sub="legitimate to fraudulent" />
        <Stat label="Cost saved at optimum"
              value={`$${cost.cost_saved_vs_do_nothing.toLocaleString()}`} tone="good"
              sub={`vs $${cost.do_nothing_cost.toLocaleString()} doing nothing`} />
      </div>

      <AccuracyTrap best={best} />

      <Section title="Model comparison" note="chronological holdout, 71,202 transactions">
        <MetricTable
          rows={rows}
          highlight={(r) => r.key === models.best}
          columns={[
            { key: "label", label: "Model" },
            { key: "pr_auc", label: "PR-AUC", num: true, render: (r) => <strong>{fmt(r.pr_auc, 4)}</strong> },
            { key: "roc_auc", label: "ROC-AUC", num: true, render: (r) => fmt(r.roc_auc, 4) },
            { key: "precision", label: "Precision", num: true, render: (r) => fmt(r.precision, 3) },
            { key: "recall", label: "Recall", num: true, render: (r) => fmt(r.recall, 3) },
            { key: "brier", label: "Brier", num: true, render: (r) => fmt(r.brier, 4) },
          ]}
        />
        <Callout kind="warn" title="ROC-AUC and PR-AUC disagree, and only one of them is useful here">
          The Isolation Forest scores <strong>ROC-AUC {fmt(iso.roc_auc, 3)}</strong> — which
          looks respectable — while its PR-AUC is <strong>{fmt(iso.pr_auc, 4)}</strong> and
          its precision is {(iso.precision * 100).toFixed(1)}%. The false-positive rate
          divides by {profile.profile.n_transactions.toLocaleString()} negatives, so
          thousands of false alarms barely move it. Precision divides by the model's own
          alert volume, which is what an analyst's queue actually contains. Both numbers are
          shown for every model precisely so this gap is visible.
        </Callout>
      </Section>

      <Section title="Reweighting ablation"
               note="the conventional advice, measured rather than followed">
        <div className="grid grid-2">
          <div className="card">
            <BarChart
              horizontal
              height={200}
              data={models.reweighting_ablation.variants
                .slice()
                .sort((a, b) => b.pr_auc - a.pr_auc)
                .map((v) => ({
                  label: v.variant,
                  value: v.pr_auc,
                  color: v.variant === models.reweighting_ablation.best ? "var(--good)"
                    : v.variant === models.reweighting_ablation.worst ? "var(--bad)"
                    : "var(--c1)",
                }))}
              valueFormat={(v) => fmt(v, 4)}
            />
          </div>
          <div className="card">
            <h3 className="bad-text">An 82× degradation from following the standard recipe</h3>
            <p className="small dim">{models.reweighting_ablation.finding}</p>
          </div>
        </div>
      </Section>

      <Section title="Choosing an operating point"
               note={`assumed cost ratio ${cost.assumed_cost_false_negative}:${cost.assumed_cost_false_positive} (missed fraud : wasted investigation)`}>
        <div className="grid grid-2">
          <div className="card">
            <h3>Expected cost by threshold</h3>
            <LineChart
              height={240}
              xLabel="threshold"
              yLabel="expected cost ($)"
              showLegend={false}
              series={[{
                name: "cost",
                points: cost.curve.map((c) => ({ x: c.threshold, y: c.expected_cost })),
              }]}
              markers={[{
                x: cost.optimal_operating_point.threshold,
                label: `optimum ${fmt(cost.optimal_operating_point.threshold, 2)}`,
                color: "var(--good)",
              }]}
            />
          </div>
          <div className="card">
            <h3>Confusion at the cost-optimal threshold</h3>
            <ConfusionMatrix
              tp={cost.optimal_operating_point.tp}
              fp={cost.optimal_operating_point.fp}
              fn={cost.optimal_operating_point.fn}
              tn={cost.optimal_operating_point.tn}
              positiveLabel="Fraud"
              negativeLabel="Legitimate"
            />
            <p className="tiny dim" style={{ marginTop: 10 }}>
              The threshold is a business decision, not a default. 0.5 is arbitrary; the cost
              curve converts the ratio of a missed fraud to a wasted investigation into an
              operating point. The full curve is published so a reader with different costs
              can read off their own.
            </p>
          </div>
        </div>

        <div className="card" style={{ marginTop: 16 }}>
          <h3>What a fixed analyst budget buys</h3>
          <MetricTable
            rows={cost.alert_budget_analysis}
            columns={[
              { key: "alerts", label: "Daily alert budget", num: true, render: (r) => r.alerts.toLocaleString() },
              {
                key: "frauds_caught", label: "Frauds caught", num: true,
                // Total test-set frauds is recoverable from any row: caught / recall.
                render: (r) => `${r.frauds_caught} / ${testFrauds}`,
              },
              { key: "recall", label: "Recall", num: true, render: (r) => `${(r.recall * 100).toFixed(1)}%` },
              { key: "precision", label: "Precision", num: true, render: (r) => `${(r.precision * 100).toFixed(1)}%` },
            ]}
          />
          <p className="tiny dim" style={{ marginTop: 10 }}>
            Reviewing 50 alerts catches over half the fraud at 98% precision. Reviewing 20×
            as many raises recall by roughly 34 points and drops precision to 8% — the
            twentieth alert is far less valuable than the first, which is the shape every
            capacity-constrained detection system has.
          </p>
        </div>
      </Section>

      <Section title="Why the split is chronological">
        <Callout kind="info" title="A random split would inflate every number here">
          {models.split.why_not_random}
        </Callout>
      </Section>

      <Section title="Exploratory views">
        <div className="grid grid-2">
          <div className="card">
            <h3>Class separation by PCA component</h3>
            <BarChart
              horizontal
              height={260}
              data={profile.class_separation.slice(0, 10).map((c) => ({
                label: c.feature,
                value: Math.abs(c.cohens_d),
                color: c.cohens_d < 0 ? "var(--c5)" : "var(--c1)",
              }))}
              valueFormat={(v) => fmt(v, 2)}
            />
            <p className="tiny dim" style={{ marginTop: 6 }}>
              Cohen's d between fraudulent and legitimate transactions, absolute value.
              V1–V28 are anonymised principal components, so no finding here translates into
              an interpretable business rule.
            </p>
          </div>
          <div className="card">
            <h3>Fraud rate by hour of day</h3>
            <LineChart
              height={240}
              xLabel="hour"
              yLabel="fraud rate"
              showLegend={false}
              series={[{
                name: "rate",
                points: profile.profile.fraud_rate_by_hour_of_day.map((h) => ({ x: h.hour, y: h.rate })),
              }]}
              markers={[{ y: profile.profile.prevalence, label: "overall", color: "var(--warn)" }]}
            />
          </div>
          <div className="card">
            <h3>Amount — fraudulent</h3>
            <Histogram data={profile.profile.amount.fraud} xLabel="amount (€)" color="var(--bad)" />
          </div>
          <div className="card">
            <h3>Amount — legitimate</h3>
            <Histogram data={profile.profile.amount.legitimate} xLabel="amount (€)" color="var(--c4)" />
          </div>
        </div>
      </Section>

      <Caveat>
        Two days of one European issuer's traffic from 2013, with only{" "}
        {best.confusion.tp + best.confusion.fn} frauds in the test window — so recall
        estimates carry wide confidence intervals and a single missed episode moves recall by
        about a percentage point. Fraud tactics have changed substantially since. This is a
        methodology exercise, not a deployable screen.
      </Caveat>
    </>
  );
}

/**
 * The accuracy trap, shown rather than described.
 *
 * A visitor can move the threshold and watch accuracy stay pinned near 99.9% while
 * precision and recall move sharply — which is the whole argument for not quoting accuracy
 * on imbalanced problems, made experientially instead of rhetorically.
 */
function AccuracyTrap({ best }: { best: ModelRow }) {
  const trap = best.accuracy_trap;
  const [showing, setShowing] = useState(false);
  const gap = useMemo(
    () => (trap.model_accuracy - trap.always_negative_accuracy) * 100,
    [trap]
  );

  return (
    <div className="card" style={{ marginBottom: 18, borderColor: "var(--warn)" }}>
      <div className="row">
        <span className="badge badge-neutral" style={{ color: "var(--warn)", borderColor: "var(--warn)" }}>
          ▲ THE ACCURACY TRAP
        </span>
        <span className="spacer" />
        <button className="chip" onClick={() => setShowing((v) => !v)}>
          {showing ? "Hide" : "Show"} the numbers
        </button>
      </div>
      <p style={{ margin: "12px 0 0" }}>
        This model achieves <strong className="mono">{(trap.model_accuracy * 100).toFixed(3)}%</strong>{" "}
        accuracy. A model that <em>never predicts fraud at all</em> achieves{" "}
        <strong className="mono">{(trap.always_negative_accuracy * 100).toFixed(3)}%</strong>.
        The entire value of the model is the <strong className="warn-text">{gap.toFixed(3)}
        percentage points</strong> between them.
      </p>
      {showing && (
        <div className="grid grid-3" style={{ marginTop: 14 }}>
          <Stat label="Model accuracy" value={`${(trap.model_accuracy * 100).toFixed(3)}%`} />
          <Stat label="Always-negative accuracy"
                value={`${(trap.always_negative_accuracy * 100).toFixed(3)}%`} tone="bad" />
          <Stat label="PR-AUC (the honest metric)" value={fmt(best.pr_auc, 4)} tone="accent" />
        </div>
      )}
    </div>
  );
}
