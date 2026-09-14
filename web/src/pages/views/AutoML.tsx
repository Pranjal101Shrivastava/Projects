import { useState } from "react";
import { useArtifacts } from "../../lib/data";
import { BarChart, fmt } from "../../components/Charts";
import { Callout, Caveat, MetricTable, Resolved, Section, Stat } from "../../components/UI";

type LeaderboardRow = {
  model: string; inductive_bias: string;
  cv_pr_auc_mean: number | null; cv_pr_auc_std: number | null;
  test_pr_auc: number; test_roc_auc: number; test_brier: number; test_f1: number;
  search_seconds: number; best_params: Record<string, unknown>;
  generalisation_gap: number | null;
};

type Tournament = {
  label: string; n_features: number; prevalence: number;
  leaderboard: LeaderboardRow[]; best: LeaderboardRow; search_protocol: string;
};

type Leak = {
  diagnostic: {
    column: string; univariate_roc_auc: number; correlation_with_target: number;
    mean_duration_subscribed: number; mean_duration_declined: number;
    zero_duration_rows: number; zero_duration_subscriptions: number;
    why_this_is_a_leak: string;
  };
  inflation: {
    clean_best_pr_auc: number; leaked_best_pr_auc: number;
    absolute_inflation: number; relative_inflation: number;
    clean_best_model: string; leaked_best_model: string;
    per_model: { model: string; clean_pr_auc: number; leaked_pr_auc: number | null }[];
    verdict: string;
  };
};

export default function AutoML() {
  const state = useArtifacts<{
    leak_demonstration: Leak;
    tournament_clean: Tournament;
    tournament_leaked: Tournament;
  }>("06_automl_tournament", ["leak_demonstration", "tournament_clean", "tournament_leaked"]);

  return (
    <Resolved state={state} what="AutoML artifacts">
      {(d) => (
        <Body leak={d.leak_demonstration} clean={d.tournament_clean} leaked={d.tournament_leaked} />
      )}
    </Resolved>
  );
}

function Body({ leak, clean, leaked }: { leak: Leak; clean: Tournament; leaked: Tournament }) {
  const [showing, setShowing] = useState<"clean" | "leaked">("clean");
  const t = showing === "clean" ? clean : leaked;
  const inf = leak.inflation;
  const diag = leak.diagnostic;

  return (
    <>
      <div className="grid grid-4" style={{ marginBottom: 8 }}>
        <Stat label="Deployable PR-AUC" value={fmt(inf.clean_best_pr_auc, 4)} tone="good"
              sub="leak-free — the honest result" />
        <Stat label="With the leak" value={fmt(inf.leaked_best_pr_auc, 4)} tone="bad"
              sub="what most published notebooks report" />
        <Stat label="Inflation" value={`+${(inf.relative_inflation * 100).toFixed(1)}%`} tone="warn"
              sub="from a single column" />
        <Stat label="Leaking column alone" value={fmt(diag.univariate_roc_auc, 4)}
              sub="ROC-AUC with no other feature" />
      </div>

      <div className="card" style={{ borderColor: "var(--bad)", marginBottom: 20 }}>
        <div className="row">
          <span className="badge badge-bad">✕ TARGET LEAKAGE</span>
          <strong>The `{diag.column}` column</strong>
        </div>
        <p style={{ margin: "12px 0 0" }}>{diag.why_this_is_a_leak}</p>
        <div className="grid grid-4" style={{ marginTop: 14 }}>
          <Stat label="Mean call — subscribed" value={`${diag.mean_duration_subscribed}s`} />
          <Stat label="Mean call — declined" value={`${diag.mean_duration_declined}s`} />
          <Stat label="Zero-duration calls" value={diag.zero_duration_rows} />
          <Stat label="…that subscribed" value={diag.zero_duration_subscriptions} tone="bad"
                sub="exactly as the causal argument predicts" />
        </div>
      </div>

      <Section title="The same tournament, run twice"
               note="identical protocol; the only difference is one column">
        <div className="card">
          <BarChart
            horizontal
            height={240}
            data={inf.per_model.flatMap((m) => [
              { label: `${m.model} — clean`, value: m.clean_pr_auc, color: "var(--good)" },
              { label: `${m.model} — leaked`, value: m.leaked_pr_auc ?? 0, color: "var(--bad)" },
            ])}
            valueFormat={(v) => fmt(v, 4)}
          />
          <p className="small dim" style={{ marginTop: 12 }}>{inf.verdict}</p>
        </div>
      </Section>

      <Section title="Leaderboards" note="toggle to compare">
        <div className="row" style={{ marginBottom: 14 }}>
          <button className={`chip${showing === "clean" ? " active" : ""}`}
                  onClick={() => setShowing("clean")}>
            Leak-free (deployable)
          </button>
          <button className={`chip${showing === "leaked" ? " active" : ""}`}
                  onClick={() => setShowing("leaked")}>
            With leak (demonstration only)
          </button>
          <span className="spacer" />
          <span className={`badge ${showing === "clean" ? "badge-real" : "badge-bad"}`}>
            {showing === "clean" ? "● honest" : "● inflated"}
          </span>
        </div>

        <MetricTable
          rows={t.leaderboard}
          highlight={(r) => r.model === t.best.model}
          columns={[
            { key: "model", label: "Model" },
            {
              key: "test_pr_auc", label: "Test PR-AUC", num: true,
              render: (r) => <strong>{fmt(r.test_pr_auc, 4)}</strong>,
            },
            {
              key: "cv_pr_auc_mean", label: "CV PR-AUC", num: true,
              render: (r) =>
                r.cv_pr_auc_mean === null ? "—" :
                `${fmt(r.cv_pr_auc_mean, 4)} ± ${fmt(r.cv_pr_auc_std ?? 0, 3)}`,
            },
            { key: "test_roc_auc", label: "ROC-AUC", num: true, render: (r) => fmt(r.test_roc_auc, 4) },
            { key: "test_brier", label: "Brier", num: true, render: (r) => fmt(r.test_brier, 4) },
            {
              key: "generalisation_gap", label: "CV → test gap", num: true,
              render: (r) =>
                r.generalisation_gap === null ? "—" : (
                  <span className={Math.abs(r.generalisation_gap) > 0.03 ? "warn-text" : undefined}>
                    {r.generalisation_gap > 0 ? "+" : ""}{fmt(r.generalisation_gap, 4)}
                  </span>
                ),
            },
            { key: "search_seconds", label: "Search", num: true, render: (r) => `${fmt(r.search_seconds, 0)}s` },
          ]}
        />
        <p className="tiny dim" style={{ marginTop: 10 }}>{t.search_protocol}</p>

        <div className="card" style={{ marginTop: 16 }}>
          <h3>Inductive biases searched</h3>
          <div className="stack">
            {t.leaderboard.map((r) => (
              <div key={r.model} className="row small" style={{ alignItems: "flex-start" }}>
                <span className="badge badge-neutral" style={{ minWidth: 170, justifyContent: "flex-start" }}>
                  {r.model}
                </span>
                <span className="dim" style={{ flex: 1, minWidth: 200 }}>{r.inductive_bias}</span>
              </div>
            ))}
          </div>
          <p className="tiny dim" style={{ marginTop: 12 }}>
            A search over boosting depths is a hyperparameter tuner, not AutoML. Including a
            linear model and a deliberately naive Bayes reference shows how much of the final
            score comes from model capacity and how much was available from any reasonable
            baseline — here Gaussian NB reaches{" "}
            {fmt(clean.leaderboard.find((r) => r.model === "gaussian_nb")?.test_pr_auc ?? 0, 3)}{" "}
            against the winner's {fmt(clean.best.test_pr_auc, 3)}.
          </p>
        </div>
      </Section>

      <Section title="How to catch a leak before it reaches a leaderboard">
        <div className="grid grid-2">
          <Callout kind="info" title="Test 1 — implausible univariate power">
            Does any single column predict the target implausibly well on its own? Here{" "}
            <code>{diag.column}</code> alone reaches ROC-AUC{" "}
            <strong>{fmt(diag.univariate_roc_auc, 4)}</strong>. That is a red flag, not a
            feature. It is a necessary check but not sufficient — a genuinely strong feature
            can also score highly.
          </Callout>
          <Callout kind="good" title="Test 2 — causal timing (the decisive one)">
            Would this value exist at the moment the prediction must be made?{" "}
            <code>{diag.column}</code> fails outright: it is generated by the very event
            being predicted. A prioritisation model has to score{" "}
            <em>before dialling</em>, when call length does not yet exist. No statistical
            test is needed to see this — only reasoning about the order of events.
          </Callout>
        </div>
      </Section>

      <Caveat>
        The leak-free leaderboard is the deployable one. Any comparison against published
        results on this dataset that include <code>duration</code> is not like-for-like.
        Macroeconomic columns are retained in the leak-free run — they are knowable at
        scoring time, but they make the model partly a function of the economic cycle, so
        performance will drift as conditions change. Randomised search with 12 draws per
        family is also a light budget, so the ranking between close families should be read
        as provisional.
      </Caveat>
    </>
  );
}
