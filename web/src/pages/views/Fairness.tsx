import { useState } from "react";
import { useArtifacts } from "../../lib/data";
import { ConfusionMatrix, LineChart, fmt, seriesColor } from "../../components/Charts";
import { Callout, Caveat, MetricTable, Resolved, Section, Stat } from "../../components/UI";

type GroupMetrics = {
  n: number; tp: number; fp: number; fn: number; tn: number;
  base_rate: number; selection_rate: number;
  fpr: number; fnr: number; tpr: number; tnr: number;
  ppv: number; npv: number; accuracy: number;
};

type Criterion = {
  definition: string;
  measured: {
    by_group: Record<string, number>;
    min: number; min_group: string; max: number; max_group: string;
    absolute_difference: number; ratio: number;
  };
  satisfied: boolean;
  note: string;
  gap: number;
  rank_closest_to_parity: number;
  disparate_impact_ratio?: number;
  passes_four_fifths_rule?: boolean;
};

type Fairness = {
  by_group: Record<string, GroupMetrics>;
  criteria: Record<string, Criterion>;
  criteria_summary: {
    tolerance_used: number; n_satisfied: number;
    closest_to_parity: { criterion: string; gap: number };
    furthest_from_parity: { criterion: string; gap: number };
    ratio_furthest_to_closest: number;
    why_ranking_matters: string;
  };
  impossibility: {
    chouldechova_identity: string;
    identity_verified_on_real_data: {
      group: string; base_rate: number;
      observed_fpr: number; fpr_implied_by_identity: number; discrepancy: number;
    }[];
    max_identity_discrepancy: number;
    base_rates: Record<string, number>;
    base_rate_gap: number;
    explanation: string;
  };
  significance: {
    comparison: string; metric: string; fpr_a: number; fpr_b: number;
    chi2: number; p_value: number; significant: boolean; note: string;
  };
  calibration: Record<string, { decile: number; n: number; observed_recidivism_rate: number }[]>;
};

type OwnModel = {
  features_used: string[];
  protected_attribute_excluded: string;
  overall: {
    pr_auc: number; pr_auc_no_skill: number; roc_auc: number; brier: number;
    accuracy_trap: { model_accuracy: number; always_negative_accuracy: number; note: string };
  };
  by_group: Record<string, GroupMetrics>;
  fpr_gap: number;
  compas_fpr_gap: number;
  finding: string;
};

type Profile = {
  rows_raw: number;
  rows_after_propublica_filters: number;
  filters_applied: Record<string, string>;
  overall_recidivism_rate: number;
  overall_high_risk_rate: number;
  group_sizes: Record<string, number>;
  groups_compared: string[];
  groups_too_small_to_compare: { group: string; n: number }[];
  min_group_size: number;
  decile_distribution: Record<string, Record<string, number>>;
};

const CRITERION_LABEL: Record<string, string> = {
  demographic_parity: "Demographic parity",
  equal_opportunity: "Equal opportunity (FNR parity)",
  predictive_equality: "Predictive equality (FPR parity)",
  calibration_ppv: "Calibration / predictive parity (PPV)",
};

export default function Fairness() {
  const state = useArtifacts<{ fairness: Fairness; own_model: OwnModel; profile: Profile }>(
    "10_fairness_audit",
    ["fairness", "own_model", "profile"]
  );
  return (
    <Resolved state={state} what="fairness audit artifacts">
      {(d) => <Body f={d.fairness} own={d.own_model} p={d.profile} />}
    </Resolved>
  );
}

function Body({ f, own, p }: { f: Fairness; own: OwnModel; p: Profile }) {
  const groups = Object.keys(f.by_group);
  const [group, setGroup] = useState(groups[0]);
  const fpr = f.criteria.predictive_equality.measured;
  const ppv = f.criteria.calibration_ppv.measured;
  const ranked = Object.entries(f.criteria).sort(
    (a, b) => a[1].rank_closest_to_parity - b[1].rank_closest_to_parity
  );

  return (
    <>
      <div className="grid grid-4" style={{ marginBottom: 8 }}>
        <Stat label="Defendants audited" value={p.rows_after_propublica_filters.toLocaleString()}
              sub={`of ${p.rows_raw.toLocaleString()} raw records`} />
        <Stat label={`False positive rate · ${fpr.max_group}`} value={`${(fpr.max * 100).toFixed(1)}%`}
              tone="bad" sub="did not reoffend, labelled high risk" />
        <Stat label={`False positive rate · ${fpr.min_group}`} value={`${(fpr.min * 100).toFixed(1)}%`}
              tone="good" sub={`gap ${fmt(fpr.absolute_difference, 4)} · ratio ${fmt(fpr.ratio, 2)}×`} />
        <Stat label="Fairness criteria satisfied" value={`${f.criteria_summary.n_satisfied} of ${ranked.length}`}
              tone="warn" sub={`at a ${f.criteria_summary.tolerance_used} tolerance`} />
      </div>

      <Callout kind="info" title="Both sides of the ProPublica argument are in this data">
        ProPublica pointed at the error rates: among defendants who did <em>not</em> go on to
        reoffend, {fpr.max_group} defendants were labelled high risk{" "}
        {fmt(fpr.ratio, 2)}× as often as {fpr.min_group} ones. Northpointe replied that the
        score means the same thing whoever receives it: among people labelled high risk, the
        reoffending rate is {(ppv.min * 100).toFixed(1)}–{(ppv.max * 100).toFixed(1)}% across
        groups. <strong>Both statements are computed below from the same table, and both are
        true.</strong> The section on impossibility shows why that is not a contradiction.
      </Callout>

      <Section title="What the tool actually did, per group"
               note="one contingency table per group — everything else is derived from these four counts">
        <div className="card">
          <div className="row" style={{ marginBottom: 14 }}>
            {groups.map((g) => (
              <button key={g} className={`chip${group === g ? " active" : ""}`}
                      onClick={() => setGroup(g)}>
                {g} <span className="faint">n={f.by_group[g].n.toLocaleString()}</span>
              </button>
            ))}
          </div>
          <ConfusionMatrix
            tp={f.by_group[group].tp} fp={f.by_group[group].fp}
            fn={f.by_group[group].fn} tn={f.by_group[group].tn}
            positiveLabel="reoffended" negativeLabel="did not"
          />
          <p className="tiny dim" style={{ marginTop: 8 }}>
            Rows are what happened; columns are what COMPAS predicted. The top-right cell is
            the one ProPublica wrote about — {f.by_group[group].fp.toLocaleString()}{" "}
            {group} defendants who did not reoffend within two years and were nonetheless
            scored high risk.
          </p>
        </div>

        <MetricTable
          rows={groups.map((g) => ({ group: g, ...f.by_group[g] }))}
          highlight={(r) => r.group === group}
          columns={[
            { key: "group", label: "Group" },
            { key: "n", label: "n", num: true, render: (r) => r.n.toLocaleString() },
            { key: "base_rate", label: "Reoffended", num: true, render: (r) => `${(r.base_rate * 100).toFixed(1)}%` },
            { key: "selection_rate", label: "Scored high risk", num: true, render: (r) => `${(r.selection_rate * 100).toFixed(1)}%` },
            { key: "fpr", label: "FPR", num: true, render: (r) => <strong className="bad-text">{(r.fpr * 100).toFixed(1)}%</strong> },
            { key: "fnr", label: "FNR", num: true, render: (r) => `${(r.fnr * 100).toFixed(1)}%` },
            { key: "ppv", label: "PPV", num: true, render: (r) => `${(r.ppv * 100).toFixed(1)}%` },
            { key: "accuracy", label: "Accuracy", num: true, render: (r) => `${(r.accuracy * 100).toFixed(1)}%` },
          ]}
        />
        <Callout kind="warn" title="Accuracy is the column to ignore">
          Accuracy is within {fmt(
            (Math.max(...groups.map((g) => f.by_group[g].accuracy)) -
             Math.min(...groups.map((g) => f.by_group[g].accuracy))) * 100, 1)} points
          across every group — which is exactly why a vendor can report it and a journalist
          can still be right. A single accuracy figure averages a false positive and a false
          negative into the same number, and here those two errors fall on different people.
        </Callout>
      </Section>

      <Section title="Four criteria, ranked by distance from parity"
               note="not a pass/fail list — the ordering is the finding">
        <MetricTable
          rows={ranked.map(([key, c]) => ({ key, ...c }))}
          columns={[
            {
              key: "key", label: "Criterion",
              render: (r) => (
                <>
                  <strong>{CRITERION_LABEL[r.key] ?? r.key}</strong>
                  <div className="tiny dim" style={{ marginTop: 2 }}>{r.definition}</div>
                </>
              ),
            },
            {
              key: "gap", label: "Gap", num: true,
              render: (r) => (
                <strong className={r.gap < 0.1 ? "warn-text" : "bad-text"}>{fmt(r.gap, 4)}</strong>
              ),
            },
            { key: "ratio", label: "Ratio", num: true, render: (r) => `${fmt(r.measured.ratio, 2)}×` },
            {
              key: "range", label: "Range across groups", num: true,
              render: (r) => (
                <span className="mono tiny">
                  {fmt(r.measured.min, 3)} ({r.measured.min_group.split("-")[0]}) →{" "}
                  {fmt(r.measured.max, 3)} ({r.measured.max_group.split("-")[0]})
                </span>
              ),
            },
            {
              key: "satisfied", label: "Within tolerance", num: true,
              render: (r) => <span className="bad-text">{r.satisfied ? "yes" : "no"}</span>,
            },
          ]}
        />
        <Callout kind="warn" title="Why a column of four 'no's would have been a worse answer">
          {f.criteria_summary.why_ranking_matters}
        </Callout>
        <div className="grid grid-2" style={{ marginTop: 14 }}>
          {ranked.map(([key, c]) => (
            <div className="card" key={key}>
              <div className="stat-label">#{c.rank_closest_to_parity} · {CRITERION_LABEL[key] ?? key}</div>
              <p className="small" style={{ margin: "8px 0 0" }}>{c.note}</p>
              {c.disparate_impact_ratio !== undefined && (
                <p className="tiny faint mono" style={{ marginTop: 8, marginBottom: 0 }}>
                  Disparate impact ratio {fmt(c.disparate_impact_ratio, 4)} · four-fifths rule:{" "}
                  {c.passes_four_fifths_rule ? "passes" : "fails"}
                </p>
              )}
            </div>
          ))}
        </div>
      </Section>

      <Section title="The impossibility, verified rather than cited"
               note="an algebraic identity — checked against the real counts">
        <div className="card">
          <p className="mono" style={{ fontSize: "1.05rem", margin: "0 0 14px" }}>
            {f.impossibility.chouldechova_identity}
          </p>
          <MetricTable
            rows={f.impossibility.identity_verified_on_real_data}
            columns={[
              { key: "group", label: "Group" },
              { key: "base_rate", label: "Base rate p", num: true, render: (r) => fmt(r.base_rate, 4) },
              { key: "observed_fpr", label: "Observed FPR", num: true, render: (r) => fmt(r.observed_fpr, 4) },
              { key: "fpr_implied_by_identity", label: "FPR the identity forces", num: true, render: (r) => fmt(r.fpr_implied_by_identity, 4) },
              {
                key: "discrepancy", label: "Discrepancy", num: true,
                render: (r) => <span className="good-text mono">{r.discrepancy.toExponential(1)}</span>,
              },
            ]}
          />
          <p className="small" style={{ marginTop: 14 }}>{f.impossibility.explanation}</p>
          <p className="tiny faint" style={{ marginBottom: 0 }}>
            Maximum discrepancy across all groups: {f.impossibility.max_identity_discrepancy.toExponential(1)}{" "}
            — the residual is rounding, not slack. Base rates differ by{" "}
            {fmt(f.impossibility.base_rate_gap, 4)}, and while that holds, equal PPV and equal
            FPR cannot both be achieved by any scoring rule whatsoever. Not by this one, not
            by a better one.
          </p>
        </div>
      </Section>

      <Section title="Is the gap just noise?" note={f.significance.comparison}>
        <div className="card">
          <div className="grid grid-3">
            <Stat label={`FPR · ${f.significance.comparison.split(" vs ")[0]}`} value={fmt(f.significance.fpr_a, 4)} tone="bad" />
            <Stat label={`FPR · ${f.significance.comparison.split(" vs ")[1]}`} value={fmt(f.significance.fpr_b, 4)} tone="good" />
            <Stat label="χ² test" value={`p = ${f.significance.p_value.toExponential(1)}`}
                  tone="accent" sub={`χ² = ${fmt(f.significance.chi2, 1)}`} />
          </div>
          <p className="small dim" style={{ margin: "12px 0 0" }}>{f.significance.note}</p>
        </div>
      </Section>

      <Section title="Calibration by decile" note="does a given score mean the same thing in each group?">
        <div className="card">
          <LineChart
            height={300}
            xLabel="COMPAS decile score"
            yLabel="observed two-year recidivism rate"
            yMin={0} yMax={1}
            series={Object.entries(f.calibration).map(([g, rows], i) => ({
              name: g,
              points: rows.map((r) => ({ x: r.decile, y: r.observed_recidivism_rate })),
              color: seriesColor(i),
            }))}
          />
          <p className="small dim" style={{ marginTop: 8 }}>
            The three curves rise together: a 7 means roughly the same probability of
            reoffending whichever group receives it. This is Northpointe's defence, and on
            this data it largely holds. It coexists with the error-rate gap above because the
            groups arrive at the score with different base rates — which is the identity in
            the previous section, seen as a picture.
          </p>
        </div>
      </Section>

      <Section title="Score distribution" note="the shape ProPublica published, recomputed">
        <div className="card">
          <LineChart
            height={280}
            xLabel="decile score"
            yLabel="share of group"
            xMin={1} xMax={10} yMin={0}
            series={p.groups_compared.map((g, i) => {
              const dist = p.decile_distribution[g];
              const total = Object.values(dist).reduce((a, b) => a + b, 0);
              return {
                name: g,
                color: seriesColor(i),
                points: Object.entries(dist).map(([d, n]) => ({ x: Number(d), y: n / total })),
              };
            })}
          />
          <p className="small dim" style={{ marginTop: 8 }}>
            Scores for {p.groups_compared[1]} defendants pile up at the low end and thin out
            steadily; for {p.groups_compared[0]} defendants the distribution is close to flat.
            No individual score is wrong in the calibration sense — the difference is in how
            the population is spread across the scale, which is what turns into the error-rate
            gap once a high-risk threshold is drawn.
          </p>
        </div>
      </Section>

      <Section title="Would dropping race from the model fix it?"
               note="a replacement model trained without the protected attribute">
        <div className="grid grid-4">
          <Stat label="Features used" value={own.features_used.length}
                sub={`'${own.protected_attribute_excluded}' deliberately excluded`} />
          <Stat label="PR-AUC vs no-skill" value={fmt(own.overall.pr_auc, 3)}
                sub={`floor ${fmt(own.overall.pr_auc_no_skill, 3)} · Brier ${fmt(own.overall.brier, 3)}`} />
          <Stat label="FPR gap · own model" value={fmt(own.fpr_gap, 4)} tone="warn"
                sub="race never seen in training" />
          <Stat label="FPR gap · COMPAS" value={fmt(own.compas_fpr_gap, 4)} tone="bad"
                sub={`own model closes ${((1 - own.fpr_gap / own.compas_fpr_gap) * 100).toFixed(0)}% of it`} />
        </div>
        <Callout kind="bad" title="Fairness through unawareness does not work">
          {own.finding}
        </Callout>
        <MetricTable
          rows={Object.keys(own.by_group).map((g) => ({
            group: g,
            compas_fpr: f.by_group[g]?.fpr ?? NaN,
            own_fpr: own.by_group[g].fpr,
            compas_ppv: f.by_group[g]?.ppv ?? NaN,
            own_ppv: own.by_group[g].ppv,
          }))}
          columns={[
            { key: "group", label: "Group" },
            { key: "compas_fpr", label: "COMPAS FPR", num: true, render: (r) => fmt(r.compas_fpr, 4) },
            { key: "own_fpr", label: "Own model FPR", num: true, render: (r) => <strong>{fmt(r.own_fpr, 4)}</strong> },
            { key: "compas_ppv", label: "COMPAS PPV", num: true, render: (r) => fmt(r.compas_ppv, 4) },
            { key: "own_ppv", label: "Own model PPV", num: true, render: (r) => <strong>{fmt(r.own_ppv, 4)}</strong> },
          ]}
        />
        <p className="small dim" style={{ marginTop: 12 }}>
          Accuracy trap, stated so it cannot be quoted out of context: the replacement model
          scores {fmt(own.overall.accuracy_trap.model_accuracy * 100, 1)}% accuracy, while a
          rule that predicts "will not reoffend" for everybody scores{" "}
          {fmt(own.overall.accuracy_trap.always_negative_accuracy * 100, 1)}%.
        </p>
      </Section>

      <Section title="What this audit does not establish">
        <Caveat>
          <strong>The label is re-arrest, not reoffending.</strong> Every figure on this page
          treats a recorded arrest within two years as ground truth for "committed another
          crime". Policing intensity is not uniform across neighbourhoods or groups, so the
          base rates that drive the impossibility result are themselves measured through a
          process that may be biased. A gap in measured base rates is not proof of a gap in
          underlying behaviour, and nothing here can separate the two.
        </Caveat>
        <Caveat>
          <strong>Scope.</strong> One county (Broward, Florida), one two-year window, and the
          {" "}{p.rows_after_propublica_filters.toLocaleString()} records that survive
          ProPublica's published filters out of {p.rows_raw.toLocaleString()}. Groups below{" "}
          {p.min_group_size} people — {p.groups_too_small_to_compare.map((g) => `${g.group} (n=${g.n})`).join(", ")}{" "}
          — are excluded from every comparison rather than reported with intervals too wide to
          mean anything. That exclusion is itself a choice with consequences: the smallest
          groups are the ones least likely to be audited anywhere.
        </Caveat>
        <Caveat>
          <strong>A threshold was imposed.</strong> COMPAS emits a 1–10 decile; "high risk"
          here means decile ≥ 5, following ProPublica. Every error-rate figure moves if that
          line moves. The calibration chart is threshold-free, which is part of why it and
          the error-rate table can disagree so sharply.
        </Caveat>
      </Section>
    </>
  );
}
