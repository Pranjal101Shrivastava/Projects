import { useMemo, useState } from "react";
import { useArtifacts } from "../../lib/data";
import { BarChart, fmt } from "../../components/Charts";
import { Callout, Caveat, MetricTable, Resolved, Section, Stat } from "../../components/UI";

type Rule = {
  antecedent: string[]; consequent: string[]; support: number; support_count: number;
  confidence: number; lift: number; leverage: number; conviction: number | null;
  zhang: number; p_value: number; q_value: number; significant: boolean;
};

type Rules = {
  significance: { n_rules_tested: number; n_significant: number; n_rejected_by_fdr: number; fdr_alpha: number; rationale: string };
  top_by_lift: Rule[];
  top_by_leverage: Rule[];
  dissociations: Rule[];
  graph: { nodes: string[]; edges: { source: string; target: string; lift: number; support: number }[] };
};

type Algorithms = {
  comparison: {
    identical_output: boolean; n_itemsets: number; speedup_fp_over_apriori: number;
    apriori: { seconds: number; database_scans: number; candidates_generated: number; pruning_rate: number; levels: any[] };
    fp_growth: { seconds: number; database_scans: number; tree_nodes_created: number };
    interpretation: string;
  };
  itemsets_by_size: Record<string, { n: number; top: { items: string[]; count: number; support: number }[] }>;
};

type Profile = {
  n_transactions: number; n_distinct_items: number; sparsity: number;
  basket_size: { mean: number; median: number; max: number; singleton_share: number };
  basket_size_distribution: { size: number; count: number }[];
  top_items: { item: string; count: number; support: number }[];
};

const setLabel = (items: string[]) => `{${items.join(", ")}}`;

export default function MarketBasket() {
  const state = useArtifacts<{ rules: Rules; algorithms: Algorithms; profile: Profile }>(
    "03_market_basket", ["rules", "algorithms", "profile"]
  );
  return (
    <Resolved state={state} what="association mining artifacts">
      {(d) => <Body rules={d.rules} algorithms={d.algorithms} profile={d.profile} />}
    </Resolved>
  );
}

function Body({ rules, algorithms, profile }: { rules: Rules; algorithms: Algorithms; profile: Profile }) {
  const c = algorithms.comparison;
  const sig = rules.significance;

  return (
    <>
      <div className="grid grid-4" style={{ marginBottom: 8 }}>
        <Stat label="Baskets" value={profile.n_transactions.toLocaleString()}
              sub={`${profile.n_distinct_items} item categories`} />
        <Stat label="FP-Growth speedup" value={`${c.speedup_fp_over_apriori.toFixed(0)}×`}
              tone="accent" sub={`${c.apriori.seconds}s → ${c.fp_growth.seconds}s`} />
        <Stat label="Rules surviving FDR" value={sig.n_significant.toLocaleString()}
              sub={`of ${sig.n_rules_tested.toLocaleString()} tested`} />
        <Stat label="Rules rejected" value={sig.n_rejected_by_fdr.toLocaleString()} tone="warn"
              sub={`α = ${sig.fdr_alpha}`} />
      </div>

      <Callout kind="good" title="Two independent implementations, byte-identical output">
        {c.interpretation} The pipeline raises an <code>AssertionError</code> and refuses to
        write results if they ever disagree — two algorithms agreeing on all{" "}
        {c.n_itemsets.toLocaleString()} itemsets is far stronger evidence of correctness than
        either one completing without error.
      </Callout>

      <Section title="Apriori versus FP-Growth" note="same input, same output, very different cost">
        <div className="grid grid-2">
          <div className="card">
            <h3>Where the time goes</h3>
            <MetricTable
              rows={[
                { metric: "Wall-clock time", apriori: `${c.apriori.seconds}s`, fp: `${c.fp_growth.seconds}s` },
                { metric: "Database scans", apriori: String(c.apriori.database_scans), fp: String(c.fp_growth.database_scans) },
                { metric: "Candidates generated", apriori: c.apriori.candidates_generated.toLocaleString(), fp: "0 — none" },
                { metric: "Pruned by downward closure", apriori: `${(c.apriori.pruning_rate * 100).toFixed(1)}%`, fp: "n/a" },
                { metric: "Tree nodes built", apriori: "n/a", fp: c.fp_growth.tree_nodes_created.toLocaleString() },
                { metric: "Frequent itemsets found", apriori: c.n_itemsets.toLocaleString(), fp: c.n_itemsets.toLocaleString() },
              ]}
              columns={[
                { key: "metric", label: "" },
                { key: "apriori", label: "Apriori", num: true },
                { key: "fp", label: "FP-Growth", num: true },
              ]}
            />
          </div>
          <div className="card">
            <h3>Apriori's lattice pruning by level</h3>
            <BarChart
              height={230}
              yLabel="itemsets"
              data={c.apriori.levels.flatMap((l: any) => [
                { label: `k=${l.k} cand`, value: l.candidates ?? 0, color: "var(--c8)" },
                { label: `k=${l.k} freq`, value: l.frequent ?? 0, color: "var(--c1)" },
              ])}
              valueFormat={(v) => v.toLocaleString()}
            />
            <p className="tiny dim" style={{ marginTop: 6 }}>
              Apriori pays one database pass per level. The downward-closure property — every
              subset of a frequent itemset is frequent — lets it discard{" "}
              {(c.apriori.pruning_rate * 100).toFixed(0)}% of candidates before counting
              them. FP-Growth avoids generating candidates at all.
            </p>
          </div>
        </div>
      </Section>

      <RuleExplorer rules={rules} />

      <Section title="Dissociations" note="pairs bought together LESS often than chance">
        <MetricTable
          rows={rules.dissociations.slice(0, 8)}
          columns={[
            { key: "rule", label: "Rule", render: (r) => <span className="mono">{setLabel(r.antecedent)} → {setLabel(r.consequent)}</span> },
            { key: "zhang", label: "Zhang", num: true, render: (r) => <span className="bad-text">{fmt(r.zhang, 3)}</span> },
            { key: "lift", label: "Lift", num: true, render: (r) => fmt(r.lift, 3) },
            { key: "support_count", label: "Baskets", num: true },
          ]}
        />
        <p className="small dim" style={{ marginTop: 12 }}>
          Lift below 1 means the pair co-occurs less than independence predicts, but lift
          cannot express <em>how strongly</em> — it is bounded below by 0 and compresses all
          negative association into a narrow range. Zhang's metric spans [−1, 1] and
          separates them properly. Beer dissociating from milk and vegetables is the clearest
          case: these are different shopping trips, not complementary products.
        </p>
      </Section>

      <Section title="The multiple-comparisons problem">
        <Callout kind="warn" title={`${sig.n_rejected_by_fdr.toLocaleString()} rules were tested and discarded`}>
          {sig.rationale}
        </Callout>
        <p className="small dim">
          Benjamini-Hochberg was chosen over Bonferroni deliberately. With{" "}
          {sig.n_rules_tested.toLocaleString()} simultaneous tests, Bonferroni controls the
          family-wise error rate but would reject genuine affinities along with the noise.
          BH bounds the <em>expected proportion</em> of false discoveries among those
          reported, which is the right guarantee for a ranked shortlist a merchandiser will
          act on. The implementation was verified against{" "}
          <code>statsmodels.multipletests</code>: maximum absolute difference in q-values
          1.1 × 10⁻¹⁶, identical rejection sets.
        </p>
      </Section>

      <Section title="Basket structure">
        <div className="grid grid-2">
          <div className="card">
            <h3>Most frequent items</h3>
            <BarChart
              horizontal
              height={300}
              data={profile.top_items.slice(0, 14).map((i) => ({ label: i.item, value: i.support }))}
              valueFormat={(v) => `${(v * 100).toFixed(1)}%`}
            />
          </div>
          <div className="card">
            <h3>Basket size distribution</h3>
            <BarChart
              height={300}
              xLabel="items in basket"
              yLabel="baskets"
              data={profile.basket_size_distribution.slice(0, 20).map((b) => ({
                label: String(b.size), value: b.count,
              }))}
              valueFormat={(v) => v.toLocaleString()}
            />
            <p className="tiny dim" style={{ marginTop: 6 }}>
              Mean {profile.basket_size.mean} items,{" "}
              {(profile.basket_size.singleton_share * 100).toFixed(1)}% are single-item
              baskets. The transaction matrix is {(profile.sparsity * 100).toFixed(1)}%
              sparse, which is why the support floor has to be set at 0.1% rather than the
              conventional 1% — at 1% only staples survive and every rule becomes a variation
              on "people buy milk".
            </p>
          </div>
        </div>
      </Section>

      <Caveat>
        Association is not causation. A lift of 3 between two items does not mean that moving
        one next to the other will raise sales of the other — it may only mean both belong to
        the same kind of shopping trip. Rules are also mined and evaluated on the same month
        of transactions from a single outlet, so persistence across periods and stores is
        untested.
      </Caveat>
    </>
  );
}

/**
 * Rule explorer.
 *
 * The measure selector is the point of the widget: switching the ranking between lift,
 * leverage and confidence reorders the table completely, which demonstrates far more
 * directly than prose that "the best rules" is a function of which measure you picked.
 */
function RuleExplorer({ rules }: { rules: Rules }) {
  const [measure, setMeasure] = useState<"lift" | "leverage" | "confidence">("lift");
  const [minSupport, setMinSupport] = useState(0);
  const [onlySignificant, setOnlySignificant] = useState(true);

  const pool = useMemo(
    () => [...rules.top_by_lift, ...rules.top_by_leverage].filter(
      (r, i, arr) => arr.findIndex((o) => setLabel(o.antecedent) + setLabel(o.consequent) === setLabel(r.antecedent) + setLabel(r.consequent)) === i
    ),
    [rules]
  );

  const shown = useMemo(() => {
    let out = pool.filter((r) => r.support_count >= minSupport);
    if (onlySignificant) out = out.filter((r) => r.significant);
    return out.sort((a, b) => (b[measure] as number) - (a[measure] as number)).slice(0, 18);
  }, [pool, measure, minSupport, onlySignificant]);

  return (
    <Section title="Rule explorer" note="ranking changes completely with the measure chosen">
      <div className="card">
        <div className="controls">
          <div className="control">
            <span className="control-label">Rank by</span>
            <div className="row">
              {(["lift", "leverage", "confidence"] as const).map((m) => (
                <button key={m} className={`chip${measure === m ? " active" : ""}`}
                        onClick={() => setMeasure(m)}>
                  {m}
                </button>
              ))}
            </div>
          </div>
          <div className="control" style={{ minWidth: 200 }}>
            <label className="control-label" htmlFor="minsup">
              Minimum baskets — {minSupport}
            </label>
            <input id="minsup" type="range" min={0} max={120} value={minSupport}
                   onChange={(e) => setMinSupport(Number(e.target.value))} />
          </div>
          <div className="control">
            <span className="control-label">FDR filter</span>
            <button className={`chip${onlySignificant ? " active" : ""}`}
                    onClick={() => setOnlySignificant((v) => !v)}>
              {onlySignificant ? "significant only" : "all rules"}
            </button>
          </div>
        </div>

        <MetricTable
          rows={shown}
          columns={[
            {
              key: "rule", label: "Rule",
              render: (r) => (
                <span className="mono small">
                  {setLabel(r.antecedent)} <span className="accent-text">→</span> {setLabel(r.consequent)}
                </span>
              ),
            },
            { key: "support_count", label: "Baskets", num: true },
            { key: "confidence", label: "Conf", num: true, render: (r) => fmt(r.confidence, 3) },
            { key: "lift", label: "Lift", num: true, render: (r) => <strong>{fmt(r.lift, 2)}</strong> },
            { key: "leverage", label: "Leverage", num: true, render: (r) => fmt(r.leverage, 5) },
            { key: "zhang", label: "Zhang", num: true, render: (r) => fmt(r.zhang, 3) },
            {
              key: "q_value", label: "q-value", num: true,
              render: (r) => (
                <span className={r.significant ? "good-text" : "bad-text"}>
                  {r.q_value < 1e-4 ? r.q_value.toExponential(1) : fmt(r.q_value, 4)}
                </span>
              ),
            },
          ]}
        />
        <p className="tiny dim" style={{ marginTop: 10 }}>
          Switch the measure and watch the table reorder. Confidence favours whatever is
          popular overall — "anything → whole milk" scores well because a quarter of baskets
          contain milk. Lift corrects for that but rewards rare coincidences. Leverage keeps
          absolute volume in view. No single measure is sufficient, which is why six are
          computed.
        </p>
      </div>
    </Section>
  );
}
