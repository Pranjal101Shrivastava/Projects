import { useState } from "react";
import { useArtifacts } from "../../lib/data";
import { Histogram, LineChart, fmt, seriesColor } from "../../components/Charts";
import { Callout, Caveat, MetricTable, Resolved, Section, Stat } from "../../components/UI";

type Perf = {
  total_return: number; annualised_return: number; annualised_volatility: number;
  sharpe: number; sortino: number; max_drawdown: number; hit_rate: number;
  n_periods: number; best_day: number; worst_day: number;
};

type Strategy = {
  folds: { fold: number; n_train: number; n_test: number; mae: number; directional_accuracy: number }[];
  directional_accuracy: number;
  information_coefficient: number;
  ic_p_value: number;
  ic_significant: boolean;
  r2_on_returns: number;
  n_trades: number;
  time_in_market: number;
  gross: Perf;
  net: Perf;
  total_cost_drag: number;
};

type Results = {
  results: Record<string, Strategy>;
  verdict: {
    benchmark_gross: Perf;
    benchmark_net: Perf;
    comparison: {
      strategy: string; net_annualised: number; net_sharpe: number; max_drawdown: number;
      beats_buy_and_hold_return: boolean; beats_buy_and_hold_sharpe: boolean;
    }[];
    any_strategy_beats_benchmark: boolean;
    cost_bps: number;
    breakeven_note: string;
    conclusion: string;
  };
  equity_curves: Record<string, number[]>;
};

type Profile = {
  profile: {
    n_days: number; date_min: string; date_max: string;
    price_start: number; price_end: number; buy_and_hold_total_return: number;
    max_adjustment_vs_close: number; adjustment_note: string;
    daily_return: { bin_centers: number[]; counts: number[]; mean: number; median: number; n: number; std: number };
    return_stats: { mean: number; std: number; skew: number; excess_kurtosis: number; annualised_volatility: number };
    stationarity: { returns_adf_pvalue: number; price_adf_pvalue: number; note: string };
  };
  random_walk_trap: {
    r2_predicting_price_with_yesterdays_price: number;
    r2_predicting_return_with_zero: number;
    explanation: string;
  };
  preparation: {
    n_features: number; features: string[]; n_rows_modelled: number; horizon_days: number;
    label_overlap: string; leakage_controls: string[];
    lookahead_test: {
      test: string; cut_row: number; perturbation: string;
      n_features_checked: number; n_rows_checked: number; max_drift: number;
      passed: boolean; note: string;
    };
  };
};

const pct = (v: number, d = 2) => `${(v * 100).toFixed(d)}%`;

export default function Backtest() {
  const state = useArtifacts<{ results: Results; profile: Profile }>(
    "12_market_backtest",
    ["results", "profile"]
  );
  return (
    <Resolved state={state} what="backtest artifacts">
      {(d) => <Body r={d.results} p={d.profile} />}
    </Resolved>
  );
}

function Body({ r, p }: { r: Results; p: Profile }) {
  const [costs, setCosts] = useState<"net" | "gross">("net");
  const names = Object.keys(r.results);
  const bench = costs === "net" ? r.verdict.benchmark_net : r.verdict.benchmark_gross;
  const trap = p.random_walk_trap;

  return (
    <>
      <div className="grid grid-4" style={{ marginBottom: 8 }}>
        <Stat label="Instrument" value={`AAPL · ${p.profile.n_days} days`}
              sub={`${p.profile.date_min} → ${p.profile.date_max}`} />
        <Stat label="Strategies beating buy-and-hold"
              value={`${r.verdict.comparison.filter((c) => c.beats_buy_and_hold_sharpe).length} of ${r.verdict.comparison.length}`}
              tone="bad" sub="risk-adjusted, after costs" />
        <Stat label="Information coefficient" tone="warn"
              value={fmt(r.results[names[0]].information_coefficient, 4)}
              sub={`p = ${fmt(r.results[names[0]].ic_p_value, 3)} — not distinguishable from 0`} />
        <Stat label="Cost assumption" value={`${r.verdict.cost_bps} bps`} tone="accent"
              sub="round trip, charged on every position change" />
      </div>

      <Callout kind="bad" title="This project reports a negative result, deliberately">
        {r.verdict.conclusion}
      </Callout>

      <Section title="The trap this project is built to avoid"
               note="the same model, scored two ways">
        <div className="grid grid-2">
          <div className="card">
            <div className="stat-label">Predicting tomorrow's PRICE with today's price</div>
            <div className="stat-value good" style={{ fontSize: "2.4rem" }}>
              R² = {fmt(trap.r2_predicting_price_with_yesterdays_price, 4)}
            </div>
            <p className="tiny faint" style={{ marginBottom: 0 }}>
              Publishable-looking. Completely worthless.
            </p>
          </div>
          <div className="card">
            <div className="stat-label">The identical model, restated as a RETURN forecast</div>
            <div className="stat-value bad" style={{ fontSize: "2.4rem" }}>
              R² = {fmt(trap.r2_predicting_return_with_zero, 4)}
            </div>
            <p className="tiny faint" style={{ marginBottom: 0 }}>
              Same model, same data, same information: none.
            </p>
          </div>
        </div>
        <Callout kind="warn" title="Why the first number is an artefact">
          {trap.explanation}
        </Callout>
        <p className="small dim">
          The statistical backing: an augmented Dickey-Fuller test rejects a unit root in
          returns (p = {fmt(p.profile.stationarity.returns_adf_pvalue, 4)}) and fails to reject
          it in price (p = {fmt(p.profile.stationarity.price_adf_pvalue, 3)}).{" "}
          {p.profile.stationarity.note}
        </p>
      </Section>

      <Section title="Did the models predict anything?" note="signal measured before any trading logic">
        <MetricTable
          rows={names.map((n) => ({ name: n, ...r.results[n] }))}
          columns={[
            { key: "name", label: "Model", render: (x) => <strong>{x.name}</strong> },
            {
              key: "information_coefficient", label: "IC (rank corr.)", num: true,
              render: (x) => (
                <strong className={x.ic_significant ? "good-text" : "bad-text"}>
                  {fmt(x.information_coefficient, 4)}
                </strong>
              ),
            },
            { key: "ic_p_value", label: "p-value", num: true, render: (x) => fmt(x.ic_p_value, 4) },
            {
              key: "directional_accuracy", label: "Directional accuracy", num: true,
              render: (x) => (
                <span className={x.directional_accuracy > 0.5 ? undefined : "bad-text"}>
                  {pct(x.directional_accuracy, 1)}
                </span>
              ),
            },
            { key: "r2_on_returns", label: "R² on returns", num: true, render: (x) => fmt(x.r2_on_returns, 4) },
            { key: "n_trades", label: "Trades", num: true },
            { key: "time_in_market", label: "Time in market", num: true, render: (x) => pct(x.time_in_market, 1) },
          ]}
        />
        <Callout kind="info" title="Directional accuracy near 50% is the honest reading">
          A coin flip scores 50%. Both models land within a few points of it, and the
          information coefficients are small, negative, and carry p-values above 0.05 — so the
          direction of the sign is not evidence of anything either. Reporting a strategy's
          returns without first showing that its <em>predictions</em> have measurable signal is
          how backtests get published; the IC column is the check that comes first.
        </Callout>
      </Section>

      <Section title="Purged walk-forward folds" note="expanding window, chronological, with a purge band">
        <div className="grid grid-2">
          {names.map((n, i) => (
            <div className="card" key={n}>
              <h3>{n}</h3>
              <LineChart
                height={220}
                xLabel="fold"
                yLabel="directional accuracy"
                yMin={0.2} yMax={0.8}
                series={[{
                  name: n,
                  color: seriesColor(i),
                  points: r.results[n].folds.map((f) => ({ x: f.fold, y: f.directional_accuracy })),
                }]}
                markers={[{ y: 0.5, label: "coin flip", color: "var(--warn)" }]}
                showLegend={false}
              />
              <MetricTable
                rows={r.results[n].folds}
                columns={[
                  { key: "fold", label: "Fold", num: true },
                  { key: "n_train", label: "Train rows", num: true },
                  { key: "n_test", label: "Test rows", num: true },
                  { key: "mae", label: "MAE", num: true, render: (f) => fmt(f.mae, 5) },
                  {
                    key: "directional_accuracy", label: "Dir. acc.", num: true,
                    render: (f) => (
                      <span className={f.directional_accuracy >= 0.5 ? "good-text" : "bad-text"}>
                        {pct(f.directional_accuracy, 1)}
                      </span>
                    ),
                  },
                ]}
              />
            </div>
          ))}
        </div>
        <Callout kind="warn" title="Fold-to-fold variance is the point of showing all five">
          Directional accuracy swings from {pct(Math.min(...names.flatMap((n) => r.results[n].folds.map((f) => f.directional_accuracy))), 1)}{" "}
          to {pct(Math.max(...names.flatMap((n) => r.results[n].folds.map((f) => f.directional_accuracy))), 1)}{" "}
          across folds of {r.results[names[0]].folds[0].n_test} rows each. Any single fold could
          be quoted as a triumph or a disaster. Averaging them is not a formality — it is the
          only reading that survives having chosen the window honestly, in advance, rather than
          after seeing the results.
        </Callout>
        <p className="small dim">
          {p.preparation.label_overlap}
        </p>
      </Section>

      <Section title="Equity curves" note="toggle costs — this is where the result is decided">
        <div className="card">
          <div className="row" style={{ marginBottom: 14 }}>
            <button className={`chip${costs === "gross" ? " active" : ""}`} onClick={() => setCosts("gross")}>
              Gross (no costs)
            </button>
            <button className={`chip${costs === "net" ? " active" : ""}`} onClick={() => setCosts("net")}>
              Net of {r.verdict.cost_bps} bps
            </button>
          </div>
          <LineChart
            height={320}
            xLabel="trading day within the out-of-sample window"
            yLabel="growth of 1 unit"
            series={[
              {
                name: "buy & hold",
                color: "var(--fg-dim)",
                dashed: true,
                points: r.equity_curves.buy_and_hold.map((v, i) => ({ x: i, y: v })),
              },
              ...names.map((n, i) => ({
                name: n,
                color: seriesColor(i),
                points: r.equity_curves[n].map((v, j) => ({ x: j, y: v })),
              })),
            ]}
            markers={[{ y: 1, label: "break-even", color: "var(--border)" }]}
          />
          <p className="tiny dim" style={{ marginTop: 8 }}>
            The plotted strategy curves are gross of costs; the table below carries both. Note
            that the benchmark curve is longer than the strategy curves — a model needs training
            history before it can trade, so the comparison window is the shorter one and every
            figure in the table is computed over each series' own periods.
          </p>
        </div>

        <MetricTable
          rows={[
            { name: "buy & hold", ...bench, isBench: true },
            ...names.map((n) => ({ name: n, ...r.results[n][costs], isBench: false })),
          ]}
          highlight={(x) => x.isBench}
          columns={[
            { key: "name", label: costs === "net" ? "After costs" : "Before costs", render: (x) => <strong>{x.name}</strong> },
            { key: "total_return", label: "Total return", num: true, render: (x) => <span className={x.total_return >= 0 ? "good-text" : "bad-text"}>{pct(x.total_return)}</span> },
            { key: "annualised_return", label: "Annualised", num: true, render: (x) => pct(x.annualised_return) },
            { key: "annualised_volatility", label: "Volatility", num: true, render: (x) => pct(x.annualised_volatility) },
            { key: "sharpe", label: "Sharpe", num: true, render: (x) => <strong className={x.sharpe >= 0 ? "good-text" : "bad-text"}>{fmt(x.sharpe, 3)}</strong> },
            { key: "sortino", label: "Sortino", num: true, render: (x) => fmt(x.sortino, 3) },
            { key: "max_drawdown", label: "Max drawdown", num: true, render: (x) => <span className="bad-text">{pct(x.max_drawdown)}</span> },
            { key: "hit_rate", label: "Hit rate", num: true, render: (x) => pct(x.hit_rate, 1) },
            { key: "n_periods", label: "Periods", num: true },
          ]}
        />

        <div className="grid grid-2" style={{ marginTop: 14 }}>
          <div className="card">
            <h3>What costs alone did</h3>
            <MetricTable
              rows={names.map((n) => ({
                name: n,
                gross_sharpe: r.results[n].gross.sharpe,
                net_sharpe: r.results[n].net.sharpe,
                drag: r.results[n].total_cost_drag,
                trades: r.results[n].n_trades,
              }))}
              columns={[
                { key: "name", label: "Model" },
                { key: "trades", label: "Trades", num: true },
                { key: "gross_sharpe", label: "Sharpe gross", num: true, render: (x) => fmt(x.gross_sharpe, 3) },
                { key: "net_sharpe", label: "Sharpe net", num: true, render: (x) => <strong className="bad-text">{fmt(x.net_sharpe, 3)}</strong> },
                { key: "drag", label: "Cost drag", num: true, render: (x) => pct(x.drag) },
              ]}
            />
          </div>
          <div className="card">
            <h3>Break-even arithmetic</h3>
            <p className="small" style={{ marginTop: 0 }}>{r.verdict.breakeven_note}</p>
            <p className="tiny faint" style={{ marginBottom: 0 }}>
              Both models here were already losing money before costs, so costs are not the
              reason this study failed — but they are the reason a study that looks marginally
              profitable gross is usually not worth running.
            </p>
          </div>
        </div>
      </Section>

      <Section title="Proving there is no look-ahead" note="a perturbation test, not an assertion">
        <div className="card">
          <div className="grid grid-4">
            <Stat label="Test" value={p.preparation.lookahead_test.test} />
            <Stat label="Perturbation" value={`+50%`} sub={`from row ${p.preparation.lookahead_test.cut_row} onward`} />
            <Stat label="Features checked" value={p.preparation.lookahead_test.n_features_checked}
                  sub={`across ${p.preparation.lookahead_test.n_rows_checked} earlier rows`} />
            <Stat label="Max drift in earlier rows" value={fmt(p.preparation.lookahead_test.max_drift, 1)}
                  tone="good" sub={p.preparation.lookahead_test.passed ? "test passed" : "test FAILED"} />
          </div>
          <p className="small" style={{ margin: "12px 0 0" }}>
            Future price and volume bars are multiplied by 1.5 from a cut point onward and the
            whole feature matrix is rebuilt. If any feature value at an <em>earlier</em> row
            changes, that feature reads the future. None moved, to the last bit.
          </p>
          <p className="small dim">{p.preparation.lookahead_test.note}</p>
          <div style={{ marginTop: 10 }}>
            <div className="stat-label">Leakage controls in force</div>
            <ul className="small" style={{ margin: "6px 0 0", paddingLeft: 18 }}>
              {p.preparation.leakage_controls.map((c, i) => <li key={i} style={{ marginBottom: 4 }}>{c}</li>)}
            </ul>
          </div>
        </div>
      </Section>

      <Section title="The series itself" note="what any strategy on this instrument is working against">
        <div className="grid grid-2">
          <div className="card">
            <h3>Daily return distribution</h3>
            <Histogram
              height={240}
              xLabel="daily return"
              data={{
                bin_centers: p.profile.daily_return.bin_centers,
                counts: p.profile.daily_return.counts,
                mean: p.profile.daily_return.mean,
                median: p.profile.daily_return.median,
                n: p.profile.daily_return.n,
              }}
              showMean
            />
            <p className="tiny dim" style={{ marginTop: 6 }}>
              Excess kurtosis {fmt(p.profile.return_stats.excess_kurtosis, 2)} and skew{" "}
              {fmt(p.profile.return_stats.skew, 3)}: fat-tailed and slightly left-leaning, not
              Gaussian. Sharpe ratios assume otherwise, which is one more reason to read the
              maximum drawdown column beside them.
            </p>
          </div>
          <div className="card">
            <h3>Sample facts</h3>
            <MetricTable
              rows={[
                { k: "Window", v: `${p.profile.date_min} → ${p.profile.date_max}` },
                { k: "Trading days", v: p.profile.n_days.toLocaleString() },
                { k: "Price start → end", v: `${fmt(p.profile.price_start, 2)} → ${fmt(p.profile.price_end, 2)}` },
                { k: "Buy & hold, full sample", v: pct(p.profile.buy_and_hold_total_return) },
                { k: "Annualised volatility", v: pct(p.profile.return_stats.annualised_volatility) },
                { k: "Forecast horizon", v: `${p.preparation.horizon_days} days` },
                { k: "Features", v: `${p.preparation.n_features} — ${p.preparation.features.slice(0, 4).join(", ")}…` },
                { k: "Rows modelled", v: p.preparation.n_rows_modelled.toLocaleString() },
              ]}
              columns={[
                { key: "k", label: "" },
                { key: "v", label: "", num: true, render: (x) => <span className="mono tiny">{x.v}</span> },
              ]}
            />
            <p className="tiny dim" style={{ marginTop: 6, marginBottom: 0 }}>
              {p.profile.adjustment_note}
            </p>
          </div>
        </div>
      </Section>

      <Section title="Why this result should not be trusted as a conclusion about markets">
        <Caveat>
          <strong>One instrument, one two-year window.</strong> AAPL between{" "}
          {p.profile.date_min} and {p.profile.date_max} is a single draw from the space of
          possible studies. Finding no edge here is evidence about this sample, not about
          momentum, machine learning, or equities in general. A study that claimed to have{" "}
          <em>found</em> an edge on this same sample would be making the stronger and far less
          defensible claim.
        </Caveat>
        <Caveat>
          <strong>The instrument was chosen with hindsight, and that cuts both ways.</strong>{" "}
          AAPL exists in 2026 and had a liquid, survivable history — a selection the market did
          not make in advance. Survivorship works in favour of buy-and-hold here, which makes
          the benchmark harder to beat than a fair universe would be. The strategies still lost
          to it on a risk-adjusted basis, but that asymmetry belongs in the reading.
        </Caveat>
        <Caveat>
          <strong>Costs are modelled, not incurred.</strong> A flat {r.verdict.cost_bps} bps
          round trip stands in for spread, commission, and slippage combined; there is no
          market-impact model, no borrow cost on short positions, no financing, and no
          assumption that an order actually fills at the close it is priced against. Real
          execution would make these results worse, not better.
        </Caveat>
        <Caveat>
          <strong>Two models were tried, and that is already a multiple-comparisons problem.</strong>{" "}
          The p-values reported for the information coefficients are not corrected for having
          fit two model families across fourteen features. Had either come out significant,
          that correction would have to be applied before the number meant anything — and the
          number of hypotheses silently tested is the usual reason published backtests fail out
          of sample.
        </Caveat>
      </Section>
    </>
  );
}
