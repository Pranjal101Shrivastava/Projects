import { useState } from "react";
import { useArtifacts } from "../../lib/data";
import { BarChart, LineChart, fmt, seriesColor } from "../../components/Charts";
import { Callout, Caveat, MetricTable, Resolved, Section, Stat } from "../../components/UI";

type BacktestRow = {
  model: string; label: string; mase_mean: number; mase_std: number;
  mase_worst_origin: number; mae_mean: number; beats_seasonal_naive: boolean | null;
  n_origins: number;
};

type SeriesPayload = {
  spec: { title: string; period: number; unit: string; structure: string };
  n_observations: number; start: string; end: string;
  observations: { t: string; y: number }[];
  decomposition: {
    trend_strength: number; seasonal_strength: number; residual_variance_share: number;
    adf_pvalue: number; kpss_pvalue: number; stationarity_verdict: string;
    acf: { lag: number; value: number }[];
    pacf: { lag: number; value: number }[];
    components: { trend: number[]; seasonal: number[]; resid: number[] };
  };
  backtest: {
    n_origins: number; horizon: number; results: BacktestRow[];
    example_forecast: Record<string, number[]>; protocol: string;
  };
};

type Synthesis = {
  cross_series_ranking: { model: string; label: string; mean_rank: number; wins: number; best_rank: number; worst_rank: number }[];
  winners_by_series: Record<string, string>;
  n_distinct_winners: number;
  horizon: number;
  protocol: string;
};

export default function Forecasting() {
  const state = useArtifacts<{ series: Record<string, SeriesPayload>; synthesis: Synthesis }>(
    "05_timeseries_forecasting", ["series", "synthesis"]
  );
  return (
    <Resolved state={state} what="forecasting artifacts">
      {(d) => <Body series={d.series} synthesis={d.synthesis} />}
    </Resolved>
  );
}

function Body({ series, synthesis }: { series: Record<string, SeriesPayload>; synthesis: Synthesis }) {
  const keys = Object.keys(series).filter((k) => !k.startsWith("_"));
  const [active, setActive] = useState(keys[0]);
  const s = series[active];

  const naiveWinners = Object.entries(synthesis.winners_by_series).filter(([, v]) =>
    v.toLowerCase().includes("naive")
  );

  return (
    <>
      <div className="grid grid-4" style={{ marginBottom: 8 }}>
        <Stat label="Series" value={keys.length} sub="deliberately different structures" />
        <Stat label="Distinct winners" value={synthesis.n_distinct_winners} tone="accent"
              sub="no single model dominates" />
        <Stat label="Forecast horizon" value={synthesis.horizon} sub="steps ahead, per origin" />
        <Stat label="Series where naive wins" value={naiveWinners.length} tone="warn"
              sub={naiveWinners.map(([k]) => k.replace(/_/g, " ")).join(", ") || "none"} />
      </div>

      <Callout kind="warn" title="The result that justifies using four series">
        On <strong>sunspots</strong>, no learned model beats the naive baseline. The solar
        cycle averages ~11 years but drifts, so it is not calendar-anchored, and every
        seasonal method mis-specifies it. A single-series study on airline passengers would
        have concluded "SARIMA wins" and been wrong about the general case. Cross-series mean
        rank is a weak recommendation by construction; the per-series table below is the real
        result.
      </Callout>

      <Section title="Cross-series ranking" note="mean rank across all four series">
        <MetricTable
          rows={synthesis.cross_series_ranking}
          highlight={(r) => r.mean_rank === synthesis.cross_series_ranking[0].mean_rank}
          columns={[
            { key: "label", label: "Model" },
            { key: "mean_rank", label: "Mean rank", num: true, render: (r) => fmt(r.mean_rank, 2) },
            { key: "wins", label: "Outright wins", num: true },
            { key: "best_rank", label: "Best", num: true },
            { key: "worst_rank", label: "Worst", num: true },
          ]}
        />
        <p className="small dim" style={{ marginTop: 12 }}>
          LightGBM never wins a series but never ranks worse than third, making it the most
          consistent non-classical option — and it loses to a forty-year-old statistical
          method on three of four series. That is worth stating plainly in a portfolio that
          could easily have shown only the case where the machine learning model won.
        </p>
      </Section>

      <Section title="Per-series results" note="select a series to inspect">
        <div className="row" style={{ marginBottom: 16 }}>
          {keys.map((k) => (
            <button key={k} className={`chip${active === k ? " active" : ""}`}
                    onClick={() => setActive(k)}>
              {series[k].spec.title}
            </button>
          ))}
        </div>

        <div className="card" style={{ marginBottom: 16 }}>
          <div className="row">
            <div>
              <h3 style={{ margin: 0 }}>{s.spec.title}</h3>
              <p className="small dim" style={{ margin: "4px 0 0" }}>{s.spec.structure}</p>
            </div>
            <span className="spacer" />
            <span className="tiny faint mono">
              {s.n_observations} obs · {s.start} → {s.end} · period {s.spec.period}
            </span>
          </div>
        </div>

        <div className="grid grid-4" style={{ marginBottom: 16 }}>
          <Stat label="Trend strength" value={fmt(s.decomposition.trend_strength, 3)} />
          <Stat label="Seasonal strength" value={fmt(s.decomposition.seasonal_strength, 3)} />
          <Stat label="ADF p-value" value={fmt(s.decomposition.adf_pvalue, 4)} />
          <Stat label="KPSS p-value" value={fmt(s.decomposition.kpss_pvalue, 4)} />
        </div>

        <Callout
          kind={s.decomposition.stationarity_verdict.startsWith("conflicting") ? "warn" : "info"}
          title="Stationarity verdict"
        >
          {s.decomposition.stationarity_verdict}
          <p className="tiny dim" style={{ margin: "8px 0 0" }}>
            ADF's null hypothesis is a unit root; KPSS's null is stationarity. They test
            opposite things, so running both distinguishes "the null holds" from "the test is
            underpowered" — and tells you whether to detrend or to difference.
          </p>
        </Callout>

        <div className="grid grid-2">
          <div className="card">
            <h3>Observations</h3>
            <LineChart
              height={240}
              yLabel={s.spec.unit}
              showLegend={false}
              series={[{
                name: s.spec.title,
                points: s.observations.map((o, i) => ({ x: i, y: o.y })),
              }]}
            />
          </div>
          <div className="card">
            <h3>STL components</h3>
            <LineChart
              height={240}
              series={[
                { name: "trend", points: s.decomposition.components.trend.map((v, i) => ({ x: i, y: v })) },
                { name: "seasonal", points: s.decomposition.components.seasonal.map((v, i) => ({ x: i, y: v })), color: seriesColor(1) },
                { name: "residual", points: s.decomposition.components.resid.map((v, i) => ({ x: i, y: v })), color: seriesColor(7) },
              ]}
            />
          </div>
          <div className="card">
            <h3>ACF</h3>
            <BarChart
              height={210}
              yLabel="autocorrelation"
              data={s.decomposition.acf.slice(1, 45).map((a) => ({
                label: String(a.lag),
                value: a.value,
                color: Math.abs(a.value) > 0.2 ? "var(--c1)" : "var(--c8)",
              }))}
              valueFormat={(v) => fmt(v, 3)}
            />
          </div>
          <div className="card">
            <h3>PACF</h3>
            <BarChart
              height={210}
              yLabel="partial autocorrelation"
              data={s.decomposition.pacf.slice(1, 45).map((a) => ({
                label: String(a.lag),
                value: a.value,
                color: Math.abs(a.value) > 0.2 ? "var(--c2)" : "var(--c8)",
              }))}
              valueFormat={(v) => fmt(v, 3)}
            />
          </div>
        </div>

        <div className="card" style={{ marginTop: 16 }}>
          <h3>Backtest — {s.backtest.n_origins} rolling origins, horizon {s.backtest.horizon}</h3>
          <MetricTable
            rows={s.backtest.results}
            highlight={(r) => r.model === s.backtest.results[0].model}
            columns={[
              { key: "label", label: "Model" },
              {
                key: "mase_mean", label: "MASE", num: true,
                render: (r) => (
                  <strong className={r.mase_mean < 1 ? "good-text" : undefined}>
                    {fmt(r.mase_mean, 3)}
                  </strong>
                ),
              },
              { key: "mase_std", label: "± std", num: true, render: (r) => fmt(r.mase_std, 3) },
              { key: "mase_worst_origin", label: "Worst origin", num: true, render: (r) => fmt(r.mase_worst_origin, 3) },
              {
                key: "beats", label: "Beats naive", num: true,
                render: (r) =>
                  r.beats_seasonal_naive === null ? "—" :
                  r.beats_seasonal_naive ? <span className="good-text">yes</span> :
                  <span className="faint">no</span>,
              },
            ]}
          />
          <p className="tiny dim" style={{ marginTop: 10 }}>{s.backtest.protocol}</p>
        </div>

        {s.backtest.example_forecast && (
          <div className="card" style={{ marginTop: 16 }}>
            <h3>Forecasts at the final origin</h3>
            <LineChart
              height={260}
              xLabel="step ahead"
              yLabel={s.spec.unit}
              series={Object.entries(s.backtest.example_forecast).map(([name, values], i) => ({
                name: name === "actual" ? "ACTUAL" : name,
                points: (values as number[]).map((y, x) => ({ x: x + 1, y })),
                color: name === "actual" ? "var(--text)" : seriesColor(i),
                width: name === "actual" ? 3 : 1.6,
                dashed: name !== "actual",
              }))}
            />
          </div>
        )}
      </Section>

      <Section title="Why MASE rather than MAPE or RMSE">
        <Callout kind="info" title="Comparing errors across incomparable units">
          These series measure passengers, degrees Celsius, prescription volumes and sunspot
          area. Averaging raw MAE across them would be meaningless. MASE divides by the
          in-sample seasonal-naive error, which makes it unit-free and anchors it:{" "}
          <strong>1.0 always means "matched seasonal naive"</strong>, below 1.0 means beaten
          it. MAPE was rejected because it is undefined at zero, explodes near it, and
          penalises over- and under-forecasts asymmetrically.
        </Callout>
      </Section>

      <Caveat>
        Hyperparameters are fixed rather than tuned per series, and the SARIMA order is fixed
        at (1,1,1)(1,1,1,m) rather than selected by AIC — both of which make the comparison
        mildly conservative towards the models that would benefit most from tuning. Series
        lengths range from 120 to 204 observations, so the number of usable rolling origins
        differs and short-series estimates are noisier.
      </Caveat>
    </>
  );
}
