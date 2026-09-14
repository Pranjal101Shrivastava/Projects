import { useMemo, useState } from "react";
import { useArtifacts } from "../../lib/data";
import {
  BarChart, Heatmap, Histogram, LineChart, ScatterChart, fmt, seriesColor,
} from "../../components/Charts";
import { Callout, Caveat, MetricTable, Resolved, Section, Stat } from "../../components/UI";

const DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

type Models = {
  models: Record<string, {
    label: string; family: string; mae: number; rmse: number; r2: number;
    skill_score_vs_baseline: number | null; baseline_mae?: number;
  }>;
  best_model: string;
  conformal: Record<string, {
    nominal_coverage: number; empirical_coverage: number; coverage_gap: number;
    mean_interval_width: number; calibrated: boolean;
  }>;
  feature_importance: { feature: string; gain: number; share: number }[];
  error_by_hour: { h: number; mae: number; mean_demand: number }[];
  error_by_zone: { zone: number; mae: number; mean_demand: number }[];
  split: Record<string, unknown>;
};

type Eda = {
  mean_demand_by_hour: { hour: number; mean_demand: number }[];
  heatmap_dow_hour: { dow: number; hour: number; mean_demand: number }[];
  daily_total: { date: string; trips: number }[];
  pickup_sample: { lat: number; lon: number; zone: number }[];
  demand_distribution: { bin_centers: number[]; counts: number[]; mean: number };
  acf_citywide: { lag: number; acf: number }[];
};

type Scorer = {
  table: Record<string, { base: number; profile: number[] }>;
  fidelity: { r2_vs_parent_model: number; mae_vs_actual: number; parent_mae_vs_actual: number; note: string };
  zones: { zone: number; label: string; trips: number; centre_lat: number; centre_lon: number; share: number }[];
};

export default function NycMobility() {
  const state = useArtifacts<{ models: Models; eda: Eda; scorer: Scorer }>(
    "01_nyc_mobility", ["models", "eda", "scorer"]
  );

  return (
    <Resolved state={state} what="NYC demand artifacts">
      {(d) => <Body models={d.models} eda={d.eda} scorer={d.scorer} />}
    </Resolved>
  );
}

function Body({ models, eda, scorer }: { models: Models; eda: Eda; scorer: Scorer }) {
  const best = models.models[models.best_model];
  const naive = models.models["seasonal_naive_168h"];
  const ladder = Object.entries(models.models).map(([key, v]) => ({ key, ...v }));
  ladder.sort((a, b) => a.mae - b.mae);

  const failedLevels = Object.entries(models.conformal).filter(([, v]) => !v.calibrated);

  return (
    <>
      <div className="grid grid-4" style={{ marginBottom: 8 }}>
        <Stat label="Best model MAE" value={fmt(best.mae, 2)} tone="accent"
              sub={`${best.label}`} />
        <Stat label="Seasonal-naive MAE" value={fmt(naive.mae, 2)}
              sub="repeat same hour last week" />
        <Stat label="Skill vs naive" value={`${(best.skill_score_vs_baseline! * 100).toFixed(1)}%`}
              tone="good" sub="reduction in mean absolute error" />
        <Stat label="Zones × hours modelled"
              value={(models.split.n_train as number).toLocaleString()}
              sub={`+ ${(models.split.n_test as number).toLocaleString()} held out`} />
      </div>

      <Callout kind="info" title="Why demand, and not trip duration">
        The TLC's FOIL release contains four columns: timestamp, latitude, longitude and
        dispatch base. There is <strong>no trip duration and no fare</strong>. A duration
        model on this data would have to invent its own target, and a model fitted to a
        synthetic target measures only how well it recovers the generator. Demand per zone
        per hour is directly countable from the records and is what dispatch decisions are
        actually made on.
      </Callout>

      <Section title="The model ladder"
               note="every rung scored on the identical embargoed holdout">
        <MetricTable
          rows={ladder}
          highlight={(r) => r.key === models.best_model}
          columns={[
            { key: "label", label: "Model" },
            { key: "family", label: "Family" },
            { key: "mae", label: "MAE", num: true, render: (r) => fmt(r.mae, 2) },
            { key: "rmse", label: "RMSE", num: true, render: (r) => fmt(r.rmse, 2) },
            { key: "r2", label: "R²", num: true, render: (r) => fmt(r.r2, 3) },
            {
              key: "skill", label: "Skill vs naive", num: true,
              render: (r) =>
                r.skill_score_vs_baseline === null ? "—" : (
                  <span className={r.skill_score_vs_baseline > 0 ? "good-text" : "bad-text"}>
                    {(r.skill_score_vs_baseline * 100).toFixed(1)}%
                  </span>
                ),
            },
          ]}
        />
        <p className="small dim" style={{ marginTop: 12 }}>
          Note that <strong>yesterday is a worse predictor than last week</strong>: the
          naive t−24h baseline scores {fmt(models.models["naive_24h"].mae, 2)} MAE against{" "}
          {fmt(naive.mae, 2)} for t−168h. Weekly periodicity dominates urban mobility, which
          is why <code>lag_168h</code> carries{" "}
          {((models.feature_importance.find((f) => f.feature === "lag_168h")?.share ?? 0) * 100).toFixed(1)}%
          of model gain below.
        </p>
      </Section>

      <Section title="Prediction intervals" note="split-conformal, with realised coverage">
        <div className="grid grid-3">
          {Object.entries(models.conformal).map(([level, v]) => (
            <div className="card" key={level}>
              <div className="row">
                <strong>{level} interval</strong>
                <span className="spacer" />
                <span className={`badge ${v.calibrated ? "badge-real" : "badge-bad"}`}>
                  {v.calibrated ? "calibrated" : "under-covers"}
                </span>
              </div>
              <div className="stat-value" style={{ fontSize: "1.3rem", marginTop: 8 }}>
                {(v.empirical_coverage * 100).toFixed(1)}%
              </div>
              <div className="stat-sub">
                realised vs {(v.nominal_coverage * 100).toFixed(0)}% nominal
                {" · "}width ±{fmt(v.mean_interval_width / 2, 1)}
              </div>
            </div>
          ))}
        </div>
        {failedLevels.length > 0 && (
          <Caveat>
            The {failedLevels.map(([k]) => k).join(" and ")} interval{" "}
            {failedLevels.length === 1 ? "under-covers" : "under-cover"}, realising only{" "}
            {failedLevels.map(([, v]) => `${(v.empirical_coverage * 100).toFixed(1)}%`).join(", ")}.
            Split conformal assumes calibration and test residuals are exchangeable. They are
            not here: calibration comes from the chronologically last slice of training data,
            and demand grew <strong>+82.9%</strong> across the six-month window, so
            test-period residuals are systematically larger. The wider 90% and 95% bands
            absorb that drift; the tight band cannot. This is a real limitation of the method
            under trend, and the fix is periodic recalibration on recent data.
          </Caveat>
        )}
      </Section>

      <Section title="What the model learned" note="LightGBM gain, normalised">
        <div className="grid grid-2">
          <div className="card">
            <h3>Feature importance</h3>
            <BarChart
              horizontal
              height={280}
              data={models.feature_importance.slice(0, 10).map((f) => ({
                label: f.feature, value: f.share,
              }))}
              valueFormat={(v) => `${(v * 100).toFixed(1)}%`}
            />
          </div>
          <div className="card">
            <h3>Citywide autocorrelation</h3>
            <LineChart
              height={280}
              xLabel="lag (hours)"
              yLabel="ACF"
              showLegend={false}
              series={[{ name: "ACF", points: eda.acf_citywide.map((a) => ({ x: a.lag, y: a.acf })) }]}
              markers={[
                { x: 24, label: "24h", color: "var(--c3)" },
                { x: 168, label: "168h", color: "var(--c2)" },
              ]}
            />
            <p className="tiny dim" style={{ marginTop: 8 }}>
              The spikes at lag 24 and 168 are the daily and weekly cycles. They are the
              empirical justification for the lag features chosen, rather than a post-hoc
              rationalisation of them.
            </p>
          </div>
        </div>
      </Section>

      <DemandExplorer scorer={scorer} />

      <Section title="Exploratory views" note="from 4.5M real dispatch records">
        <div className="grid grid-2">
          <div className="card">
            <h3>Demand by hour of week</h3>
            <Heatmap
              rows={7}
              cols={24}
              rowLabels={DOW}
              colLabels={Array.from({ length: 24 }, (_, i) => String(i))}
              valueLabel="mean demand"
              cells={eda.heatmap_dow_hour.map((c) => ({
                row: c.dow, col: c.hour, value: c.mean_demand,
              }))}
            />
          </div>
          <div className="card">
            <h3>Pickup locations by learned zone</h3>
            <ScatterChart
              height={300}
              xLabel="longitude"
              yLabel="latitude"
              radius={1.6}
              opacity={0.5}
              points={eda.pickup_sample.map((p) => ({ x: p.lon, y: p.lat, c: p.zone }))}
            />
            <p className="tiny dim" style={{ marginTop: 6 }}>
              6,000-point sample. Zones are K-means centroids over pickup coordinates —
              learned from where demand actually is, rather than imposed as a uniform grid
              that would be mostly water and parkland.
            </p>
          </div>
          <div className="card">
            <h3>Daily volume — the growth that broke calibration</h3>
            <LineChart
              height={240}
              xLabel="day"
              yLabel="trips"
              showLegend={false}
              areaUnder
              series={[{
                name: "trips",
                points: eda.daily_total.map((d, i) => ({ x: i, y: d.trips })),
              }]}
            />
            <p className="tiny dim" style={{ marginTop: 6 }}>
              Apr–Sep 2014. Volume rose 82.9% from the first four weeks to the last, which is
              why the tightest conformal band under-covers.
            </p>
          </div>
          <div className="card">
            <h3>Zone-hour demand distribution</h3>
            <Histogram data={eda.demand_distribution} xLabel="pickups per zone-hour" />
          </div>
        </div>
      </Section>

      <Section title="Where the model is weakest">
        <div className="grid grid-2">
          <div className="card">
            <h3>Error by hour of day</h3>
            <LineChart
              height={230}
              xLabel="hour"
              yLabel="MAE"
              series={[
                { name: "MAE", points: models.error_by_hour.map((e) => ({ x: e.h, y: e.mae })) },
                {
                  name: "mean demand", color: "var(--c8)", dashed: true,
                  points: models.error_by_hour.map((e) => ({ x: e.h, y: e.mean_demand / 10 })),
                },
              ]}
            />
            <p className="tiny dim" style={{ marginTop: 6 }}>
              Mean demand shown ÷10 for scale. Error tracks demand: the evening peak is both
              the busiest and the least predictable period.
            </p>
          </div>
          <div className="card">
            <h3>Error by zone</h3>
            <BarChart
              height={230}
              yLabel="MAE"
              data={models.error_by_zone
                .slice()
                .sort((a, b) => b.mae - a.mae)
                .map((e) => ({
                  label: scorer.zones.find((z) => z.zone === e.zone)?.label ?? `zone ${e.zone}`,
                  value: e.mae,
                }))}
              valueFormat={(v) => fmt(v, 1)}
            />
            <p className="tiny dim" style={{ marginTop: 6 }}>
              A single citywide MAE understates performance in quiet zones and overstates it
              in busy ones — high-volume zones dominate the aggregate.
            </p>
          </div>
        </div>
      </Section>
    </>
  );
}

/**
 * Interactive demand forecaster.
 *
 * Runs the exported hour-of-week × zone surrogate client-side. The surrogate's agreement
 * with the parent LightGBM model is displayed next to the control, so the demo reads as a
 * documented approximation rather than as the trained model.
 */
function DemandExplorer({ scorer }: { scorer: Scorer }) {
  const zones = useMemo(
    () => scorer.zones.slice().sort((a, b) => b.trips - a.trips),
    [scorer.zones]
  );
  const [zone, setZone] = useState(zones[0].zone);
  const [dow, setDow] = useState(4);
  const [hour, setHour] = useState(18);

  const entry = scorer.table[String(zone)];
  const hourOfWeek = dow * 24 + hour;
  const prediction = entry ? entry.base * entry.profile[hourOfWeek] : 0;

  const dayCurve = useMemo(() => {
    if (!entry) return [];
    return Array.from({ length: 24 }, (_, h) => ({
      x: h,
      y: entry.base * entry.profile[dow * 24 + h],
    }));
  }, [entry, dow]);

  return (
    <Section title="Forecast a zone-hour" note="runs entirely in your browser">
      <div className="card">
        <div className="controls">
          <div className="control" style={{ minWidth: 210 }}>
            <label className="control-label" htmlFor="zone">Zone</label>
            <select id="zone" value={zone} onChange={(e) => setZone(Number(e.target.value))}>
              {zones.map((z) => (
                <option key={z.zone} value={z.zone}>
                  {z.label} ({(z.share * 100).toFixed(1)}% of trips)
                </option>
              ))}
            </select>
          </div>
          <div className="control">
            <label className="control-label" htmlFor="dow">Day</label>
            <select id="dow" value={dow} onChange={(e) => setDow(Number(e.target.value))}>
              {DOW.map((d, i) => <option key={d} value={i}>{d}</option>)}
            </select>
          </div>
          <div className="control" style={{ minWidth: 200 }}>
            <label className="control-label" htmlFor="hour">Hour — {String(hour).padStart(2, "0")}:00</label>
            <input id="hour" type="range" min={0} max={23} value={hour}
                   onChange={(e) => setHour(Number(e.target.value))} />
          </div>
          <div className="control">
            <div className="control-label">Expected pickups</div>
            <div className="stat-value accent" style={{ fontSize: "1.9rem" }}>
              {Math.round(prediction)}
            </div>
          </div>
        </div>

        <LineChart
          height={220}
          xLabel="hour of day"
          yLabel="expected pickups"
          showLegend={false}
          areaUnder
          series={[{ name: "expected", points: dayCurve, color: seriesColor(0) }]}
          markers={[{ x: hour, label: `${String(hour).padStart(2, "0")}:00`, color: "var(--warn)" }]}
        />

        <Callout kind="warn" title="This is a surrogate, not the trained model">
          A 600-tree gradient boosted model is impractical to ship to a browser, so this
          control evaluates an exported hour-of-week × zone profile instead. Its agreement
          with the parent model is <strong>R² = {fmt(scorer.fidelity.r2_vs_parent_model, 4)}</strong>{" "}
          on the same holdout — MAE {fmt(scorer.fidelity.mae_vs_actual, 2)} against the
          parent's {fmt(scorer.fidelity.parent_mae_vs_actual, 2)}. It reproduces the periodic
          structure but carries no autoregressive state, so it cannot react to a demand shock
          the way the real model does.
        </Callout>
      </div>
    </Section>
  );
}
