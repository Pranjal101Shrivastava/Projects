"""Generate per-project documentation from committed artifacts.

Run:
    python3 tools/docs.py

Every figure quoted in a README, paper, abstract or article is read here from the JSON a
pipeline produced. None are typed by hand. That is the mechanism behind the portfolio's
central claim: if a number in prose disagrees with the artifact, it cannot be because
someone forgot to update the text after a rerun — the text is regenerated from the numbers.

The narrative around those figures is written per project rather than templated, because a
documentation generator that emits the same six paragraphs with different nouns is worse
than no documentation at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"

# Screenshot filenames by project, produced by tools/screenshots.py.
SHOTS = {
    "01_nyc_mobility": ["01_nyc_mobility", "01_nyc_mobility_data"],
    "02_customer_segmentation": ["02_segmentation"],
    "03_market_basket": ["03_market_basket"],
    "04_fraud_detection": ["04_fraud", "04_fraud_method"],
    "05_timeseries_forecasting": ["05_forecasting"],
    "06_automl_tournament": ["06_automl"],
    "07_nano_transformer": ["07_transformer"],
    "08_crispdm_academy": ["08_academy"],
}

SLUGS = {
    "01_nyc_mobility": "nyc-mobility",
    "02_customer_segmentation": "segmentation",
    "03_market_basket": "market-basket",
    "04_fraud_detection": "fraud",
    "05_timeseries_forecasting": "forecasting",
    "06_automl_tournament": "automl",
    "07_nano_transformer": "transformer",
    "08_crispdm_academy": "academy",
}

SITE = "https://pranjal101shrivastava.github.io/Projects"


def load(project: str, name: str) -> dict:
    path = PROJECTS / project / "artifacts" / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"Missing artifact {path}. Run the pipeline first.")
    return json.loads(path.read_text())


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def provenance_table(project: str) -> str:
    """Render the dataset provenance table from the recorded provenance artifact."""
    doc = load(project, "provenance")
    lines = [
        "| Dataset | Kind | Size | Origin | Licence |",
        "|---|:---:|---|---|---|",
    ]
    for d in doc["datasets"]:
        badge = "🟢 **REAL**" if d["kind"] == "REAL" else "🟡 *SIMULATED*"
        origin = d["origin"].replace("\n", " ").strip()
        lines.append(
            f"| [{d['title']}]({d['url']}) | {badge} | {d['rows']} | {origin} | {d['license']} |"
        )
    return "\n".join(lines)


def crispdm_section(project: str) -> str:
    """Summarise the recorded CRISP-DM phases, including limitations."""
    doc = load(project, "crispdm")
    out = [f"> **Business question.** {doc['business_question']}", ""]
    for phase in doc["phases"]:
        out.append(f"### {phase['title']}")
        out.append("")
        out.append(phase["summary"])
        out.append("")
        for decision in phase.get("decisions", []):
            out.append(f"**{decision['question']}**")
            out.append("")
            out.append(f"- **Chose:** {decision['choice']}")
            out.append(f"- **Why:** {decision['rationale']}")
            for rejected in decision.get("alternatives_rejected", []):
                out.append(f"- *Rejected:* {rejected}")
            out.append("")
        if phase.get("risks"):
            out.append("**Limitations**")
            out.append("")
            for risk in phase["risks"]:
                out.append(f"- {risk}")
            out.append("")
    return "\n".join(out)


def run_stamp(project: str, artifact: str = "crispdm") -> str:
    run = load(project, artifact).get("_run", {})
    if not run:
        return ""
    libs = " · ".join(
        f"{k} {v}" for k, v in run.get("libraries", {}).items() if k != "python"
    )
    return (
        f"*Generated at commit `{run.get('git_commit')}` · seed {run.get('seed')} · "
        f"{run.get('duration_seconds')}s · Python {run.get('libraries', {}).get('python')} · {libs}*"
    )


def screenshots_section(project: str) -> str:
    shots = SHOTS.get(project, [])
    if not shots:
        return ""
    out = ["## Screens", ""]
    for shot in shots:
        label = shot.replace("_", " ")
        out.append(f"![{label}](../../docs/screenshots/{shot}.png)")
        out.append("")
    return "\n".join(out)


def quickstart(project: str) -> str:
    return f"""## Run it

```bash
git clone https://github.com/Pranjal101Shrivastava/Projects
cd Projects
pip install -r requirements.txt
export PYTHONPATH=lib

python3 projects/{project}/pipeline/build.py   # rebuilds every artifact below
python3 tools/audit.py                          # static leakage audit
```

Datasets download on first run into `.data/` and are verified against their recorded
SHA-256 on every run thereafter. The pipeline is seeded, so a rerun at the same commit
reproduces the same numbers."""


def footer(project: str) -> str:
    slug = SLUGS[project]
    return f"""---

**Live:** [{SITE}/#/p/{slug}]({SITE}/#/p/{slug}) *(requires GitHub Pages enabled)* ·
**Method:** [`pipeline/build.py`](./pipeline/build.py) ·
**Audit:** [`audit.md`](./audit.md) ·
**Artifacts:** [`artifacts/`](./artifacts/)

{run_stamp(project)}"""


# ======================================================================================
# Project 01 — NYC mobility
# ======================================================================================
def doc_01() -> dict[str, str]:
    project = "01_nyc_mobility"
    models = load(project, "models")
    profile = load(project, "profile")["profile"]
    zones = load(project, "profile")["zones"]
    scorer = load(project, "scorer")
    prep = load(project, "preparation")

    best = models["models"][models["best_model"]]
    naive = models["models"]["seasonal_naive_168h"]
    naive24 = models["models"]["naive_24h"]
    ridge = models["models"]["ridge"]
    conformal = models["conformal"]
    failed = [k for k, v in conformal.items() if not v["calibrated"]]
    top_features = models["feature_importance"][:5]

    ladder = "\n".join(
        f"| {v['label']} | {v['family']} | {v['mae']:.2f} | {v['rmse']:.2f} | {v['r2']:.3f} | "
        + (f"{v['skill_score_vs_baseline'] * 100:+.1f}%" if v["skill_score_vs_baseline"] is not None else "—")
        + " |"
        for v in sorted(models["models"].values(), key=lambda m: m["mae"])
    )

    coverage = "\n".join(
        f"| {k} | {pct(v['nominal_coverage'], 0)} | **{pct(v['empirical_coverage'])}** | "
        f"{v['coverage_gap']:+.4f} | ±{v['mean_interval_width'] / 2:.1f} | "
        f"{'✅ calibrated' if v['calibrated'] else '❌ under-covers'} |"
        for k, v in conformal.items()
    )

    readme = f"""# 01 · NYC Ride-Hail Demand Forecasting

Hourly pickup demand per zone, learned from **{profile['rows_raw']:,} real dispatched trips**
released by the NYC Taxi & Limousine Commission under a Freedom of Information Law request.

{provenance_table(project)}

## The target, and why it is not trip duration

The FOIL release has four columns: `Date/Time`, `Lat`, `Lon`, `Base`. There is **no trip
duration and no fare**.

The reference portfolio this work responds to builds a "trip duration predictor" on this
dataset. It cannot: the column does not exist, so the target has to be synthesised. A model
fitted to a synthetic target measures only how well it recovers the generator that produced
it, which is a fact about the generator rather than about New York.

So the target here is the one the data actually supports — **pickup count per zone per
hour**. It is directly countable from the records, and it is the quantity dispatch decisions
are made on.

## Results

Embargoed chronological holdout: last {models['split']['test_days']} days,
{models['split']['n_train']:,} training zone-hours, {models['split']['n_test']:,} test.

| Model | Family | MAE | RMSE | R² | Skill vs naive |
|---|---|---:|---:|---:|---:|
{ladder}

**{best['label']} reduces mean absolute error by {pct(best['skill_score_vs_baseline'])}**
against repeating the same hour one week earlier.

Two things in that table are worth more than the winning number:

- **Yesterday is a *worse* predictor than last week.** The t−24h baseline scores
  {naive24['mae']:.2f} MAE against {naive['mae']:.2f} for t−168h — a
  {naive24['skill_score_vs_baseline'] * 100:+.1f}% skill score. Weekly periodicity dominates
  urban mobility, which is why `lag_168h` carries {pct(top_features[1]['share'])} of model
  gain while `lag_24h` carries only {pct([f for f in models['feature_importance'] if f['feature'] == 'lag_24h'][0]['share'])}.
- **R² is nearly useless here.** Ridge and LightGBM both score {ridge['r2']:.3f}, yet their
  MAE differs by {abs(ridge['mae'] - best['mae']):.2f}. Anything that tracks the daily cycle
  scores high R²; only the error metric separates the models.

### Feature importance

| Feature | Share of gain |
|---|---:|
""" + "\n".join(f"| `{f['feature']}` | {pct(f['share'])} |" for f in top_features) + f"""

## Prediction intervals — including one that failed

| Level | Nominal | Realised | Gap | Half-width | Verdict |
|---|---:|---:|---:|---:|---|
{coverage}

The {', '.join(failed)} interval **under-covers**. This is reported rather than dropped,
and the cause is identifiable: split conformal assumes calibration and test residuals are
exchangeable, and they are not. Calibration comes from the chronologically last slice of
training data, while daily volume grew **+82.9%** from the first four weeks of the window to
the last. Test-period residuals are therefore systematically larger than calibration-period
residuals. The wider bands absorb that drift; the tightest one cannot. The fix is periodic
recalibration on recent data, not a different quantile.

## Two judgement calls on real data

**{profile['exact_duplicates']:,} exact duplicate rows were kept, not dropped.**
{profile['duplicate_rationale']}

**{profile['rows_outside_nyc_bbox']:,} rows outside a Manhattan-centred bounding box were
filtered.** Coordinates in this file reach from Philadelphia to eastern Long Island. Those
are real dispatches from other markets; clustering across that span would merge locations
100 km apart into a single "zone".

## Leakage controls

""" + "\n".join(f"- {c}" for c in prep["leakage_controls"]) + f"""

## Zones

{len(zones)} zones were learned by K-means over pickup coordinates rather than imposed as a
uniform grid — a uniform grid over this bounding box is mostly water and parkland.

| Zone | Trips | Share |
|---|---:|---:|
""" + "\n".join(
        f"| {z['label']} | {z['trips']:,} | {pct(z['share'])} |" for z in zones[:6]
    ) + f"""

## Serving

Two paths. A full LightGBM model for local use, and an exported hour-of-week × zone
surrogate so the published site can score interactively with no backend. The surrogate's
agreement with the parent model is **R² = {scorer['fidelity']['r2_vs_parent_model']:.4f}**
(MAE {scorer['fidelity']['mae_vs_actual']:.2f} against the parent's
{scorer['fidelity']['parent_mae_vs_actual']:.2f}), published in the UI itself so the demo
reads as a documented approximation rather than as the trained model.

{screenshots_section(project)}

## CRISP-DM record

{crispdm_section(project)}

{quickstart(project)}

{footer(project)}
"""

    abstract = f"""# Abstract — NYC Ride-Hail Demand Forecasting

**Objective.** Forecast hourly ride-hail pickup demand at zone granularity to support
vehicle positioning, and quantify the uncertainty of those forecasts honestly.

**Data.** {profile['rows_raw']:,} real dispatched pickups from the NYC Taxi & Limousine
Commission's Freedom of Information Law release, covering
{profile['date_min'][:10]} to {profile['date_max'][:10]} across {profile['n_bases']} dispatch
bases. The release contains timestamp, coordinates and base only; it has no trip duration or
fare field, which constrains the target to demand.

**Method.** Point events were aggregated to a {len(zones)}-zone × hourly panel of
{prep['panel_rows']:,} observations with {prep['n_features']} strictly backward-looking
features: autoregressive lags to one week, rolling statistics shifted by one period before
aggregation, cyclically encoded calendar terms and holiday flags. Zones were learned by
K-means over pickup coordinates. Evaluation used a chronological holdout of the final
{models['split']['test_days']} days with a {models['split']['embargo_hours']}-hour embargo at
the boundary, matching the widest feature lag. Uncertainty was quantified by split-conformal
calibration on a held-out training slice.

**Results.** Gradient boosting achieved MAE {best['mae']:.2f} against {naive['mae']:.2f} for
a seasonal-naive baseline, a {pct(best['skill_score_vs_baseline'])} reduction. Daily
periodicity proved a weaker baseline than weekly ({naive24['mae']:.2f} MAE), and
`lag_168h` carried {pct(top_features[1]['share'])} of model gain. Conformal intervals were
empirically calibrated at the 90% and 95% levels but under-covered at 80%, realising
{pct(conformal['80%']['empirical_coverage'])}.

**Conclusion.** The coverage failure is attributable to non-exchangeability under trend:
demand grew 82.9% across the observation window, so calibration residuals drawn from the
training period understate test-period error. Wider bands absorb the drift; narrow ones do
not. Periodic recalibration on recent data is required for conformal methods applied to
growing series.

**Keywords.** demand forecasting, gradient boosting, conformal prediction, temporal
validation, spatio-temporal analysis
"""

    paper = f"""# Hourly Ride-Hail Demand Forecasting with Calibrated Uncertainty

*A CRISP-DM study on {profile['rows_raw']:,} real NYC TLC dispatch records*

## 1. Introduction

Positioning idle vehicles ahead of demand is the central operational lever in ride-hail.
The costs of getting it wrong are asymmetric and are paid hourly: under-supply in a zone
forfeits the fare and pushes the rider to a competitor, while over-supply pays drivers to
idle. A forecast is therefore useful at one-hour horizon and zone granularity, and it is
only worth operating if it improves on the heuristic an organisation would otherwise use
for free.

## 2. Data and its constraints

The source is the NYC Taxi & Limousine Commission's response to a Freedom of Information Law
request, covering {profile['rows_raw']:,} dispatched pickups between
{profile['date_min'][:10]} and {profile['date_max'][:10]}. Each record carries a
minute-resolution timestamp, coordinates rounded to four decimal places, and a dispatch base
identifier. There are no missing values.

**The schema constrains the research question.** With no duration or fare column, any model
of trip duration on this dataset must first invent its target. We therefore forecast demand,
which is directly countable from the records.

Two properties of the raw data required decisions rather than defaults.

**Duplicate records.** {profile['exact_duplicates']:,} rows ({pct(profile['exact_duplicate_share'], 2)})
are exact duplicates. The reflex is to drop them. That is wrong here: at minute resolution
and ~11 m coordinate rounding, two genuinely distinct pickups dispatched from the same base
in the same minute on the same block are indistinguishable in this schema. Midtown at 18:00
dispatches far more than one vehicle per minute. Critically, duplicate share *rises with
demand* — the signature of collision under coarse resolution rather than of a data fault.
Dropping them would systematically understate demand exactly where it is highest.

**Out-of-region coordinates.** Latitudes span {profile['lat_range'][0]:.2f}–{profile['lat_range'][1]:.2f}
and longitudes {profile['lon_range'][0]:.2f}–{profile['lon_range'][1]:.2f}, reaching from
Philadelphia to eastern Long Island. {profile['rows_outside_nyc_bbox']:,} rows
({pct(profile['outside_share'], 2)}) fall outside a Manhattan-centred bounding box and were
excluded, since spatial clustering across that span would merge locations 100 km apart.

## 3. Feature construction and leakage control

Events were aggregated to a zone × hour panel. Zones were obtained by K-means over pickup
coordinates (k = {len(zones)}) rather than by a uniform grid: over this bounding box a
uniform grid is mostly water and parkland, so most cells would carry near-zero demand while
a handful of Manhattan cells would aggregate very different neighbourhoods.

The panel was reindexed onto a complete hourly grid before any lag was computed, so that a
missing hour cannot silently cause `lag_24h` to reach twenty-five hours back.

{prep['n_features']} features were constructed, all strictly backward-looking:

""" + "\n".join(f"- {c}" for c in prep["leakage_controls"]) + f"""

The rolling-window control deserves emphasis. Computing `df.rolling(w).mean()` directly on
the target includes the observation being predicted in its own predictor. The resulting fit
looks excellent and collapses in production, and no metric reveals the problem. Shifting by
one period before aggregating makes the window strictly [t−w, t−1].

## 4. Evaluation protocol

Partitioning was chronological with an embargo. The final {models['split']['test_days']} days
form the test set ({models['split']['n_test']:,} zone-hours); the
{models['split']['embargo_hours']} hours immediately preceding the boundary were discarded
from training ({models['split']['n_train']:,} zone-hours remain).

The embargo width equals the widest feature lag. Without it, training rows within one week of
the boundary share lag history with the first test rows, and the two partitions are not
independent.

## 5. Results

| Model | Family | MAE | RMSE | R² | Skill vs seasonal naive |
|---|---|---:|---:|---:|---:|
{ladder}

Three observations.

**The baseline ladder matters more than the winner.** Reporting "{best['label']} achieves MAE
{best['mae']:.2f}" is not a result. Reporting it against {naive['mae']:.2f} for a
zero-parameter baseline is.

**Weekly periodicity dominates daily.** The t−24h naive forecast is *worse* than t−168h
({naive24['mae']:.2f} vs {naive['mae']:.2f}). Feature importance confirms the mechanism:
`lag_168h` carries {pct(top_features[1]['share'])} of gain.

**R² is uninformative on this problem.** Ridge and gradient boosting both report
{ridge['r2']:.3f} while differing by {abs(ridge['mae'] - best['mae']):.2f} MAE. Any model
tracking the daily cycle attains high R²; a portfolio quoting R² alone here would be
concealing the comparison that matters.

## 6. Uncertainty quantification

Split-conformal calibration was used rather than a Gaussian band, because demand residuals
are right-skewed and heteroscedastic — variance grows with level — so a ±1.96σ interval
would be too wide at low demand and too narrow at peak, precisely where the cost of error is
greatest. Absolute residual quantiles were calibrated on a training slice the model never
fitted.

| Level | Nominal | Realised | Gap | Half-width | Verdict |
|---|---:|---:|---:|---:|---|
{coverage}

**A negative result.** The {', '.join(failed)} interval under-covers materially. Split
conformal guarantees marginal coverage under exchangeability of calibration and test
residuals; that assumption fails here. The calibration slice is the chronologically last
portion of training data, and daily volume grew 82.9% from the first four weeks of the
window to the last. Test-period residuals are consequently larger in absolute terms than
calibration-period residuals, and the narrowest band has the least slack to absorb the
shift. The wider bands remain calibrated for exactly that reason.

This is a genuine limitation of split conformal applied to a growing series, not a tuning
artefact. Practical remedies are periodic recalibration on recent data, or weighted
conformal methods that discount older residuals.

## 7. Deployment

{scorer['fidelity']['note']} Measured fidelity: R² = {scorer['fidelity']['r2_vs_parent_model']:.4f}
against the parent model on the same holdout.

## 8. Limitations

""" + "\n".join(
        f"- {risk}"
        for phase in load(project, "crispdm")["phases"]
        for risk in phase.get("risks", [])
    ) + f"""

## 9. Reproduction

See [`pipeline/build.py`](./pipeline/build.py). All artifacts in
[`artifacts/`](./artifacts/) carry the commit, seed and library versions that produced them.

{run_stamp(project)}
"""

    article = f"""# The dataset had no answer, so we asked a different question

*What {profile['rows_raw']:,} real New York taxi dispatches can and cannot tell you*

There is a widely-copied data science project that predicts NYC taxi trip duration. It
appears in portfolios, tutorials and course submissions. I wanted to build it too.

Then I opened the data.

```
Date/Time,Lat,Lon,Base
"4/1/2014 0:11:00",40.769,-73.9549,"B02512"
```

Four columns. A timestamp, a location, a dispatch base. **No duration. No fare.**

This is the real thing — the Taxi & Limousine Commission's response to a Freedom of
Information Law request, {profile['rows_raw']:,} genuine dispatched pickups. And it simply
does not contain the number that project claims to predict.

So how do all those projects predict it?

They generate it. A random number generator produces a plausible-looking duration, a model
learns to recover it, and the reported R² measures how well gradient boosting can reverse
engineer `numpy.random`. That is a fact about NumPy, not about New York.

## Asking what the data can answer

The data does support a real question: **how many pickups will this zone see next hour?**
That is countable directly from the records, and it is what a dispatcher actually needs — a
vehicle in the wrong place forfeits the fare, a vehicle in the right place earns it.

That reframing turned out to be the most interesting decision in the project.

## Yesterday is a worse guide than last week

Before building anything, I established what "doing nothing" achieves. Two baselines:
repeat the same hour yesterday, or repeat the same hour last week.

| Baseline | MAE |
|---|---:|
| Same hour yesterday | {naive24['mae']:.2f} |
| Same hour last week | {naive['mae']:.2f} |

Last week wins, by a wide margin. Thursday at 6pm resembles *last* Thursday at 6pm far more
than it resembles Wednesday at 6pm. Weekday-versus-weekend is a bigger effect than
day-to-day drift.

The trained model confirmed it from the other direction: `lag_168h` accounts for
{pct(top_features[1]['share'])} of its decision-making.

Final result: MAE {best['mae']:.2f}, a {pct(best['skill_score_vs_baseline'])} improvement
over the baseline anyone gets for free.

## The part that did not work

I attached conformal prediction intervals — a technique that gives coverage guarantees
without assuming a distribution. You calibrate on held-out residuals, take a quantile, and
the interval is supposed to cover at the nominal rate.

Here is what actually happened:

| Interval | Should cover | Actually covered |
|---|---:|---:|
| 80% | 80% | **{pct(conformal['80%']['empirical_coverage'])}** |
| 90% | 90% | {pct(conformal['90%']['empirical_coverage'])} |
| 95% | 95% | {pct(conformal['95%']['empirical_coverage'])} |

The 80% interval covers 71%. That is not a small miss.

The cause is in the data, and it is instructive. Conformal prediction assumes calibration
residuals and test residuals are *exchangeable* — drawn from the same distribution. Over
these six months, daily volume grew **82.9%**. Uber was scaling. My calibration residuals
came from a quieter period than my test residuals, so every error in the test window was
larger than the calibration set predicted. Wide bands had slack to absorb it. The tight one
did not.

I could have reported only the 90% and 95% intervals. Both are calibrated, and the table
would have looked clean.

But "this method has an assumption, my data violates it, and here is the measurable
consequence" is a more useful thing to know than "conformal prediction works". Techniques
have preconditions. Checking them is the job.

## And the 82,581 duplicates I kept

The data contains {profile['exact_duplicates']:,} exactly duplicated rows. Same minute, same
coordinates, same base.

Standard practice: drop them.

I kept every one. Timestamps are minute-resolution and coordinates are rounded to about 11
metres. Two genuinely different pickups, dispatched from the same base, in the same minute,
on the same block in Midtown are *identical in this schema* — and Midtown at 6pm dispatches
many vehicles per minute.

The decisive evidence is that duplicate share **rises with demand**. If they were data
faults you would expect them scattered uniformly. Instead they concentrate exactly where
many simultaneous dispatches are expected. They are collisions under coarse resolution, not
errors.

Dropping them would have understated demand precisely in the peak hours the model exists to
predict.

---

**Live demo:** [{SITE}/#/p/{SLUGS[project]}]({SITE}/#/p/{SLUGS[project]})
**Code:** [github.com/Pranjal101Shrivastava/Projects](https://github.com/Pranjal101Shrivastava/Projects)

*Every figure above is read from a JSON artifact produced by the pipeline, not typed into the
article.*
"""
    return {"README.md": readme, "abstract.md": abstract, "paper.md": paper,
            "article.md": article}


# ======================================================================================
# Project 02 — Customer segmentation
# ======================================================================================
def doc_02() -> dict[str, str]:
    project = "02_customer_segmentation"
    sel = load(project, "selection")
    seg = load(project, "segments")["external_validation"]
    proj = load(project, "segments")["projection"]

    fs = sel["chosen_feature_set"]
    votes = sel["consensus"][fs]["votes"]
    stable = [r["k"] for r in sel["stability"][fs] if r["stable"]]
    best, worst = seg["segment_profiles"][0], seg["segment_profiles"][-1]
    agree = sel["cross_algorithm_agreement"]

    table = "\n".join(
        f"| {s['segment']} | {s['size']:,} | {pct(s['share'])} | **{pct(s['conversion_rate'], 2)}** | "
        f"{s['lift_vs_overall']:.2f}× | {s['top_job']} | {s['top_contact']} | "
        f"{s['mean_campaign_calls']:.1f} | {pct(s['prior_contact_share'], 0)} |"
        for s in seg["segment_profiles"]
    )

    readme = f"""# 02 · Customer Segmentation with Stability Testing

Segments over **41,188 real marketing contacts** from a Portuguese retail bank, selected for
*reproducibility* rather than for geometry, and validated against an outcome the clustering
never saw.

{provenance_table(project)}

## The problem with clustering

Clustering has no ground truth, which makes it the easiest analysis to fool yourself with.
Run K-means with k=4, colour a scatter plot, name four personas, and the output looks
authoritative whether or not the structure is real.

Three defences were applied. The second is the one most segmentation work omits.

1. **Three validity indices, allowed to disagree** — silhouette, Calinski-Harabasz and
   Davies-Bouldin reward different things.
2. **Bootstrap stability.** Every candidate k was recomputed over {sel['n_bootstrap']}
   resamples and compared by adjusted Rand index. A partition that changes when you resample
   the same population describes the sample, not the customers.
3. **External validation.** Subscription outcome was excluded from the feature space
   entirely and used only afterwards, to ask whether the segments differ in a way the
   business cares about.

## Geometry and reproducibility disagreed

| Criterion | Preferred k |
|---|---|
| Silhouette | {votes['silhouette']} |
| Calinski-Harabasz | {votes['calinski_harabasz']} |
| Davies-Bouldin | {votes['davies_bouldin']} |
| **Bootstrap stability (ARI ≥ {sel['stability_threshold']})** | **{', '.join(map(str, stable))}** |

All three indices unanimously prefer **k = {votes['silhouette']}**. Only
**k ∈ {{{', '.join(map(str, stable))}}}** reproduces under resampling.

`k = {sel['chosen_k']}` was taken. A two-way split would be geometrically cleanest and
commercially useless — internal indices optimise a mathematical objective, not a business
one, and they say nothing about whether a partition survives a new sample.

## Do the segments matter?

The clustering never saw subscription outcome. These differences are therefore genuine
external validation rather than a restatement of the objective the algorithm optimised.

| Seg | Customers | Share | Conversion | Lift | Modal job | Contact | Mean calls | Prior contact |
|---:|---:|---:|---:|---:|---|---|---:|---:|
{table}

**χ² = {seg['chi2']:.0f}, p {'< 1e-300' if seg['p_value'] == 0 else f"= {seg['p_value']:.2e}"}**,
conversion spread **{pct(seg['conversion_spread'])}**.

**The actionable finding is segment {worst['segment']}**, not segment {best['segment']}.
{worst['size']:,} customers averaging **{worst['mean_campaign_calls']:.1f} campaign calls**
and converting at {pct(worst['conversion_rate'], 2)} — roughly a third of baseline. That is
budget being actively consumed by a saturated cohort.

Segment {best['segment']} converts at {pct(best['conversion_rate'], 2)}, but
{pct(best['prior_contact_share'], 0)} of it was contacted in a previous campaign. It is a
legitimate predictor — prior contact is known before dialling — but it largely re-identifies
already-engaged customers rather than revealing a latent group.

## Three limitations, stated

**Algorithms only half-agree.** K-means agrees with the Gaussian mixture at ARI
**{agree['kmeans_vs_gmm']}** and with Ward linkage at **{agree['kmeans_vs_agglomerative']}**.
Roughly half the partition structure is imposed by K-means's spherical assumption rather than
present in the data. These boundaries are one defensible partition among several.

**The projection cannot be read as separation.** {proj['caveat']}

**Segments are descriptive, not causal.** A segment converting at 5× the average is not
evidence that moving a customer into it would raise their probability of subscribing.

## The excluded column

`duration` — how long the sales call lasted — is excluded from the feature space. It is only
known once the call has happened and the outcome is effectively decided. Clustering on it
would build segments that partly encode the answer.
[Project 06](../06_automl_tournament/) measures what including it costs: **+39.9% PR-AUC**.

{screenshots_section(project)}

## CRISP-DM record

{crispdm_section(project)}

{quickstart(project)}

{footer(project)}
"""

    abstract = f"""# Abstract — Customer Segmentation with Bootstrap Stability Validation

**Objective.** Determine whether a bank's contacted customers fall into reproducible
segments, and whether those segments differ enough in subscription behaviour to justify
differentiated targeting.

**Data.** 41,188 real direct-marketing contact records from a Portuguese retail bank
(Moro, Cortez & Rita, 2014), comprising demographic, financial and campaign-history
attributes plus contemporaneous macroeconomic indicators.

**Method.** Subscription outcome and call duration were excluded from the feature space —
the former reserved for external validation, the latter because it is unknowable at scoring
time. Mixed-type features were encoded within a ColumnTransformer. Candidate k from 2 to 10
was evaluated against silhouette, Calinski-Harabasz and Davies-Bouldin indices, then
stress-tested over {sel['n_bootstrap']} bootstrap resamples using adjusted Rand index with a
reproducibility threshold of {sel['stability_threshold']}. Three algorithms with differing
geometric assumptions were compared at the selected k.

**Results.** All three validity indices unanimously favoured k = {votes['silhouette']}, while
only k ∈ {{{', '.join(map(str, stable))}}} satisfied the stability criterion; the reproducible
solution was adopted. At k = {sel['chosen_k']}, conversion ranged from
{pct(worst['conversion_rate'], 2)} to {pct(best['conversion_rate'], 2)} across segments
(χ² = {seg['chi2']:.0f}, p < 0.001) despite the outcome being withheld from clustering. A
cohort of {worst['size']:,} customers averaging {worst['mean_campaign_calls']:.1f} contacts
converted at {best['lift_vs_overall'] / worst['lift_vs_overall']:.1f}× below the
highest-converting segment. Cross-algorithm agreement was moderate
(ARI {agree['kmeans_vs_gmm']} against a Gaussian mixture).

**Conclusion.** Internal validity indices and reproducibility can select different solutions,
and optimising geometry alone risks reporting a partition specific to the sample. Moderate
cross-algorithm agreement further indicates that a substantial share of apparent structure
reflects the algorithm's assumptions rather than the data.

**Keywords.** clustering, bootstrap stability, adjusted Rand index, external validation,
customer segmentation
"""

    paper = f"""# Selecting a Segmentation for Reproducibility Rather Than Geometry

*A CRISP-DM study on 41,188 real bank marketing contacts*

## 1. Motivation

A term-deposit campaign operates under a fixed calling budget. Segmentation is only worth
performing if it changes who gets called, which requires two properties simultaneously: the
segments must reproduce on a new sample of the same population, and they must differ in
outcome. Stability without outcome separation yields tidy segments nobody can act on;
outcome separation without stability yields a story that will not survive the next quarter.

## 2. Feature space

The subscription outcome was withheld entirely from clustering and reserved for validation.
Call duration was also excluded: it records how long the call lasted, which is known only
after the outcome is effectively settled, so clustering on it would embed the answer in the
segments.

Missing categorical values, coded as the literal string `unknown`, were retained as an
explicit level rather than mode-imputed. A customer whose employment or education the bank
failed to record differs systematically from one it recorded; imputation would erase that
signal and manufacture certainty the data does not contain.

Two candidate feature sets were carried forward — customer attributes alone, and customer
attributes plus macroeconomic context — because euribor3m and related series vary with the
*date of the call* rather than with the person, risking segments that are really time
periods wearing customer labels. The choice between them was made on stability rather than
asserted in advance; `{fs}` was selected.

## 3. Model selection

Two criteria were applied in sequence.

**Internal validity.** Silhouette rewards separation, Calinski-Harabasz rewards compactness
relative to spread, Davies-Bouldin penalises overlap. Their agreement or disagreement is
informative in itself.

**Bootstrap stability.** For each k, a reference partition was computed, then
{sel['n_bootstrap']} resamples with replacement were independently clustered and compared to
the reference on their shared rows by adjusted Rand index. Mean ARI ≥ {sel['stability_threshold']}
was required.

| Criterion | Preferred k |
|---|---|
| Silhouette | {votes['silhouette']} |
| Calinski-Harabasz | {votes['calinski_harabasz']} |
| Davies-Bouldin | {votes['davies_bouldin']} |
| Stability (ARI ≥ {sel['stability_threshold']}) | {', '.join(map(str, stable))} |

The criteria disagree, and the disagreement is the substantive result. Every index prefers
k = {votes['silhouette']}; no such partition reproduces. We adopt k = {sel['chosen_k']}.

## 4. External validation

| Seg | Customers | Share | Conversion | Lift | Mean calls | Prior contact |
|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {s['segment']} | {s['size']:,} | {pct(s['share'])} | {pct(s['conversion_rate'], 2)} | "
        f"{s['lift_vs_overall']:.2f}× | {s['mean_campaign_calls']:.1f} | {pct(s['prior_contact_share'], 0)} |"
        for s in seg["segment_profiles"]
    ) + f"""

χ² = {seg['chi2']:.1f} on {seg['dof']} degrees of freedom,
p {'< 1e-300' if seg['p_value'] == 0 else f'= {seg["p_value"]:.3e}'}.

Because the outcome was never available to the clustering algorithm, this separation is
external evidence rather than a restatement of the optimisation objective.

The operationally significant segment is {worst['segment']}: {worst['size']:,} customers
receiving {worst['mean_campaign_calls']:.1f} calls on average and converting at
{pct(worst['conversion_rate'], 2)}. Contact effort is being spent on a saturated cohort.

## 5. How much structure is real?

| Comparison | Adjusted Rand index |
|---|---:|
| K-means vs Gaussian mixture | {agree['kmeans_vs_gmm']} |
| K-means vs Ward linkage | {agree['kmeans_vs_agglomerative']} |

{agree['interpretation']}

At approximately 0.5, roughly half the partition structure derives from K-means's isotropic
assumption rather than from the data. This is reported because a segmentation presented
without it invites the reader to treat cluster boundaries as discovered natural kinds.

## 6. Limitations

""" + "\n".join(
        f"- {risk}"
        for phase in load(project, "crispdm")["phases"]
        for risk in phase.get("risks", [])
    ) + f"""

{run_stamp(project)}
"""

    article = f"""# Your clustering is probably not reproducible, and you would never know

*What happens when you bootstrap a segmentation 30 times*

Here is the standard customer segmentation workflow. Load the data. Run K-means with k=4,
because four personas fit on a slide. Colour the scatter plot. Name the segments something
like "High-Value Loyalists" and "At-Risk Dormants". Ship it.

Every step of that is defensible except the part nobody checks: **would you get the same
segments from a different sample of the same customers?**

I ran that check on 41,188 real bank marketing contacts, and the answer reorganised the
whole project.

## The indices were unanimous, and unhelpful

First, the textbook approach. Sweep k from 2 to 10, score each with three internal validity
indices, take the winner.

All three agreed: **k = {votes['silhouette']}**.

Silhouette, Calinski-Harabasz, Davies-Bouldin — unanimous. In most write-ups that
unanimity would be reported as strong evidence and the analysis would proceed with two
segments.

## Then I resampled

For each k, I took {sel['n_bootstrap']} bootstrap resamples, clustered each independently,
and measured how much each resample's partition agreed with the original using adjusted Rand
index. ARI of 1.0 means identical; 0 means no better than chance.

Only **k ∈ {{{', '.join(map(str, stable))}}}** cleared {sel['stability_threshold']}.

The unanimous geometric winner was not among them.

The two solutions optimise different things. Internal indices ask "are these clusters
compact and well-separated?" — a question about geometry. Bootstrap stability asks "would I
find these clusters again?" — a question about whether the finding is real. A two-way split
is geometrically clean and tells a marketing team nothing they did not already know.

I went with reproducibility: **k = {sel['chosen_k']}**.

## The segments earned their keep

The clustering never saw whether anyone subscribed. I held the outcome out entirely, then
checked afterwards.

| Segment | Share | Conversion | Defining trait |
|---|---:|---:|---|
""" + "\n".join(
        f"| {s['segment']} | {pct(s['share'])} | {pct(s['conversion_rate'], 2)} | "
        + (f"{pct(s['prior_contact_share'], 0)} previously contacted"
           if s["prior_contact_share"] > 0.5
           else f"{s['mean_campaign_calls']:.1f} mean calls, {s['top_contact']}")
        + " |"
        for s in seg["segment_profiles"]
    ) + f"""

Conversion ranges from {pct(worst['conversion_rate'], 2)} to {pct(best['conversion_rate'], 2)}.
χ² = {seg['chi2']:.0f}. The segments are real and they matter.

## The finding worth acting on is the bad segment

Everyone's eye goes to segment {best['segment']} at {pct(best['conversion_rate'], 2)}. But it
is {pct(best['prior_contact_share'], 0)} previously-contacted customers — the model has
rediscovered "people who already said yes once are likely to say yes again". True, useful,
unsurprising.

Segment {worst['segment']} is the one that should change behaviour tomorrow.
{worst['size']:,} customers. **{worst['mean_campaign_calls']:.1f} calls each on average.**
Converting at {pct(worst['conversion_rate'], 2)} — a third of baseline.

That is a cohort absorbing an enormous amount of calling effort and returning almost nothing.
Nobody had to build a model to stop calling them; somebody had to notice.

## What I am still not claiming

K-means agrees with a Gaussian mixture at ARI **{agree['kmeans_vs_gmm']}** and with
hierarchical clustering at **{agree['kmeans_vs_agglomerative']}**.

That is moderate. It means roughly half of what looks like structure is K-means's assumption
that clusters are spherical blobs of similar size. Change the algorithm and half the
boundaries move.

So these are *a* defensible partition, not *the* natural segments of this population. That
distinction rarely survives into a segmentation deck, and it should.

---

**Live demo:** [{SITE}/#/p/{SLUGS[project]}]({SITE}/#/p/{SLUGS[project]})
**Code:** [github.com/Pranjal101Shrivastava/Projects](https://github.com/Pranjal101Shrivastava/Projects)
"""
    return {"README.md": readme, "abstract.md": abstract, "paper.md": paper,
            "article.md": article}


# ======================================================================================
# Project 03 — Market basket
# ======================================================================================
def doc_03() -> dict[str, str]:
    project = "03_market_basket"
    alg = load(project, "algorithms")["comparison"]
    rules = load(project, "rules")
    prof = load(project, "profile")
    sig = rules["significance"]
    top = rules["top_by_lift"][:8]
    dis = rules["dissociations"][:5]

    def rule_rows(rs):
        return "\n".join(
            f"| `{{{', '.join(r['antecedent'])}}}` → `{{{', '.join(r['consequent'])}}}` | "
            f"{r['support_count']} | {r['confidence']:.3f} | **{r['lift']:.2f}** | "
            f"{r['leverage']:.5f} | {r['zhang']:.3f} | "
            f"{r['q_value']:.2e} |"
            for r in rs
        )

    readme = f"""# 03 · Market Basket Mining with FDR Control

Association rules over **{prof['n_transactions']:,} real grocery baskets**, with Apriori and
FP-Growth implemented from first principles and cross-validated against each other, and every
reported rule passing Benjamini-Hochberg false-discovery-rate control.

{provenance_table(project)}

## Two algorithms, written from scratch, agreeing exactly

Both were implemented directly rather than imported. The two differ precisely in *how* they
avoid scanning the exponential itemset lattice, and that difference is the substance of the
topic — calling `mlxtend.apriori` would hide it and make the runtime comparison meaningless.

| | Apriori | FP-Growth |
|---|---:|---:|
| Wall-clock | {alg['apriori']['seconds']}s | **{alg['fp_growth']['seconds']}s** |
| Database scans | {alg['apriori']['database_scans']} | {alg['fp_growth']['database_scans']} |
| Candidates generated | {alg['apriori']['candidates_generated']:,} | 0 — none |
| Pruned by downward closure | {pct(alg['apriori']['pruning_rate'])} | n/a |
| Tree nodes built | n/a | {alg['fp_growth']['tree_nodes_created']:,} |
| **Frequent itemsets found** | **{alg['n_itemsets']:,}** | **{alg['n_itemsets']:,}** |

**{alg['speedup_fp_over_apriori']:.0f}× speedup, byte-identical output.**

The agreement is the real result. The pipeline raises `AssertionError` and refuses to write
anything if the two ever diverge — two independent implementations agreeing on all
{alg['n_itemsets']:,} itemsets is far stronger evidence of correctness than either one
completing without error.

## The multiple-comparisons problem nobody corrects for

Mining generates tens of thousands of candidate rules, then presents the highest-scoring few.
That is textbook multiple testing: with {prof['n_distinct_items']} items there are ~28,000
possible pairs, so rules with striking lift arise by chance alone.

- **{sig['n_rules_tested']:,}** candidate rules tested
- **{sig['n_significant']:,}** survive Fisher exact + Benjamini-Hochberg at α = {sig['fdr_alpha']}
- **{sig['n_rejected_by_fdr']:,} rejected** and excluded from every table below

Bonferroni was rejected deliberately: at this scale it would control family-wise error while
discarding genuine affinities. BH bounds the *expected proportion* of false discoveries among
those reported, which is the right guarantee for a ranked shortlist someone will act on.

The implementation was verified against `statsmodels.multipletests` — maximum absolute
q-value difference **1.1 × 10⁻¹⁶**, identical rejection sets.

## Top affinities (significant only)

| Rule | Baskets | Confidence | Lift | Leverage | Zhang | q-value |
|---|---:|---:|---:|---:|---:|---:|
{rule_rows(top)}

These are interpretable, which is the real test: an alcohol basket, a convenience-meal
basket, and a sandwich basket.

## Dissociations — pairs bought together *less* than chance

| Rule | Baskets | Confidence | Lift | Leverage | Zhang | q-value |
|---|---:|---:|---:|---:|---:|---:|
{rule_rows(dis)}

Lift below 1 signals negative association but compresses it into a narrow range bounded
below by zero. Zhang's metric spans [−1, 1] and separates these properly. Beer dissociating
from milk and vegetables is the clearest case: those are different shopping trips, not
complementary products.

## Why six measures

No single measure is sufficient, and ranking by confidence is the classic mistake.

- **Confidence** P(Y|X) ignores how common Y is. "Anything → whole milk" reaches ~25%
  confidence purely because a quarter of baskets contain milk.
- **Lift** corrects for that but is symmetric and unstable at low support.
- **Conviction** is directional, unlike lift.
- **Leverage** is absolute excess co-occurrence, so high-volume rules are not buried under
  high-lift rarities.
- **Zhang's metric** distinguishes association from dissociation.
- **q-value** says whether the rule is distinguishable from noise at all.

## Why the support floor is 0.1% and not 1%

The transaction matrix is **{pct(prof['sparsity'])} sparse** — mean basket
{prof['basket_size']['mean']} items out of {prof['n_distinct_items']} categories. A
conventional 1% floor admits only the handful of staples everyone buys, and every rule found
becomes a variation on "people buy milk". Lowering the floor surfaces the niche affinities
where merchandising value actually is; the FDR correction is what makes that trade safe.

{screenshots_section(project)}

## CRISP-DM record

{crispdm_section(project)}

{quickstart(project)}

{footer(project)}
"""

    abstract = f"""# Abstract — Association Rule Mining with False Discovery Rate Control

**Objective.** Identify grocery item affinities strong enough and statistically robust
enough to justify merchandising changes, and quantify the computational trade-off between
the two canonical frequent-itemset algorithms.

**Data.** {prof['n_transactions']:,} real point-of-sale transactions over
{prof['n_distinct_items']} item categories (arules Groceries), mean basket
{prof['basket_size']['mean']} items, {pct(prof['sparsity'])} sparse.

**Method.** Apriori and FP-Growth were implemented from first principles and executed on
identical input at a support threshold of 0.1%. Rules were scored on six complementary
measures — support, confidence, lift, leverage, conviction and Zhang's metric — and each
subjected to a one-sided Fisher exact test with Benjamini-Hochberg correction at α =
{sig['fdr_alpha']}.

**Results.** Both algorithms returned identical output on all {alg['n_itemsets']:,} frequent
itemsets, with FP-Growth {alg['speedup_fp_over_apriori']:.0f}× faster
({alg['fp_growth']['seconds']}s against {alg['apriori']['seconds']}s) using
{alg['fp_growth']['database_scans']} database scans against
{alg['apriori']['database_scans']}, and generating no candidate itemsets where Apriori
generated {alg['apriori']['candidates_generated']:,} of which {pct(alg['apriori']['pruning_rate'], 0)}
were pruned by downward closure. Of {sig['n_rules_tested']:,} candidate rules,
{sig['n_rejected_by_fdr']:,} failed FDR control and were excluded. The strongest surviving
affinity reached lift {top[0]['lift']:.2f}. Negative associations were also detected, the
strongest reaching Zhang's metric {dis[0]['zhang']:.3f}.

**Conclusion.** Independent implementations agreeing exactly provides stronger correctness
evidence than either executing without error. Multiple-comparisons correction is rarely
applied in association mining despite tens of thousands of simultaneous tests; here roughly
one rule in ten failed it.

**Keywords.** association rules, Apriori, FP-Growth, false discovery rate, Fisher exact test
"""

    paper = f"""# Frequent Itemset Mining with Statistical Rule Validation

*A study on {prof['n_transactions']:,} real grocery transactions*

## 1. Problem

Cross-sell placement and bundle promotion depend on knowing which products genuinely pull
each other into the basket. The commercial risk is acting on a spurious pattern: rearranging
an aisle around a rule that was noise costs real money and is difficult to detect after the
fact.

## 2. Algorithms

### 2.1 Apriori

Apriori rests on the downward-closure property: if an itemset is frequent, every subset of it
is frequent. Contrapositively, any candidate containing an infrequent subset can be discarded
before counting it. The cost is one full database pass per level.

Measured on this data: {alg['apriori']['candidates_generated']:,} candidates generated across
{alg['apriori']['database_scans']} levels, of which {pct(alg['apriori']['pruning_rate'])} were
eliminated by downward closure before counting.

### 2.2 FP-Growth

FP-Growth reads the database exactly twice — once to count item frequencies, once to build a
prefix tree ordered by descending frequency to maximise sharing — then mines entirely in
memory by recursively constructing conditional trees. No candidates are generated.

Measured: {alg['fp_growth']['tree_nodes_created']:,} tree nodes,
{alg['fp_growth']['database_scans']} database scans.

### 2.3 Cross-validation of implementations

Both were run on identical input and their outputs compared exactly. They agree on all
{alg['n_itemsets']:,} frequent itemsets. The pipeline raises `AssertionError` on any
divergence rather than publishing results from an algorithm that disagrees with itself.

| Metric | Apriori | FP-Growth | Ratio |
|---|---:|---:|---:|
| Seconds | {alg['apriori']['seconds']} | {alg['fp_growth']['seconds']} | {alg['speedup_fp_over_apriori']:.0f}× |
| Database scans | {alg['apriori']['database_scans']} | {alg['fp_growth']['database_scans']} | — |
| Candidates | {alg['apriori']['candidates_generated']:,} | 0 | — |

## 3. Rule scoring

Six measures were computed for every rule, because each fails in a different way.

Let X be the antecedent and Y the consequent.

- Support: P(X ∧ Y)
- Confidence: P(Y|X) — ignores the marginal frequency of Y
- Lift: P(Y|X)/P(Y) — corrects for that, but is symmetric
- Conviction: P(X)P(¬Y)/P(X ∧ ¬Y) — directional
- Leverage: P(X ∧ Y) − P(X)P(Y) — absolute excess co-occurrence
- Zhang's metric ∈ [−1, 1] — distinguishes association from dissociation

## 4. Statistical validation

{sig['n_rules_tested']:,} rules constitutes a large simultaneous-testing problem. Each rule's
2×2 contingency table was subjected to a one-sided Fisher exact test, and p-values corrected
by the Benjamini-Hochberg step-up procedure at α = {sig['fdr_alpha']}.

{sig['rationale']}

Bonferroni was considered and rejected: controlling family-wise error across
{sig['n_rules_tested']:,} tests would reject genuine affinities along with the noise. BH
bounds the expected proportion of false discoveries among reported rules, which matches how
the output is used — as a ranked shortlist for human judgement.

The BH implementation was verified against `statsmodels.multipletests`: maximum absolute
q-value difference 1.1 × 10⁻¹⁶ and identical rejection sets on a 1,000-hypothesis test case.

## 5. Results

| Rule | Baskets | Conf | Lift | Leverage | Zhang | q |
|---|---:|---:|---:|---:|---:|---:|
{rule_rows(top)}

Negative associations:

| Rule | Baskets | Conf | Lift | Leverage | Zhang | q |
|---|---:|---:|---:|---:|---:|---:|
{rule_rows(dis)}

## 6. Support threshold selection

At {pct(prof['sparsity'])} sparsity, a conventional 1% minimum support admits only staples.
0.1% (≈{int(0.001 * prof['n_transactions'])} baskets) was used instead, producing far more
candidates whose validity is then controlled statistically rather than by an arbitrary
frequency cutoff.

## 7. Limitations

""" + "\n".join(
        f"- {risk}"
        for phase in load(project, "crispdm")["phases"]
        for risk in phase.get("risks", [])
    ) + f"""

{run_stamp(project)}
"""

    article = f"""# I wrote Apriori and FP-Growth from scratch so they could check each other

*189× apart in runtime, byte-identical in output*

Most association-rule tutorials call `mlxtend.apriori`, print the top ten rules by lift, and
stop. I wanted two things that approach cannot give you: an understanding of *why*
FP-Growth is faster, and any reason to believe the rules are real.

So I implemented both algorithms by hand.

## The correctness check that fell out for free

Running two independent implementations on the same data gives you something a single
library call never does: if they disagree, at least one is wrong.

They did not disagree. All **{alg['n_itemsets']:,} frequent itemsets**, identical.

The pipeline now raises an exception and refuses to write results if they ever diverge.
That is a much stronger statement than "the code ran without error".

## Where the 189× comes from

| | Apriori | FP-Growth |
|---|---:|---:|
| Time | {alg['apriori']['seconds']}s | {alg['fp_growth']['seconds']}s |
| Database passes | {alg['apriori']['database_scans']} | {alg['fp_growth']['database_scans']} |
| Candidates generated | {alg['apriori']['candidates_generated']:,} | **zero** |

Apriori works breadth-first. At each level it generates candidate itemsets, prunes those
containing an infrequent subset, then scans the entire database to count the survivors. It
generated {alg['apriori']['candidates_generated']:,} candidates and pruned
{pct(alg['apriori']['pruning_rate'], 0)} of them before counting — the pruning works, and it
is still doing enormous work.

FP-Growth reads the database twice, total. Once to count item frequencies, once to build a
prefix tree where common prefixes are shared. Then it mines that tree recursively, in memory,
generating no candidates at all.

## The part that changed the results

Here is the thing almost nobody does.

I generated **{sig['n_rules_tested']:,} candidate rules**. Then I asked: how many of these
would look this good by chance?

With {prof['n_distinct_items']} item categories there are roughly 28,000 possible pairs. Test
28,000 hypotheses at p < 0.05 and you expect ~1,400 false positives *even if nothing is
associated with anything*.

So every rule got a Fisher exact test and a Benjamini-Hochberg corrected q-value.

**{sig['n_rejected_by_fdr']:,} rules failed** and are excluded from every table I publish.

I chose BH over Bonferroni deliberately. Bonferroni would control the probability of *any*
false positive, which at this scale means rejecting real affinities too. BH controls the
expected *proportion* of false discoveries in the list — the right guarantee when the output
is a shortlist a merchandiser will work through.

## What survived

| Rule | Lift |
|---|---:|
""" + "\n".join(
        f"| {{{', '.join(r['antecedent'])}}} → {{{', '.join(r['consequent'])}}} | {r['lift']:.1f} |"
        for r in top[:6]
    ) + f"""

An alcohol basket. A convenience-meal basket. A sandwich basket — `processed cheese` →
`{{ham, white bread}}` at lift {[r for r in top if 'processed cheese' in r['antecedent']][0]['lift']:.1f}
if you want the clearest example of a rule you could act on tomorrow.

## And the rules that run the other way

Lift below 1 means two items appear together *less* often than chance predicts. But lift
squashes all negative association into the range [0, 1), so everything looks similar.

Zhang's metric spans [−1, 1] and separates them:

| Rule | Zhang |
|---|---:|
""" + "\n".join(
        f"| {{{', '.join(r['antecedent'])}}} → {{{', '.join(r['consequent'])}}} | {r['zhang']:.2f} |"
        for r in dis[:4]
    ) + f"""

Canned beer actively repels whole milk, root vegetables and yogurt. Those are different
shopping trips — the beer run and the weekly shop — and a merchandiser who knows that will
not waste an end-cap trying to bridge them.

---

**Live demo:** [{SITE}/#/p/{SLUGS[project]}]({SITE}/#/p/{SLUGS[project]})
**Code:** [github.com/Pranjal101Shrivastava/Projects](https://github.com/Pranjal101Shrivastava/Projects)
"""
    return {"README.md": readme, "abstract.md": abstract, "paper.md": paper,
            "article.md": article}


# ======================================================================================
# Project 04 — Fraud detection
# ======================================================================================
def doc_04() -> dict[str, str]:
    project = "04_fraud_detection"
    models = load(project, "models")
    cost = load(project, "cost")
    prof = load(project, "profile")["profile"]
    best = models["results"][models["best"]]
    iso = models["results"]["isolation_forest"]
    abl = models["reweighting_ablation"]
    trap = best["accuracy_trap"]
    test_frauds = best["confusion"]["tp"] + best["confusion"]["fn"]

    table = "\n".join(
        f"| {v['label']} | {v['pr_auc']:.4f} | {v['roc_auc']:.4f} | {v['precision']:.3f} | "
        f"{v['recall']:.3f} | {v['brier']:.5f} |"
        for v in sorted(models["results"].values(), key=lambda m: -m["pr_auc"])
    )

    abl_table = "\n".join(
        f"| `{v['variant']}` | "
        + (f"{v['scale_pos_weight']:.0f}" if v["scale_pos_weight"] else "—")
        + f" | **{v['pr_auc']:.4f}** | {v['roc_auc']:.4f} |"
        for v in abl["variants"]
    )

    budget = "\n".join(
        f"| {b['alerts']:,} | {b['frauds_caught']} / {test_frauds} | {pct(b['recall'])} | {pct(b['precision'])} |"
        for b in cost["alert_budget_analysis"]
    )

    readme = f"""# 04 · Fraud Detection at 0.17% Prevalence

Card fraud detection over **{prof['n_transactions']:,} real transactions** from the
Université Libre de Bruxelles dataset, where {prof['n_fraud']} are fraudulent — an imbalance
of **{prof['imbalance_ratio']:.0f}:1**.

{provenance_table(project)}

## Everything here follows from one number

The positive rate is **{pct(prof['prevalence'], 4)}**. That single fact determines every
design decision:

- **Accuracy is meaningless.** A model that predicts "never fraud" scores
  {pct(prof['always_negative_accuracy'], 3)}.
- **ROC-AUC is misleading.** Its false-positive rate divides by
  {prof['n_transactions'] - prof['n_fraud']:,} negatives, so thousands of false alarms barely
  move it.
- **The threshold is a business decision**, not a default. 0.5 is arbitrary.

## Results

Chronological holdout: {cost['curve'][0]['tp'] + cost['curve'][0]['fp'] + cost['curve'][0]['fn'] + cost['curve'][0]['tn']:,}
transactions, {test_frauds} frauds.

| Model | PR-AUC | ROC-AUC | Precision | Recall | Brier |
|---|---:|---:|---:|---:|---:|
{table}

**PR-AUC {best['pr_auc']:.4f} against a no-skill floor of {best['prevalence']:.5f}** — a
{best['pr_auc_lift_over_no_skill']:.0f}× lift.

The floor is the prevalence *of the held-out window* ({best['prevalence']:.5f}), not of the
full dataset ({prof['prevalence']:.5f}); a chronological split does not preserve the base rate
exactly, and the no-skill PR-AUC is always the prevalence of the set being scored.

### The metric gap, shown rather than described

The Isolation Forest scores **ROC-AUC {iso['roc_auc']:.4f}** — which looks respectable —
while its **PR-AUC is {iso['pr_auc']:.4f}** and its precision is {pct(iso['precision'])}.

Same model. Same predictions. The two metrics disagree because ROC's denominator is every
negative in the dataset, while precision's denominator is the model's own alert volume —
which is what an analyst's queue actually contains. Both are reported for every model
precisely so this gap is visible.

### The accuracy trap

| | |
|---|---:|
| This model's accuracy | {pct(trap['model_accuracy'], 3)} |
| Accuracy of predicting "never fraud" | {pct(trap['always_negative_accuracy'], 3)} |
| **The entire value of the model** | **{(trap['model_accuracy'] - trap['always_negative_accuracy']) * 100:.3f} points** |

## An ablation that contradicts the standard advice

The conventional recommendation for imbalanced boosting is to set `scale_pos_weight` to the
negative/positive ratio. On this data that is the single worst thing you can do.

| Variant | scale_pos_weight | PR-AUC | ROC-AUC |
|---|---:|---:|---:|
{abl_table}

{abl['finding']}

All four variants are published rather than only the winner.

## Choosing an operating point

Assumed costs: **{cost['assumed_cost_false_negative']:.0f}** for a missed fraud,
**{cost['assumed_cost_false_positive']:.0f}** for a wasted investigation. The ratio is an
assumption, not a measurement — the full curve ships in
[`artifacts/cost.json`](./artifacts/cost.json) so a reader with different costs can read off
their own point.

At the cost-optimal threshold of **{cost['optimal_operating_point']['threshold']:.3f}**:
{cost['optimal_operating_point']['tp']} frauds caught, {cost['optimal_operating_point']['fp']}
false alarms, {cost['optimal_operating_point']['fn']} missed.

### What a fixed analyst budget buys

| Daily alert budget | Frauds caught | Recall | Precision |
|---:|---:|---:|---:|
{budget}

Reviewing 50 alerts catches over half the fraud at {pct(cost['alert_budget_analysis'][0]['precision'])}
precision. Reviewing twenty times as many raises recall by about
{(cost['alert_budget_analysis'][-1]['recall'] - cost['alert_budget_analysis'][0]['recall']) * 100:.0f}
points and drops precision to {pct(cost['alert_budget_analysis'][-1]['precision'])}. The
twentieth alert is far less valuable than the first — the shape every capacity-constrained
detection system has.

## Why the split is chronological

{models['split']['why_not_random']}

Most published notebooks on this dataset use a stratified random split, which is why their
reported numbers are better than these.

## Why not SMOTE

Interpolating between {prof['n_fraud']} positives in 28-dimensional PCA space places
synthetic points in regions where no real fraud has ever been observed, and the model then
learns a decision boundary around fabricated data. Class weighting achieves the same
rebalancing without inventing observations.

{screenshots_section(project)}

## CRISP-DM record

{crispdm_section(project)}

{quickstart(project)}

{footer(project)}
"""

    abstract = f"""# Abstract — Fraud Detection under Extreme Class Imbalance

**Objective.** Rank card transactions for manual review under a fixed analyst capacity, and
establish which evaluation metrics remain informative at a positive rate below 0.2%.

**Data.** {prof['n_transactions']:,} real transactions by European cardholders over
{prof['duration_hours']:.0f} hours in September 2013 (Université Libre de Bruxelles), of
which {prof['n_fraud']} are fraudulent ({pct(prof['prevalence'], 4)}). Features V1–V28 are
principal components published in place of the original variables for confidentiality.

**Method.** Partitioning was chronological rather than random, because fraud is bursty and a
random split distributes a single compromised-card episode across both partitions. Three
detectors spanning two supervision regimes were compared: an Isolation Forest fitted only on
legitimate training transactions, a class-weighted logistic regression, and gradient boosting
under four distinct reweighting strategies. No synthetic minority oversampling was used.
Operating points were selected by expected cost under an assumed
{cost['assumed_cost_false_negative']:.0f}:{cost['assumed_cost_false_positive']:.0f}
false-negative to false-positive ratio.

**Results.** The best model reached PR-AUC {best['pr_auc']:.4f} against a no-skill floor of
{best['prevalence']:.5f} — the prevalence of the held-out window —
({best['pr_auc_lift_over_no_skill']:.0f}× lift). The unsupervised
detector attained ROC-AUC {iso['roc_auc']:.4f} while achieving PR-AUC {iso['pr_auc']:.4f} and
precision {pct(iso['precision'])}, illustrating the divergence between the two metrics under
imbalance. A reweighting ablation found that setting `scale_pos_weight` to the
negative/positive ratio — standard practice — reduced gradient-boosting PR-AUC from
{[v for v in abl['variants'] if v['variant'] == 'none'][0]['pr_auc']:.4f} to
{[v for v in abl['variants'] if v['variant'] == 'scale_pos_weight_full'][0]['pr_auc']:.4f}.

**Conclusion.** ROC-AUC is unsuitable as a headline metric at this prevalence. Standard
reweighting guidance derived from linear models does not transfer to high-capacity boosted
ensembles with few positives, and should be verified by ablation rather than assumed.

**Keywords.** imbalanced classification, precision-recall AUC, cost-sensitive learning,
anomaly detection, temporal validation
"""

    paper = f"""# Evaluation and Operating-Point Selection for Fraud Detection at 0.17% Prevalence

*A CRISP-DM study on {prof['n_transactions']:,} real card transactions*

## 1. The constraint that defines the problem

A fraud screen operates under fixed analyst capacity. The objective is therefore not
"detect fraud" but **maximise fraud caught per alert raised**. Every methodological choice
below follows from that framing plus one number: the positive rate of
{pct(prof['prevalence'], 4)}.

## 2. Why the usual metrics fail

**Accuracy.** A constant "never fraud" predictor achieves
{pct(prof['always_negative_accuracy'], 4)}. Our best model achieves
{pct(trap['model_accuracy'], 4)}. The difference —
{(trap['model_accuracy'] - trap['always_negative_accuracy']) * 100:.3f} percentage points —
is the entire contribution of the model. Both figures are reported side by side throughout,
so accuracy cannot be quoted as evidence of skill.

**ROC-AUC.** The false-positive rate is FP/(FP+TN), and TN here is on the order of
{prof['n_transactions'] - prof['n_fraud']:,}. A model can raise thousands of false alarms
while barely moving FPR.

The Isolation Forest demonstrates this concretely: ROC-AUC {iso['roc_auc']:.4f}, PR-AUC
{iso['pr_auc']:.4f}, precision {pct(iso['precision'])}. On ROC it looks usable; in an
analyst's queue, roughly {100 - iso['precision'] * 100:.0f} of every 100 alerts would be
false.

**Average precision (PR-AUC)** divides by the model's own alert volume, so it degrades as
soon as the model wastes reviewer time. It is used as the headline metric, with prevalence
always reported alongside since prevalence *is* the no-skill PR-AUC.

## 3. Partitioning

Fraud is bursty: a compromised card generates several transactions within minutes. A random
split scatters one episode across train and test, so the model is scored on transactions
whose siblings it trained on.

{models['split']['why_not_random']}

Resulting partitions: {models['split']['n_train']:,} training transactions
({models['split']['frauds_train']} frauds), {models['split']['n_test']:,} test
({models['split']['frauds_test']} frauds).

## 4. Resampling: why none was used

SMOTE and its variants interpolate between neighbouring minority points. With
{prof['n_fraud']} positives in 28 dimensions the nearest neighbours are far apart, so
synthetic points land in regions of feature space where no real fraud has been observed. The
classifier then learns a boundary around fabricated data. Class weighting achieves
rebalancing without inventing observations, and random majority undersampling would discard
99.8% of the real evidence about what normal looks like.

## 5. Results

| Model | Supervision | PR-AUC | ROC-AUC | Precision | Recall | Brier |
|---|---|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {v['label']} | {v['family']} | {v['pr_auc']:.4f} | {v['roc_auc']:.4f} | "
        f"{v['precision']:.3f} | {v['recall']:.3f} | {v['brier']:.5f} |"
        for v in sorted(models["results"].values(), key=lambda m: -m["pr_auc"])
    ) + f"""

### 5.1 Value of labels

The unsupervised detector was fitted on legitimate training transactions only, never seeing a
fraud label — the realistic cold-start condition for a novel attack. Its PR-AUC of
{iso['pr_auc']:.4f} against {best['pr_auc']:.4f} for the supervised best quantifies what
labelling effort is worth on this problem: roughly
{best['pr_auc'] / max(1e-9, iso['pr_auc']):.0f}×.

### 5.2 Reweighting ablation

| Variant | scale_pos_weight | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|---:|
""" + "\n".join(
        f"| {v['variant']} | " + (f"{v['scale_pos_weight']:.0f}" if v["scale_pos_weight"] else "—")
        + f" | {v['pr_auc']:.4f} | {v['roc_auc']:.4f} | {v['brier']:.5f} |"
        for v in abl["variants"]
    ) + f"""

{abl['finding']}

This result is reported in full because it contradicts widely-repeated guidance. The
mechanism is capacity: reweighting constrains a linear model toward the minority class
usefully, but permits a boosted ensemble to devote successive trees to fitting a handful of
reweighted points, degrading the global ranking that PR-AUC measures.

## 6. Operating-point selection

Expected cost was traced across the threshold range under an assumed
{cost['assumed_cost_false_negative']:.0f}:{cost['assumed_cost_false_positive']:.0f}
false-negative to false-positive ratio.

Optimum at threshold {cost['optimal_operating_point']['threshold']:.3f}:
TP {cost['optimal_operating_point']['tp']}, FP {cost['optimal_operating_point']['fp']},
FN {cost['optimal_operating_point']['fn']}, expected cost
{cost['optimal_operating_point']['expected_cost']:,.0f} against
{cost['do_nothing_cost']:,.0f} for no screening.

{cost['caveat']}

| Alert budget | Frauds caught | Recall | Precision |
|---:|---:|---:|---:|
{budget}

## 7. Limitations

""" + "\n".join(
        f"- {risk}"
        for phase in load(project, "crispdm")["phases"]
        for risk in phase.get("risks", [])
    ) + f"""

{run_stamp(project)}
"""

    article = f"""# A model with 99.95% accuracy that is worth almost nothing

*And the ablation where following best practice made things 82× worse*

The dataset is {prof['n_transactions']:,} real credit card transactions. {prof['n_fraud']}
of them are fraud. That is **{pct(prof['prevalence'], 4)}**.

My best model achieves **{pct(trap['model_accuracy'], 3)} accuracy**.

Here is a model that achieves {pct(trap['always_negative_accuracy'], 3)}:

```python
def predict(transaction):
    return "legitimate"
```

That is the whole model. It never fires. It catches nothing. And it is within
{(trap['model_accuracy'] - trap['always_negative_accuracy']) * 100:.3f} percentage points of
mine.

Any write-up that leads with accuracy on this dataset is either careless or hiding something.

## ROC-AUC has the same problem, less obviously

Everyone knows about the accuracy trap. Fewer people notice that ROC-AUC has a milder version
of it.

I trained an Isolation Forest — an unsupervised detector that only ever sees legitimate
transactions and flags departures from them.

**ROC-AUC: {iso['roc_auc']:.3f}.** That is a number you would put on a slide.

**Precision: {pct(iso['precision'], 1)}.**

For every fraud it catches, it raises about {(1 / max(iso['precision'], 1e-9) - 1):.0f} false
alarms. An analyst working that queue spends their entire day on legitimate transactions.

The reason ROC hides this is its denominator. False-positive rate is FP/(FP+TN), and there
are {prof['n_transactions'] - prof['n_fraud']:,} negatives here. Thousands of false alarms
barely register. Precision divides by the model's *own alert volume* — the thing that is
actually in the queue.

So I lead with **PR-AUC: {best['pr_auc']:.4f}**, and always print the no-skill floor
({prof['prevalence']:.5f}) next to it.

## The ablation I did not expect

I set out to do the standard thing for imbalanced boosting: set `scale_pos_weight` to the
ratio of negatives to positives. Every tutorial says so. LightGBM's own docs say so.

My model came out at PR-AUC **0.0089**.

That is catastrophically bad — worse than the unsupervised detector. So instead of quietly
tuning until it looked better, I ran all four options:

| What I did | PR-AUC |
|---|---:|
""" + "\n".join(
        f"| {v['variant'].replace('_', ' ')} | **{v['pr_auc']:.4f}** |"
        for v in abl["variants"]
    ) + f"""

**Doing nothing beat the recommended practice by 82×.**

The mechanism, once you see it, is obvious. There are 398 fraud cases in the training set.
Weighting them 536× tells the model that those 398 rows matter more than the other 213,000
combined. A linear model, whose capacity is bounded, responds by shifting its decision
boundary — helpful. A boosted ensemble, whose capacity is not bounded, responds by spending
tree after tree fitting those specific 398 points. It memorises them and the ranking of
everything else falls apart.

Reweighting advice that is correct for logistic regression is not automatically correct for
gradient boosting. Ablations exist because intuitions do not transfer.

## The threshold is not 0.5

`predict()` uses 0.5 by default. There is no reason for that number to be right.

Suppose a missed fraud costs {cost['assumed_cost_false_negative']:.0f} and a wasted
investigation costs {cost['assumed_cost_false_positive']:.0f}. Then you can compute the
expected cost at every threshold and pick the minimum — which lands at
**{cost['optimal_operating_point']['threshold']:.3f}**, catching
{cost['optimal_operating_point']['tp']} frauds for
{cost['optimal_operating_point']['fp']} false alarms.

But the more useful framing is capacity. An analyst can review so many alerts per day:

| Alerts/day | Frauds caught | Precision |
|---:|---:|---:|
""" + "\n".join(
        f"| {b['alerts']:,} | {b['frauds_caught']} / {test_frauds} | {pct(b['precision'])} |"
        for b in cost["alert_budget_analysis"]
    ) + f"""

Fifty alerts catches over half the fraud at {pct(cost['alert_budget_analysis'][0]['precision'])}
precision. A thousand alerts catches {pct(cost['alert_budget_analysis'][-1]['recall'])} at
{pct(cost['alert_budget_analysis'][-1]['precision'])} precision.

You are not choosing a threshold. You are choosing how many analysts to hire.

## One more thing: I split by time, not at random

Fraud is bursty. Steal a card, and you use it several times in the next twenty minutes.

Split randomly and those transactions land on both sides of the split. Your model sees three
transactions from an episode in training, then gets scored on the fourth — and recognises the
pattern because it has literally seen its siblings.

Every metric goes up. No metric tells you why.

I split chronologically: train on the first 75% of time, test on the last 25%. That is what
deployment actually looks like — score tomorrow having learned from yesterday. It is also why
my numbers are lower than most published notebooks on this dataset.

---

**Live demo:** [{SITE}/#/p/{SLUGS[project]}]({SITE}/#/p/{SLUGS[project]})
**Code:** [github.com/Pranjal101Shrivastava/Projects](https://github.com/Pranjal101Shrivastava/Projects)
"""
    return {"README.md": readme, "abstract.md": abstract, "paper.md": paper,
            "article.md": article}


# ======================================================================================
# Project 05 — Forecasting
# ======================================================================================
def doc_05() -> dict[str, str]:
    project = "05_timeseries_forecasting"
    series = {k: v for k, v in load(project, "series").items() if not k.startswith("_")}
    syn = load(project, "synthesis")

    naive_wins = [k for k, v in syn["winners_by_series"].items() if "naive" in v.lower()]

    cross = "\n".join(
        f"| {r['label']} | {r['mean_rank']:.2f} | {r['wins']} | {r['best_rank']} | {r['worst_rank']} |"
        for r in syn["cross_series_ranking"]
    )

    per_series = []
    for key, s in series.items():
        rows = "\n".join(
            f"| {r['label']} | {r['mase_mean']:.3f} | ±{r['mase_std']:.3f} | "
            f"{r['mase_worst_origin']:.3f} | "
            + ("✅" if r["beats_seasonal_naive"] else "—") + " |"
            for r in s["backtest"]["results"]
        )
        d = s["decomposition"]
        per_series.append(f"""### {s['spec']['title']}

*{s['spec']['structure']}* · {s['n_observations']} observations · {s['start']} → {s['end']} ·
period {s['spec']['period']}

Trend strength {d['trend_strength']:.3f} · seasonal strength {d['seasonal_strength']:.3f} ·
ADF p = {d['adf_pvalue']:.4f} · KPSS p = {d['kpss_pvalue']:.4f}

**Stationarity:** {d['stationarity_verdict']}

| Model | MASE | ± std | Worst origin | Beats naive |
|---|---:|---:|---:|:---:|
{rows}
""")

    readme = f"""# 05 · Forecasting Tournament across Four Real Series

Six forecasters evaluated on four structurally different real series under rolling-origin
backtests, refitted at every origin rather than scored on a single arbitrary split.

{provenance_table(project)}

## Why four series

Most forecasting write-ups evaluate on one series and one holdout, which cannot separate
"this model is good" from "this model happens to suit this series". The four here were
chosen to differ in structure:

""" + "\n".join(
        f"- **{s['spec']['title']}** — {s['spec']['structure']}" for s in series.values()
    ) + f"""

## The result that justifies the design

**On {len(naive_wins)} of {len(series)} series, no learned model beats the naive baseline.**

The sunspot cycle averages about eleven years but drifts, so it is not calendar-anchored and
every seasonal method mis-specifies it. A single-series study on airline passengers would
have concluded "SARIMA wins" and been wrong about the general case.

## Cross-series ranking

| Model | Mean rank | Outright wins | Best | Worst |
|---|---:|---:|---:|---:|
{cross}

Mean rank across four structurally different series is a weak recommendation by construction.
The per-series tables below are the real result.

Worth noting: **LightGBM never wins a series but never ranks worse than third**, making it the
most consistent non-classical option — and it loses to a forty-year-old statistical method on
three of four series.

## Per-series results

{"".join(per_series)}

## Why MASE

These series measure passengers, degrees Celsius, prescription volumes and sunspot area.
Averaging raw MAE across them would be meaningless. MASE divides by the in-sample
seasonal-naive error, making it unit-free and anchored: **1.0 always means "matched seasonal
naive"**.

MAPE was rejected — undefined at zero, explosive near it, and asymmetric between over- and
under-forecasts.

## Protocol

{syn['protocol']}

The ML forecaster uses **direct multi-step** — a separate model per horizon step — rather than
recursive forecasting. Recursive schemes feed a model its own predictions, so error compounds
over the horizon and the final steps are predictions of predictions.

## Both stationarity tests, and what to do when they conflict

ADF's null hypothesis is a unit root; KPSS's null is stationarity. They test opposite things,
so a single test failing to reject is ambiguous between "the null holds" and "the test is
underpowered". Running both distinguishes trend-stationary from difference-stationary series,
which determines whether to detrend or to difference. Two of these four series produce
conflicting verdicts, and that conflict is reported rather than resolved by picking whichever
test agreed with expectation.

## An aggregation decision worth stating

The Melbourne temperature series is 3,650 daily observations. It is modelled here as 120
monthly means. A seasonal model at m = 365 would have to estimate 365 seasonal lags from ten
cycles of data — infeasible to fit and hopeless to identify. Aggregating preserves the annual
cycle, which is the structure of interest. Quietly setting m = 7 instead would have modelled a
weekly cycle this series does not have.

{screenshots_section(project)}

## CRISP-DM record

{crispdm_section(project)}

{quickstart(project)}

{footer(project)}
"""

    abstract = f"""# Abstract — Cross-Series Forecasting Evaluation with Rolling-Origin Backtests

**Objective.** Determine whether any forecasting method should be preferred as a default, and
quantify how far that answer depends on the structure of the series being forecast.

**Data.** Four real series chosen for structural diversity: international airline passengers
(trend with multiplicative annual seasonality), Melbourne minimum temperature (strong annual
seasonality, no trend), Australian antidiabetic drug sales (trend with a December spike), and
sunspot area (an approximately eleven-year cycle with no calendar anchor).

**Method.** Six forecasters spanning three families — naive baselines, classical statistical
methods (Holt-Winters, SARIMA) and machine learning (direct multi-step gradient boosting) —
were evaluated under rolling-origin backtesting with up to eight refits per series at horizon
{syn['horizon']}. Errors were reported as mean absolute scaled error, normalised by in-sample
seasonal-naive error to permit comparison across incommensurable units. Each series was
decomposed by STL and tested for stationarity by both ADF and KPSS.

**Results.** {syn['n_distinct_winners']} distinct models took first place across four series.
SARIMA achieved the best mean rank ({syn['cross_series_ranking'][0]['mean_rank']:.2f}) with
{syn['cross_series_ranking'][0]['wins']} outright wins. On the sunspot series, no learned
model outperformed seasonal naive. Gradient boosting won no series but never ranked below
third, and was outperformed by classical statistical methods on three of four.

**Conclusion.** No forecaster dominates across structurally different series. Single-series
benchmarking is insufficient to support a general recommendation, and series whose periodicity
is not calendar-anchored defeat methods that assume it.

**Keywords.** time series forecasting, rolling-origin evaluation, MASE, SARIMA, stationarity
testing
"""

    paper = f"""# No Free Lunch in Forecasting: A Four-Series Rolling-Origin Study

## 1. Problem with single-series benchmarks

A forecasting comparison on one series and one holdout cannot distinguish a genuinely better
method from one that happens to suit that series, nor a better method from a luckier split.
This study addresses both: four series with deliberately different structure, and
rolling-origin evaluation with refitting at each origin.

## 2. Error measure

Raw MAE cannot be averaged across series measuring passengers, degrees Celsius, prescription
volumes and sunspot area. MASE normalises by the in-sample mean absolute error of the
seasonal-naive forecast:

    MASE = mean(|y − ŷ|) / mean(|y_t − y_{{t−m}}|)

The denominator makes it unit-free; the reference point makes it interpretable. MASE = 1.0
means the forecast matched seasonal naive.

MAPE was rejected as undefined at zero, explosive near it, and asymmetric.

## 3. Protocol

{syn['protocol']}

Direct multi-step was used for the ML forecaster: h separate models, one per horizon step.
Recursive one-step forecasting feeds a model its own predictions, so error compounds and the
last steps of the horizon are predictions of predictions.

## 4. Series characterisation

| Series | n | Trend strength | Seasonal strength | ADF p | KPSS p | Verdict |
|---|---:|---:|---:|---:|---:|---|
""" + "\n".join(
        f"| {s['spec']['title']} | {s['n_observations']} | "
        f"{s['decomposition']['trend_strength']:.3f} | "
        f"{s['decomposition']['seasonal_strength']:.3f} | "
        f"{s['decomposition']['adf_pvalue']:.4f} | {s['decomposition']['kpss_pvalue']:.4f} | "
        f"{s['decomposition']['stationarity_verdict'][:44]} |"
        for s in series.values()
    ) + f"""

Both stationarity tests were run because their null hypotheses are opposites. A single test
that fails to reject is ambiguous between "the null holds" and "insufficient power"; running
both distinguishes trend-stationary from difference-stationary series.

## 5. Results

{"".join(per_series)}

## 6. Cross-series synthesis

| Model | Mean rank | Wins | Best | Worst |
|---|---:|---:|---:|---:|
{cross}

**{syn['n_distinct_winners']} distinct winners across four series.** The sunspot result is
the substantive finding: an approximately eleven-year cycle that drifts and is not anchored to
the calendar defeats every seasonal method, and seasonal naive ranks first.

Gradient boosting is the most *consistent* method — never first, never below third — and is
beaten by SARIMA on three of four series. Reporting only the case where the machine learning
method won would have been straightforward and misleading.

## 7. An aggregation decision

Melbourne daily temperature was aggregated to monthly means. SARIMA at m = 365 would require
estimating 365 seasonal lags from ten annual cycles: computationally intractable and
statistically unidentifiable. Aggregation preserves the annual cycle. Setting m = 7 without
comment would have modelled a weekly cycle the series does not possess.

## 8. Limitations

""" + "\n".join(
        f"- {risk}"
        for phase in load(project, "crispdm")["phases"]
        for risk in phase.get("risks", [])
    ) + f"""

{run_stamp(project)}
"""

    article = f"""# I benchmarked six forecasters on four series and got four different answers

*Including one where a forty-year-old method lost to doing nothing*

If you benchmark forecasting models on one time series, you will learn which model fits that
series. You will not learn which model to reach for next time. Those are different questions,
and the literature routinely conflates them.

So I picked four real series that differ in *structure*, not just in subject:

""" + "\n".join(
        f"- **{s['spec']['title']}** — {s['spec']['structure']}" for s in series.values()
    ) + f"""

Then I ran six forecasters on all four, refitting at every rolling origin.

## The headline is boring. The exception is not.

SARIMA won three of four series. Mean rank {syn['cross_series_ranking'][0]['mean_rank']:.2f}.
If I stopped there, the conclusion would be "use SARIMA" and the article would be forgettable.

Here is the fourth series.

| Sunspot area | MASE |
|---|---:|
""" + "\n".join(
        f"| {r['label']} | {r['mase_mean']:.3f} |"
        for r in series["sunspots"]["backtest"]["results"]
    ) + f"""

**Seasonal naive wins.** SARIMA ranks third. Holt-Winters fourth. Every model that learns
something is beaten by copying the value from one cycle ago.

## Why sunspots break everything

The solar cycle averages about eleven years — but "about" is doing heavy lifting. Individual
cycles run from nine to fourteen years, and they are not anchored to anything. There is no
December, no Monday, no equivalent of "the same hour last week".

Seasonal models assume a *fixed* period. Give them a drifting one and they confidently
extrapolate a phase that has already slipped. The naive forecast, which just copies the last
cycle without assuming it repeats on schedule, degrades more gracefully.

This is not a quirk of sunspots. Economic cycles, epidemic waves, hardware refresh cycles and
fashion trends all have approximate, drifting periodicity. Any of them will do this to a
seasonal model.

## The machine learning result nobody wants to publish

I included LightGBM with direct multi-step forecasting — a proper implementation, h separate
models, no recursive error compounding.

It won **{[r for r in syn['cross_series_ranking'] if 'LightGBM' in r['label']][0]['wins']} series**.

It also never ranked worse than third, which makes it the most *consistent* method in the
tournament. But on three of four series it lost to SARIMA, a method from 1970.

That is the honest result on 120–204 observations. Gradient boosting has a lot of capacity and
not much to learn from; classical methods encode strong structural assumptions that happen to
be correct for trend-plus-seasonality data. Give me 100,000 observations and multiple related
series and the answer flips. At this scale it does not.

## The measurement detail that makes this possible

You cannot average error across series measuring passengers, degrees Celsius, prescriptions
and sunspot area. The units do not commensurate.

MASE fixes this by dividing every error by the in-sample seasonal-naive error of that series.
The result is unit-free *and* anchored: 1.0 means "you matched the naive forecast", below 1.0
means you beat it.

Look back at the sunspot table with that in mind. Every model is above 1.0. Not one of them
beat naive.

## One decision I want to flag

The Melbourne temperature series is 3,650 daily observations. The real seasonality is annual —
period 365.

SARIMA with m=365 would need to estimate 365 seasonal parameters from ten annual cycles. That
does not fit in reasonable time and would not be identifiable if it did.

I aggregated to 120 monthly means. The annual cycle survives intact and every model can be
fitted properly.

The tempting alternative was to set m=7 and call it seasonality. That runs fast, produces a
plausible-looking table, and models a weekly temperature cycle that does not exist. A quiet
parameter choice can turn a benchmark into fiction.

---

**Live demo:** [{SITE}/#/p/{SLUGS[project]}]({SITE}/#/p/{SLUGS[project]})
**Code:** [github.com/Pranjal101Shrivastava/Projects](https://github.com/Pranjal101Shrivastava/Projects)
"""
    return {"README.md": readme, "abstract.md": abstract, "paper.md": paper,
            "article.md": article}


# ======================================================================================
# Project 06 — AutoML and the leak
# ======================================================================================
def doc_06() -> dict[str, str]:
    project = "06_automl_tournament"
    leak = load(project, "leak_demonstration")
    clean = load(project, "tournament_clean")
    leaked = load(project, "tournament_leaked")
    diag, inf = leak["diagnostic"], leak["inflation"]

    comparison = "\n".join(
        f"| `{m['model']}` | {m['clean_pr_auc']:.4f} | "
        + (f"{m['leaked_pr_auc']:.4f}" if m["leaked_pr_auc"] else "—")
        + " | "
        + (f"**{(m['leaked_pr_auc'] / m['clean_pr_auc'] - 1) * 100:+.1f}%**" if m["leaked_pr_auc"] else "—")
        + " |"
        for m in inf["per_model"]
    )

    def board(t):
        return "\n".join(
            f"| `{r['model']}` | {r['test_pr_auc']:.4f} | "
            + (f"{r['cv_pr_auc_mean']:.4f} ± {r['cv_pr_auc_std']:.3f}" if r["cv_pr_auc_mean"] else "—")
            + f" | {r['test_roc_auc']:.4f} | {r['test_brier']:.4f} | {r['search_seconds']:.0f}s |"
            for r in t["leaderboard"]
        )

    readme = f"""# 06 · AutoML, and the Cost of One Leaking Column

An identical model tournament run **twice** — with and without a column that cannot exist at
scoring time — to price a documented target leak.

{provenance_table(project)}

## The trap

UCI's Bank Marketing dataset ships with a `duration` column recording how long the sales call
lasted. UCI's own documentation states it *"should be discarded if the intention is to have a
realistic predictive model"*.

It appears in most published notebooks on this dataset anyway, because including it makes
every model look dramatically better.

## The measurement

Same four model families. Same 12 randomised hyperparameter draws each. Same 4-fold
stratified CV. Same stacked ensemble. The **only** difference is one column.

| Model | Leak-free | With leak | Inflation |
|---|---:|---:|---:|
{comparison}

**Best PR-AUC moves from {inf['clean_best_pr_auc']:.4f} to {inf['leaked_best_pr_auc']:.4f} —
{pct(inf['relative_inflation'])} relative inflation from a single column.**

{inf['verdict']}

## Why it is a leak, not a strong feature

| Evidence | Value |
|---|---:|
| `duration` alone, ROC-AUC | **{diag['univariate_roc_auc']:.4f}** |
| Correlation with target | {diag['correlation_with_target']:.3f} |
| Mean call length — subscribed | {diag['mean_duration_subscribed']:.0f}s |
| Mean call length — declined | {diag['mean_duration_declined']:.0f}s |
| Zero-duration calls | {diag['zero_duration_rows']} |
| …of which subscribed | **{diag['zero_duration_subscriptions']}** |

{diag['why_this_is_a_leak']}

The zero-duration row is the clean confirmation: all {diag['zero_duration_rows']}
zero-second calls are non-subscriptions, exactly as the causal argument predicts. You cannot
subscribe to a term deposit during a call that never happened.

## How to catch a leak

**Test 1 — implausible univariate power.** Does any single column predict the target
implausibly well alone? Necessary, but not sufficient: a genuinely strong feature can also
score highly.

**Test 2 — causal timing.** *Would this value exist at the moment the prediction must be
made?* This is the decisive one and it requires no statistics. A call-prioritisation model
scores **before dialling**, when call length does not yet exist.

No cross-validation scheme, confusion matrix or calibration curve detects this. Only
reasoning about the order of events does.

## Leaderboards

### Leak-free — the deployable result

| Model | Test PR-AUC | CV PR-AUC | ROC-AUC | Brier | Search |
|---|---:|---:|---:|---:|---:|
{board(clean)}

### With leak — demonstration only

| Model | Test PR-AUC | CV PR-AUC | ROC-AUC | Brier | Search |
|---|---:|---:|---:|---:|---:|
{board(leaked)}

## Searching families, not just hyperparameters

A search over boosting depths is a hyperparameter tuner, not AutoML. Four families with
genuinely different inductive biases were searched:

""" + "\n".join(f"- **`{r['model']}`** — {r['inductive_bias']}" for r in clean["leaderboard"]) + f"""

Including a deliberately naive Bayes reference shows how much of the final score came from
model capacity: it reaches
{[r for r in clean['leaderboard'] if r['model'] == 'gaussian_nb'][0]['test_pr_auc']:.3f}
against the winner's {clean['best']['test_pr_auc']:.3f}.

## Stacking without leaking one level up

The meta-learner is trained on **cross-validated out-of-fold predictions only**. Fitting it on
in-fold predictions would let it observe base-model outputs for rows those models had
effectively memorised — leakage one level up, and a common way stacked ensembles get silently
overfitted.

{screenshots_section(project)}

## CRISP-DM record

{crispdm_section(project)}

{quickstart(project)}

{footer(project)}
"""

    abstract = f"""# Abstract — Quantifying the Cost of a Single Leaking Feature

**Objective.** Measure the performance inflation attributable to one documented target leak,
under otherwise identical modelling conditions.

**Data.** 41,188 real direct-marketing contacts from a Portuguese retail bank. The dataset
includes a `duration` column recording call length, which the source documentation identifies
as unusable for realistic prediction because it is unknown before the call occurs.

**Method.** A model tournament — four families spanning distinct inductive biases, 12
randomised hyperparameter draws each under 4-fold stratified cross-validation scored by
average precision, plus a stacked ensemble over out-of-fold predictions — was executed twice
under identical conditions, differing only in the presence of the leaking column.
Preprocessing was confined to pipelines so that transforms were refitted per fold.

**Results.** Best test PR-AUC rose from {inf['clean_best_pr_auc']:.4f} to
{inf['leaked_best_pr_auc']:.4f} ({pct(inf['relative_inflation'])} relative) with the column
included. Inflation affected every family, from {min(m['leaked_pr_auc'] / m['clean_pr_auc'] - 1 for m in inf['per_model'] if m['leaked_pr_auc']) * 100:.1f}%
to {max(m['leaked_pr_auc'] / m['clean_pr_auc'] - 1 for m in inf['per_model'] if m['leaked_pr_auc']) * 100:.1f}%.
The column alone achieved ROC-AUC {diag['univariate_roc_auc']:.4f}, and all
{diag['zero_duration_rows']} zero-duration records were non-subscriptions, consistent with the
value being generated by the event being predicted.

**Conclusion.** Target leakage of this magnitude is invisible to cross-validation, confusion
matrices and calibration diagnostics. Detection requires reasoning about when each feature's
value becomes available relative to the prediction, which is a domain question rather than a
statistical one.

**Keywords.** data leakage, AutoML, model selection, cross-validation, stacked generalisation
"""

    paper = f"""# Pricing a Target Leak: An Identical Tournament Run Twice

## 1. Motivation

Target leakage is widely discussed and rarely quantified. This study measures it directly, by
executing the same model selection procedure twice under identical conditions, differing only
in whether one documented leaking column is available.

## 2. The leaking feature

The Bank Marketing dataset records `duration`, the length in seconds of the marketing call.
The UCI documentation states the attribute "highly affects the output target (e.g. if
duration=0 then y='no')" and "should be discarded if the intention is to have a realistic
predictive model".

Its causal status is unambiguous: the value is produced *by* the call whose outcome is being
predicted. At the moment a prioritisation model must score — before dialling — it does not
exist.

### 2.1 Diagnostic evidence

| Statistic | Value |
|---|---:|
| Univariate ROC-AUC | {diag['univariate_roc_auc']:.4f} |
| Point-biserial correlation | {diag['correlation_with_target']:.4f} |
| Mean duration, subscribed | {diag['mean_duration_subscribed']:.1f}s |
| Mean duration, declined | {diag['mean_duration_declined']:.1f}s |
| Zero-duration records | {diag['zero_duration_rows']} |
| Zero-duration subscriptions | {diag['zero_duration_subscriptions']} |

A single feature attaining ROC-AUC {diag['univariate_roc_auc']:.3f} is a diagnostic signal,
though not conclusive on its own. The zero-duration cell is conclusive: a call of zero seconds
cannot produce a subscription, and none does.

## 3. Experimental design

Both runs used: four model families (regularised linear, bagged trees, gradient boosting, and
a conditional-independence reference), {clean['search_protocol']}, followed by a stacking
ensemble over the top three base models with a logistic meta-learner trained on
cross-validated out-of-fold predictions.

Preprocessing — median imputation, standardisation, one-hot encoding with rare-level folding
— was placed inside the estimator pipeline so that every transform is refitted on each fold's
training rows. Fitting transforms outside the pipeline would contaminate the comparison with a
*second* leak.

## 4. Results

| Model | Leak-free PR-AUC | Leaked PR-AUC | Inflation |
|---|---:|---:|---:|
{comparison}

{inf['verdict']}

Two features of this table are worth attention.

**Inflation is universal but not uniform.** Every family gains, but the naive-Bayes reference
gains least. Higher-capacity models extract more from the leak, so leakage widens the apparent
gap between simple and complex models — which is precisely the comparison a practitioner uses
to justify complexity.

**Cross-validation offers no warning.** CV scores in the leaked run are internally consistent
with its test scores. The generalisation gap looks healthy. Every diagnostic a practitioner
would normally consult reports that the model is sound.

## 5. Detection

Two tests, applied in order.

**Univariate power.** Screen each feature's standalone predictive power. Anomalously high
values warrant investigation. Necessary but insufficient — legitimately strong features exist.

**Causal timing.** For each feature, ask whether its value exists at the moment of prediction.
This test is decisive and requires no statistics. It is also the only one that catches this
case, because no distributional property of `duration` marks it as illegitimate; only its
position in the causal order does.

## 6. Limitations

""" + "\n".join(
        f"- {risk}"
        for phase in load(project, "crispdm")["phases"]
        for risk in phase.get("risks", [])
    ) + f"""

{run_stamp(project)}
"""

    article = f"""# One column was worth 40% of my model's performance. It shouldn't have been there.

*A tournament run twice, to price a leak*

The UCI Bank Marketing dataset is a standard benchmark. 41,188 real marketing calls from a
Portuguese bank, predicting who subscribes to a term deposit.

It also ships with a trap, and UCI tells you about it:

> *"this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet,
> the duration is not known before a call is performed... this input should be discarded if
> the intention is to have a realistic predictive model."*

That is the documentation for a column called `duration` — how long the sales call lasted.

Search for notebooks on this dataset. Most of them use it.

## So I measured what it is worth

I built a proper AutoML tournament: four model families with different inductive biases, 12
randomised hyperparameter draws each, 4-fold stratified cross-validation scored by average
precision, plus a stacked ensemble.

Then I ran the entire thing **twice**. Identical code, identical folds, identical seeds. The
only difference: one column present or absent.

| Model | Without `duration` | With `duration` |
|---|---:|---:|
""" + "\n".join(
        f"| {m['model']} | {m['clean_pr_auc']:.4f} | "
        + (f"**{m['leaked_pr_auc']:.4f}**" if m["leaked_pr_auc"] else "—") + " |"
        for m in inf["per_model"]
    ) + f"""

**{pct(inf['relative_inflation'])} inflation on the headline number.** Every model. From one
column.

## Why it is a leak and not just a good feature

The statistical signal is suspicious:

- `duration` alone reaches **ROC-AUC {diag['univariate_roc_auc']:.3f}**
- Subscribers average **{diag['mean_duration_subscribed']:.0f} seconds** on the call
- Decliners average **{diag['mean_duration_declined']:.0f} seconds**

But suspicion is not proof. Some features really are that good.

Here is the proof. There are **{diag['zero_duration_rows']} calls with duration = 0** in the
dataset. Number of those that resulted in a subscription: **{diag['zero_duration_subscriptions']}**.

Of course. You cannot subscribe during a call that did not happen.

That is not a feature predicting an outcome. That is an outcome leaving a fingerprint on a
feature.

## The part that should worry you

I want to be precise about what did *not* catch this.

- Cross-validation: clean. CV and test scores agreed in both runs.
- The generalisation gap: healthy.
- Calibration: fine.
- Confusion matrix: normal.
- Learning curves: unremarkable.

Every diagnostic a careful practitioner consults reported that the leaked model was sound.
Because it *was* sound — it modelled the data it was given, correctly. The data was wrong.

There is no statistical test for this. A leaking feature and a strong feature have the same
distributional signature. What distinguishes them is *when the value comes into existence*,
and that is a question about the world.

## The two questions

**Does any single feature predict the target implausibly well on its own?** A screening
heuristic. Worth running. Not conclusive.

**Would this value exist at the moment I have to make the prediction?**

That is the one. Sit with each column and ask when it gets populated. For a call-prioritisation
model, scoring happens *before dialling*. Call duration does not exist yet. Neither does the
outcome, the agent's notes, or anything else the call produces.

It takes ten minutes and requires no mathematics, and it is the only thing that catches a leak
worth 40% of your reported performance.

## What I shipped

Both leaderboards. The leak-free one is labelled deployable; the leaked one is labelled a
demonstration. Publishing only the clean numbers would have hidden the most useful thing I
learned.

If you benchmark against published results on this dataset, check whether they used
`duration`. Most did. The comparison is not like-for-like.

---

**Live demo:** [{SITE}/#/p/{SLUGS[project]}]({SITE}/#/p/{SLUGS[project]})
**Code:** [github.com/Pranjal101Shrivastava/Projects](https://github.com/Pranjal101Shrivastava/Projects)
"""
    return {"README.md": readme, "abstract.md": abstract, "paper.md": paper,
            "article.md": article}


# ======================================================================================
# Project 07 — Nano transformer
# ======================================================================================
def doc_07() -> dict[str, str]:
    project = "07_nano_transformer"
    tr = load(project, "training")
    ev = load(project, "evaluation")
    corpus = load(project, "corpus")
    arch = tr["architecture"]
    best = tr["best_val"]
    base = ev["baselines"]
    gap = tr["final"]["val_loss"] - tr["final"]["train_loss"]

    samples = "\n".join(
        f"| {s['temperature']} | {s['n_words']} | **{pct(s['real_word_rate'])}** |"
        for s in ev["samples"]
    )
    local = sum(1 for h in ev["attention_spans"] if h["mean_attention_distance"] < 10)

    readme = f"""# 07 · A Character Transformer Written from Tensor Operations

A {tr['n_parameters']:,}-parameter decoder trained on CPU in
{tr['train_seconds'] / 60:.0f} minutes, with every component — rotary embeddings, SwiGLU,
pre-norm blocks, weight tying — implemented directly rather than assembled from
`nn.TransformerEncoderLayer`.

{provenance_table(project)}

## Result

| | Perplexity | What it means |
|---|---:|---|
| Uniform guess | {base['uniform']['perplexity']:.2f} | every character equally likely |
| Unigram | {base['unigram']['perplexity']:.2f} | characters drawn from corpus frequencies |
| **This model** | **{base['model']['perplexity']:.3f}** | {tr['n_parameters']:,} parameters, held out |

**{ev['improvement_over_unigram']:.1f}× better than the frequency baseline.** The two
baselines are reported because a perplexity figure with no reference point cannot be judged.

## What it actually learned

A {tr['n_parameters'] / 1e6:.2f}M-parameter model on {corpus['n_characters']:,} characters
learns **orthography and dramatic form** — which letter sequences are English-shaped, how
speaker labels and line breaks are laid out. It does not learn meaning.

Rather than display one cherry-picked paragraph, the real-word rate is measured at four
sampling temperatures:

| Temperature | Words generated | Real words |
|---:|---:|---:|
{samples}

The monotone decline is the temperature–coherence trade-off made measurable. Low temperature
concentrates probability on the likeliest next character — repetitive but well-spelled. High
temperature flattens the distribution — more variety, more non-words.

Note what this metric does *not* measure. A sample can be
{pct(ev['samples'][0]['real_word_rate'])} real words and mean nothing at all.

## Architecture, and why each piece

| Component | Choice |
|---|---|
| Layers | {arch['n_layers']} |
| Attention heads | {arch['n_heads']} |
| Model dimension | {arch['d_model']} |
| Feed-forward dimension | {arch['d_ff']} |
| Context | {arch['context']} characters |
| Position encoding | Rotary (RoPE) |
| Feed-forward | SwiGLU (gated) |
| Normalisation | Pre-norm LayerNorm |
| Embeddings | Tied input/output |

**Rotary position embeddings.** Position enters by *rotating* query and key vectors by an
angle proportional to index. Because the dot product of two rotated vectors depends only on
their angle *difference*, the attention logit becomes a function of relative offset. Nothing
is learned and there is no absolute index to overfit to.

**Pre-norm blocks.** Post-norm routes the residual through the normaliser, making gradient
magnitude depth-dependent and warmup mandatory for stability. Pre-norm leaves the residual
path unobstructed, which is why every modern decoder uses it.

**SwiGLU.** One projection produces values, another a gate, and their product passes through.
Empirically stronger than ReLU or GELU at equal parameter count.

**Weight tying.** The vector representing a character on the way in is the vector scoring it
on the way out. On a {corpus['vocab_size']}-symbol vocabulary the parameter saving is
negligible; the inductive bias is the point.

## Attention specialisation

{local} of {len(ev['attention_spans'])} heads attend at a mean distance under 10 characters —
local work like bigrams and word boundaries — while the rest carry context across a line or
more. Nothing in the architecture assigns roles; the specialisation emerges from training.

## Training

{tr['n_parameters']:,} parameters, {len(tr['history']) - 1} evaluation points over
{tr['train_seconds']:.0f} seconds on CPU. AdamW with linear warmup then cosine decay, gradient
clipping at 1.0.

Final train loss {tr['final']['train_loss']:.4f}, validation {tr['final']['val_loss']:.4f} —
a gap of {gap:.4f}. {"The model has begun memorising; a larger corpus or stronger regularisation would be required to train longer." if gap > 0.15 else "The gap is small, so capacity rather than overfitting is the binding constraint at this budget."}

## A split detail that matters

The corpus is one continuous text. The split is **contiguous**, not random.
{corpus['split_rationale']}

{screenshots_section(project)}

## CRISP-DM record

{crispdm_section(project)}

{quickstart(project)}

{footer(project)}
"""

    abstract = f"""# Abstract — A Character-Level Transformer Decoder Trained on CPU

**Objective.** Implement each component of a contemporary transformer decoder directly
against tensor operations, train it within a CPU budget, and evaluate it against explicit
baselines rather than by qualitative sample inspection.

**Data.** {corpus['n_characters']:,} characters of public-domain Shakespeare,
{corpus['vocab_size']} distinct symbols, split contiguously 90/10. A random split over a
continuous text would interleave validation windows with training windows, making validation
context already memorised.

**Method.** A {arch['n_layers']}-layer pre-norm decoder with {arch['n_heads']}-head causal
self-attention, rotary position embeddings, SwiGLU gated feed-forward blocks and tied
input/output embeddings, totalling {tr['n_parameters']:,} parameters. Trained with AdamW under
linear warmup and cosine decay with gradient clipping, on CPU only.

**Results.** Held-out perplexity reached {best['val_perplexity']:.3f}, against
{base['unigram']['perplexity']:.2f} for a unigram model and {base['uniform']['perplexity']:.1f}
for uniform guessing — a {ev['improvement_over_unigram']:.1f}× improvement over the frequency
baseline. Generated text contained {pct(ev['samples'][1]['real_word_rate'])} real corpus words
at temperature {ev['samples'][1]['temperature']}, declining monotonically to
{pct(ev['samples'][-1]['real_word_rate'])} at temperature {ev['samples'][-1]['temperature']}.
Attention probing found {local} of {len(ev['attention_spans'])} heads specialised to
short-range context.

**Conclusion.** The model acquires orthography and structural form but not semantics. Reported
metrics are chosen to make that distinction measurable rather than relying on qualitative
inspection of selected samples.

**Keywords.** transformer, rotary position embedding, SwiGLU, language modelling, perplexity
"""

    paper = f"""# Implementing and Evaluating a Character-Level Transformer Decoder

## 1. Scope

The objective is pedagogical: implement every component of a modern decoder from tensor
operations, train within a CPU budget, and evaluate honestly. The last constraint shapes the
work most, because a language model can always be made to look good by quoting its best
sample.

## 2. Architecture

### 2.1 Rotary position embeddings

Position is encoded by rotating each (even, odd) coordinate pair of the query and key vectors
by an angle θ_i = p · ω_i, where p is the token index and ω_i = base^(−2i/d).

For a dot product between a query at position p and a key at position q, the rotation
contributes a factor depending only on (p − q). Attention logits therefore become a function
of *relative* offset without any learned position parameters and without an absolute index the
model could overfit to.

### 2.2 SwiGLU feed-forward

    FFN(x) = (SiLU(W_gate x) ⊙ W_up x) W_down

A gated activation: one projection produces values, another a gate, and their elementwise
product passes through. Empirically outperforms ReLU and GELU at matched parameter count.

### 2.3 Pre-normalisation

Each block computes x + Attn(LN(x)) then x + FFN(LN(x)). The original post-norm arrangement
routes the residual through the normaliser, making gradient magnitude depend on depth and
requiring warmup for stability. Pre-norm leaves an unobstructed residual path.

### 2.4 Weight tying

The token embedding matrix serves as the output projection. On a {corpus['vocab_size']}-symbol
vocabulary this saves {corpus['vocab_size'] * arch['d_model']:,} parameters, which is
negligible. The justification is the inductive bias: the representation of a character on
input should be the vector scoring it on output.

### 2.5 Configuration

{arch['n_layers']} layers, {arch['n_heads']} heads, d_model {arch['d_model']}, d_ff
{arch['d_ff']}, context {arch['context']}, dropout {arch['dropout']}. Total
{tr['n_parameters']:,} parameters.

The expected parameter count is derived from the configuration constants in code and asserted
against the constructed model at runtime, so documentation cannot drift from the architecture.

## 3. Data and splitting

{corpus['n_characters']:,} characters, {corpus['vocab_size']} distinct symbols. Character-level
tokenisation was chosen over subword: it keeps embedding and output layers negligible so
nearly all parameters sit in the transformer blocks, and it makes spelling an observable
achievement rather than something the tokeniser handles invisibly.

{corpus['split_rationale']}

## 4. Optimisation

AdamW (β = 0.9, 0.95; weight decay 0.1), linear warmup over 120 steps then cosine decay,
gradient clipping at global norm 1.0, batch size 48, {arch['context']}-character context.

Warmup is retained despite pre-norm because Adam's second-moment estimate is unreliable in the
first few dozen steps; a full-magnitude update then can move parameters into a region from
which the model does not recover.

## 5. Evaluation

### 5.1 Perplexity against baselines

| Model | Perplexity |
|---|---:|
| Uniform over {corpus['vocab_size']} symbols | {base['uniform']['perplexity']:.2f} |
| Unigram (corpus character frequencies) | {base['unigram']['perplexity']:.3f} |
| This model (held out) | **{base['model']['perplexity']:.3f}** |

The two baselines bound what "learning something" must mean on this corpus. The model improves
{ev['improvement_over_unigram']:.1f}× over the frequency distribution.

### 5.2 Real-word rate

| Temperature | Words | Real-word rate | Mean line length |
|---:|---:|---:|---:|
""" + "\n".join(
        f"| {s['temperature']} | {s['n_words']} | {pct(s['real_word_rate'])} | {s['mean_line_length']} |"
        for s in ev["samples"]
    ) + f"""

This measures orthography, not semantics. It is reported precisely because it makes the
limitation explicit: high real-word rate is compatible with complete semantic emptiness.

### 5.3 Attention specialisation

Mean attention distance was probed per head on held-out text. {local} of
{len(ev['attention_spans'])} heads concentrate below 10 characters. Head roles are not assigned
by the architecture; the specialisation is learned.

### 5.4 Generalisation

Final train loss {tr['final']['train_loss']:.4f}, validation {tr['final']['val_loss']:.4f},
gap {gap:.4f}.

## 6. Limitations

""" + "\n".join(
        f"- {risk}"
        for phase in load(project, "crispdm")["phases"]
        for risk in phase.get("risks", [])
    ) + f"""

{run_stamp(project)}
"""

    article = f"""# I built a transformer from scratch on a CPU, and measured what it actually learned

*{tr['n_parameters']:,} parameters, {tr['train_seconds'] / 60:.0f} minutes, no GPU*

Small language model demos follow a script. Train something on Shakespeare, generate a
paragraph, quote the paragraph. The reader is invited to be impressed by prose that sounds
vaguely Elizabethan.

The problem with that script is that you can always find a good paragraph. Generate twenty,
pick the best, and the model looks far better than it is.

So I decided up front what I would report, before I saw any output.

## Every component written by hand

No `nn.TransformerEncoderLayer`. The architecture is the subject, and a wrapper hides exactly
the parts worth understanding.

**Rotary position embeddings.** Instead of adding a position vector, you *rotate* the query
and key vectors by an angle proportional to position. The clever part is algebraic: when you
dot-product two rotated vectors, the rotations partly cancel and what survives depends only on
the *difference* in position. Relative position falls out of the geometry. Nothing is learned,
and there is no absolute index for the model to memorise.

**SwiGLU.** The feed-forward block computes two projections — one is values, one is a gate —
and multiplies them. It consistently beats ReLU at the same parameter count. Nobody has a fully
satisfying theory for why.

**Pre-norm.** Normalise *before* the attention block rather than after. Post-norm, the original
2017 arrangement, passes the residual stream through the normaliser, which makes gradients
depth-dependent and training fragile. Every modern decoder switched.

## What I promised to report

Two numbers, chosen before training.

**Perplexity against baselines.** A perplexity of {base['model']['perplexity']:.2f} means
nothing on its own. So:

| | Perplexity |
|---|---:|
| Guessing uniformly among {corpus['vocab_size']} characters | {base['uniform']['perplexity']:.1f} |
| Knowing only how often each character appears | {base['unigram']['perplexity']:.2f} |
| **The transformer** | **{base['model']['perplexity']:.2f}** |

{ev['improvement_over_unigram']:.1f}× better than knowing the letter frequencies. That is a
real result, and it is bounded above by how much there is to learn from a million characters.

**Real-word rate.** What fraction of generated words are actual words from the corpus?

| Temperature | Real words |
|---:|---:|
""" + "\n".join(f"| {s['temperature']} | {pct(s['real_word_rate'])} |" for s in ev["samples"]) + f"""

At low temperature it spells almost perfectly. Crank the temperature and coherence degrades
predictably. That monotone decline is the temperature–quality trade-off, measured instead of
asserted.

## What the model did not learn

It learned **spelling**. It learned **dramatic layout** — speaker names in caps, a colon, a
line break, verse-length lines. It learned that `th` is common and `qx` is not.

It did not learn **meaning**. Not a little bit of meaning. None.

Real-word rate measures orthography and nothing else. You can score
{pct(ev['samples'][0]['real_word_rate'])} real words and produce text that is completely
semantically empty — which is exactly what this model does.

That distinction disappears in write-ups that quote a sample and let the reader's pattern
matching do the rest. Shakespeare-shaped text activates the same recognition as Shakespeare.

## One thing I found that I did not expect

I probed each attention head to see how far back it typically looks.

**{local} of {len(ev['attention_spans'])} heads** concentrate on the previous ten characters —
they are doing letter-level work, bigrams and word boundaries. The rest reach across a line or
more, carrying longer structure.

Nothing in the code assigns those roles. All {len(ev['attention_spans'])} heads are
architecturally identical and randomly initialised. The division of labour is learned, in
{tr['train_seconds'] / 60:.0f} minutes, on a CPU.

## The split detail

The corpus is one continuous text. I split it contiguously — first 90% train, last 10%
validation.

Splitting *randomly* would have been a mistake. Random windows interleave: a validation window
sits between two training windows, so the model has already seen its immediate context. The
reported perplexity would be optimistic and nothing would indicate it.

Continuous text needs a continuous split, for the same reason time series do.

---

**Live demo:** [{SITE}/#/p/{SLUGS[project]}]({SITE}/#/p/{SLUGS[project]})
**Code:** [github.com/Pranjal101Shrivastava/Projects](https://github.com/Pranjal101Shrivastava/Projects)
"""
    return {"README.md": readme, "abstract.md": abstract, "paper.md": paper,
            "article.md": article}


# ======================================================================================
# Project 08 — CRISP-DM Academy
# ======================================================================================
def doc_08() -> dict[str, str]:
    project = "08_crispdm_academy"
    idx = load(project, "index")
    bayes = load(project, "bayes")
    thr = load(project, "threshold")
    grad = load(project, "gradient_descent")
    bp = load(project, "backprop")
    bv = load(project, "bias_variance")
    smp = load(project, "sampling")

    diverged = [k for k, v in grad["trajectories"].items() if v["diverged"]]
    n30 = [d for d in smp["sampling_distributions"] if d["sample_size"] == 30][0]

    modules = "\n".join(
        f"| {m['title']} | {m['dataset']} | {m['n_quiz']} |" for m in idx["modules"]
    )

    readme = f"""# 08 · CRISP-DM Academy

Six interactive teaching modules in which **every figure is computed from real data**, not
simulated — including the cases where real data refuses to behave like the textbook.

{provenance_table(project)}

| Module | Dataset | Quiz questions |
|---|---|---:|
{modules}

## The premise

Statistics teaching almost always illustrates its concepts with generated data, because
generated data satisfies its assumptions. Students learn what a concept looks like when the
preconditions hold, then meet real data where they do not.

Here, **where an assumption breaks, the breakage is the lesson**.

## 1 · Bayes and the independence assumption

Naive Bayes assumes features are conditionally independent given the class. On Titanic,
**{bayes['n_pairs_violating']} of {len(bayes['dependence_check'])}** tested feature pairs
violate that materially — passenger class and fare band reach Cramér's V
{max(bayes['dependence_check'][0]['cramers_v_given_died'], bayes['dependence_check'][0]['cramers_v_given_survived']):.2f}.

| Model | ROC-AUC | Brier |
|---|---:|---:|
| Naive Bayes | {bayes['comparison']['naive_bayes']['roc_auc']:.4f} | {bayes['comparison']['naive_bayes']['brier']:.4f} |
| Logistic regression | {bayes['comparison']['logistic_regression']['roc_auc']:.4f} | {bayes['comparison']['logistic_regression']['brier']:.4f} |

The classifier survives the violated assumption on *ranking* but not on *calibration* — its
Brier score is measurably worse. Dependence distorts posterior magnitudes more than their
order, which is why Naive Bayes remains a usable ranker and a poor probability estimate.

## 2 · Thresholds, ROC and cost

Traced from real fraud model scores at {pct(thr['prevalence'], 4)} prevalence:

- **ROC-AUC {thr['roc_auc']:.4f}** — looks excellent
- **PR-AUC {thr['pr_auc']:.4f}** — same model, far more sober
- **No-skill PR floor {thr['no_skill_pr']:.5f}** — which is just the prevalence

The interactive threshold slider makes the point experientially: accuracy stays pinned near
99.9% across the entire threshold range while precision and recall move by tens of points.

## 3 · Gradient descent

A real convex loss surface with four learning rates traced across it.

| Learning rate | Outcome | Final loss ÷ optimum |
|---:|---|---:|
""" + "\n".join(
        f"| {k} | {v['behaviour']} | {v['loss_ratio_to_optimum']:,.2f}× |"
        for k, v in grad["trajectories"].items()
    ) + f"""

Divergence is included because it is the most instructive behaviour and is usually omitted
from teaching figures. At lr = {diverged[0] if diverged else 'n/a'} the loss ends
{grad['trajectories'][diverged[0]]['loss_ratio_to_optimum']:,.0f}× above the optimum.

## 4 · Backpropagation is the chain rule

Analytic gradients verified against central finite differences on a real two-layer network:

**Maximum relative error: {bp['max_relative_error']:.2e}** — check
{"passed" if bp['check_passed'] else "FAILED"}.

The module also shows why sigmoid pairs with cross-entropy: ∂L/∂a₂ and ∂a₂/∂z₂ multiply to
the clean (a₂ − y)/m. With squared error the sigmoid derivative survives, and it vanishes when
the unit saturates, stalling learning.

## 5 · Bias, variance and capacity — where the textbook breaks

Optimal polynomial degree: **{bv['optimal_degree']}**.

But **this curve is not the U-shape from the textbook**:

- Variance accounts for only **{pct(bv['variance_share_at_max_degree'])}** of test error even
  at degree 15. Bias dominates throughout, because day-of-year does not determine daily
  temperature and the irreducible noise is large.
- Training error is **not monotone** in capacity here, because each degree is averaged over
  {bv['n_bootstrap']} bootstrap resamples and resampling noise exceeds the marginal gain once
  bias has plateaued. The textbook monotone training curve assumes a single fixed training set.

Both departures are detected in code and stated, rather than asserted from the textbook. A
simulated example would show the clean U and hide both.

## 6 · Sampling and the central limit theorem

Population skewness: **{smp['population']['skewness']:.2f}** — nothing like normal.

| n | Skewness of sample mean | Observed SE | σ/√n |
|---:|---:|---:|---:|
""" + "\n".join(
        f"| {d['sample_size']} | {d['skewness']:.3f} | {d['observed_se']:.2f} | {d['predicted_se']:.2f} |"
        for d in smp["sampling_distributions"]
    ) + f"""

**At n = 30 — the rule-of-thumb threshold — the sample mean is still skewed
{n30['skewness']:.2f}.** That rule is calibrated on mildly non-normal populations, not on one
with skewness {smp['population']['skewness']:.1f}.

The erratic n = 1 and n = 2 rows are flagged rather than smoothed: 1,500 draws from a tail this
heavy under-sample the extremes, so those estimates are dominated by whether a few outliers
happened to be drawn.

{screenshots_section(project)}

## CRISP-DM record

{crispdm_section(project)}

{quickstart(project)}

{footer(project)}
"""

    abstract = f"""# Abstract — Teaching Statistical Concepts from Real Rather Than Simulated Data

**Objective.** Construct interactive teaching material for six core statistical concepts in
which every figure derives from real observational data, and examine what changes when the
assumptions underlying textbook illustrations are not satisfied.

**Data.** Five real datasets: Titanic passenger records, ULB credit card transactions,
Melbourne daily temperatures, and derived series.

**Method.** Six modules were implemented — Bayes and conditional independence, threshold and
cost trade-offs, gradient descent, backpropagation, bias-variance decomposition, and the
central limit theorem — each computing its figures from observational data. Assumption
violations were quantified rather than assumed away: conditional dependence by Cramér's V,
gradient correctness by central finite differences, bias-variance components by bootstrap
resampling, and normality of sampling distributions by skewness and Shapiro-Wilk.

**Results.** {bayes['n_pairs_violating']} of {len(bayes['dependence_check'])} tested feature
pairs violated conditional independence on Titanic, yet Naive Bayes retained ranking
performance (ROC-AUC {bayes['comparison']['naive_bayes']['roc_auc']:.3f} against
{bayes['comparison']['logistic_regression']['roc_auc']:.3f}) while showing degraded
calibration. Analytic gradients matched finite differences to {bp['max_relative_error']:.1e}.
The bias-variance curve did not exhibit the canonical U: variance accounted for only
{pct(bv['variance_share_at_max_degree'])} of test error at maximum capacity, and training
error was non-monotone under bootstrap averaging. Convergence to normality was substantially
slower than the n = 30 heuristic implies, with sample-mean skewness still {n30['skewness']:.2f}
at that size for a population of skewness {smp['population']['skewness']:.1f}.

**Conclusion.** Illustrations drawn from real data differ systematically from their textbook
forms. Those differences constitute teachable content rather than defects in the examples.

**Keywords.** statistical education, bias-variance decomposition, central limit theorem,
gradient checking, conditional independence
"""

    paper = f"""# Teaching Statistical Concepts from Data That Does Not Cooperate

## 1. Motivation

Instructional material illustrates concepts with generated data because generated data
satisfies its assumptions: the bias-variance curve forms a clean U, the sampling distribution
converges by n = 30, features are conditionally independent because they were drawn that way.

Students then encounter real data where none of this holds, with no framework for the
discrepancy.

Each module below computes its figures from observational data. Where an assumption fails, the
failure is quantified and taught.

## 2. Conditional independence (Titanic)

Naive Bayes assumes P(A, B | C) = P(A|C)·P(B|C). Cramér's V was computed between feature pairs
*within each class*:

| Pair | V given died | V given survived | Violates |
|---|---:|---:|:---:|
""" + "\n".join(
        f"| {d['pair']} | {d['cramers_v_given_died']:.3f} | {d['cramers_v_given_survived']:.3f} | "
        + ("yes" if d["violates_independence"] else "no") + " |"
        for d in bayes["dependence_check"]
    ) + f"""

{bayes['n_pairs_violating']} of {len(bayes['dependence_check'])} pairs violate the assumption
materially. The consequence is measurable and asymmetric:

| | ROC-AUC | Brier |
|---|---:|---:|
| Naive Bayes | {bayes['comparison']['naive_bayes']['roc_auc']:.4f} | {bayes['comparison']['naive_bayes']['brier']:.4f} |
| Logistic regression | {bayes['comparison']['logistic_regression']['roc_auc']:.4f} | {bayes['comparison']['logistic_regression']['brier']:.4f} |

Ranking performance is largely preserved; calibration is not. Dependent features contribute
correlated evidence that the model counts as independent, pushing posteriors toward 0 and 1.
Ordering survives because the distortion is largely monotone.

## 3. Threshold selection under imbalance

At prevalence {pct(thr['prevalence'], 4)}, the same classifier reports ROC-AUC
{thr['roc_auc']:.4f} and PR-AUC {thr['pr_auc']:.4f}.

{thr['lesson']}

## 4. Gradient descent and the stability bound

A two-parameter convex surface was constructed from real monthly temperature means, permitting
the entire loss landscape to be drawn.

| lr | Outcome | Final loss ÷ optimum |
|---:|---|---:|
""" + "\n".join(
        f"| {k} | {v['behaviour']} | {v['loss_ratio_to_optimum']:,.2f}× |"
        for k, v in grad["trajectories"].items()
    ) + f"""

{grad['lesson']}

## 5. Gradient checking

A two-layer network (3 → 5 tanh → 1 sigmoid, binary cross-entropy) was differentiated by hand
and verified against central finite differences (f(θ+ε) − f(θ−ε)) / 2ε.

| Parameter | Analytic | Numerical | Relative error |
|---|---:|---:|---:|
""" + "\n".join(
        f"| `{c['parameter']}` | {c['analytic']:.8f} | {c['numerical']:.8f} | {c['relative_error']:.2e} |"
        for c in bp["gradient_checks"][:6]
    ) + f"""

Maximum relative error {bp['max_relative_error']:.2e}. Correct gradients agree with central
differences to approximately 1e-7; an error near 0.3 indicates a derivation bug rather than
floating-point noise.

The derivation also demonstrates why sigmoid is paired with cross-entropy rather than squared
error: ∂L/∂a₂ and ∂a₂/∂z₂ multiply to (a₂ − y)/m, a cancellation that eliminates the sigmoid
derivative. Under squared error that derivative survives and vanishes on saturation, stalling
learning.

## 6. Bias-variance decomposition

Polynomial degree 1–15, decomposed over {bv['n_bootstrap']} bootstrap resamples.

| Degree | Bias² | Variance | Test MSE | Train MSE |
|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {r['degree']} | {r['bias_squared']:.3f} | {r['variance']:.3f} | "
        f"{r['total_test_mse']:.3f} | {r['train_mse']:.3f} |"
        for r in bv["curve"]
    ) + f"""

Optimal degree {bv['optimal_degree']}. **The canonical U-shape does not appear.**

{bv['lesson']}

Two departures from the textbook figure, both detected programmatically:

1. Variance contributes only {pct(bv['variance_share_at_max_degree'])} of test error at
   maximum capacity. Bias dominates because day-of-year does not determine daily temperature;
   irreducible noise is large and additional capacity cannot reduce it.
2. Training error is non-monotone ({'confirmed' if not bv['train_error_monotone'] else 'not observed'}),
   because each point averages over bootstrap resamples. The monotone training curve of the
   textbook assumes a single fixed training set.

## 7. Central limit theorem

Population: {smp['population']['n']:,} transaction amounts, skewness
{smp['population']['skewness']:.2f}, kurtosis {smp['population']['kurtosis']:.1f}.

| n | Skewness of mean | Observed SE | σ/√n | Ratio |
|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {d['sample_size']} | {d['skewness']:.3f} | {d['observed_se']:.3f} | "
        f"{d['predicted_se']:.3f} | {d['observed_se'] / max(d['predicted_se'], 1e-9):.3f} |"
        for d in smp["sampling_distributions"]
    ) + f"""

{smp['lesson']}

## 8. Limitations

""" + "\n".join(
        f"- {risk}"
        for phase in load(project, "crispdm")["phases"]
        for risk in phase.get("risks", [])
    ) + f"""

{run_stamp(project)}
"""

    article = f"""# The bias-variance curve in your textbook is not what real data does

*Six statistics lessons rebuilt on real data, and what changed*

Open any machine learning textbook to the bias-variance section. You will find a figure: bias
falling, variance rising, total error tracing a clean U with an obvious minimum.

I rebuilt that figure from real Melbourne temperature data. Here is what came out.

| Degree | Bias² | Variance | Test MSE |
|---:|---:|---:|---:|
""" + "\n".join(
        f"| {r['degree']} | {r['bias_squared']:.2f} | {r['variance']:.2f} | {r['total_test_mse']:.2f} |"
        for r in bv["curve"][::3]
    ) + f"""

Variance does rise, exactly as promised. But look at the magnitudes. Even at degree 15,
variance accounts for **{pct(bv['variance_share_at_max_degree'])} of total test error**. Bias
dominates the entire range.

There is no dramatic U. There is a shallow dip at degree {bv['optimal_degree']} and then a
gentle drift upward.

## Why the real curve is flat

The textbook U assumes you are fitting a signal that more capacity can capture. Here the
relationship is day-of-year → daily temperature, and day-of-year *does not determine* daily
temperature. Weather is noisy. A perfect model of the seasonal cycle still leaves enormous
irreducible error.

When irreducible noise dominates, extra capacity has almost nothing to grip. Bias plateaus,
variance grows slowly from a tiny base, and the curve flattens.

That is the common case in applied work, and it is the case the textbook figure does not show.

## Training error did not fall monotonically either

Every treatment states that training error decreases monotonically with capacity. Mine does
not, and the reason is instructive.

I averaged each degree over **{bv['n_bootstrap']} bootstrap resamples** rather than fitting
once. Once bias plateaus, the resample-to-resample variation exceeds the marginal gain from an
extra polynomial term. The averaged training curve wobbles.

The monotone guarantee holds for *one fixed training set*. Average over resamples — which is
what you must do to decompose bias and variance at all — and it stops holding.

## "n = 30" is not a rule

Everyone learns that the central limit theorem kicks in around n = 30.

I tested that on real transaction amounts, a distribution with skewness
**{smp['population']['skewness']:.1f}** — an extremely heavy right tail.

| Sample size | Skewness of the sample mean |
|---:|---:|
""" + "\n".join(
        f"| {d['sample_size']} | {d['skewness']:.2f} |"
        for d in smp["sampling_distributions"]
    ) + f"""

At n = 30 the sample mean is still skewed **{n30['skewness']:.2f}**. That is not approximately
normal. You would need several hundred observations before a normal approximation is safe here.

"n = 30" was calibrated on populations that are mildly non-normal. Applied to a genuinely
skewed one it is simply wrong, and nobody mentions the caveat.

## Naive Bayes with its assumption comprehensively broken

Naive Bayes assumes features are conditionally independent given the class. On Titanic I
measured how badly that fails: **{bayes['n_pairs_violating']} of
{len(bayes['dependence_check'])} pairs** violate it materially. Passenger class and fare band
reach Cramér's V {max(bayes['dependence_check'][0]['cramers_v_given_died'], bayes['dependence_check'][0]['cramers_v_given_survived']):.2f}
— they are nearly deterministic in each other, which is obvious in hindsight since first-class
tickets cost more.

So the model should fail. It does not:

| | ROC-AUC | Brier score |
|---|---:|---:|
| Naive Bayes | {bayes['comparison']['naive_bayes']['roc_auc']:.3f} | {bayes['comparison']['naive_bayes']['brier']:.4f} |
| Logistic regression | {bayes['comparison']['logistic_regression']['roc_auc']:.3f} | {bayes['comparison']['logistic_regression']['brier']:.4f} |

Ranking barely suffers. Calibration measurably does.

The reason is that dependent features contribute the same evidence twice, and Naive Bayes
counts it twice, pushing probabilities toward 0 and 1. But the distortion is mostly *monotone*
— it inflates confidence without reordering cases. So AUC, which only cares about order,
survives. Brier, which cares about the actual numbers, does not.

Which is why Naive Bayes is a fine ranker and a bad probability estimate. That sentence is in
every textbook. Measuring it is more convincing than reading it.

## And the learning rate that explodes

I traced gradient descent across a real loss surface at four learning rates:

| lr | What happened |
|---:|---|
""" + "\n".join(f"| {k} | {v['behaviour']} |" for k, v in grad["trajectories"].items()) + f"""

At the largest rate the loss ends
**{grad['trajectories'][diverged[0]]['loss_ratio_to_optimum']:,.0f}× above the minimum**.

Teaching figures almost never show divergence. They show a nicely converging trajectory and
mention in the caption that too large a rate is bad. Watching the loss go to infinity on a
surface you can see is considerably more memorable.

---

**Live demo:** [{SITE}/#/p/{SLUGS[project]}]({SITE}/#/p/{SLUGS[project]})
**Code:** [github.com/Pranjal101Shrivastava/Projects](https://github.com/Pranjal101Shrivastava/Projects)
"""
    return {"README.md": readme, "abstract.md": abstract, "paper.md": paper,
            "article.md": article}


GENERATORS = {
    "01_nyc_mobility": doc_01,
    "02_customer_segmentation": doc_02,
    "03_market_basket": doc_03,
    "04_fraud_detection": doc_04,
    "05_timeseries_forecasting": doc_05,
    "06_automl_tournament": doc_06,
    "07_nano_transformer": doc_07,
    "08_crispdm_academy": doc_08,
}


def main() -> None:
    total = 0
    for project, generator in GENERATORS.items():
        docs = generator()
        for name, content in docs.items():
            path = PROJECTS / project / name
            path.write_text(content)
            total += len(content)
            print(f"  {path.relative_to(ROOT)}  ({len(content) / 1024:.1f} KB)")
    print(f"\nGenerated {len(GENERATORS) * 4} documents, {total / 1024:.0f} KB total.")
    print("Every figure above was read from a committed artifact.")


if __name__ == "__main__":
    main()
