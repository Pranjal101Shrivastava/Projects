# Hourly Ride-Hail Demand Forecasting with Calibrated Uncertainty

*A CRISP-DM study on 4,534,327 real NYC TLC dispatch records*

## 1. Introduction

Positioning idle vehicles ahead of demand is the central operational lever in ride-hail.
The costs of getting it wrong are asymmetric and are paid hourly: under-supply in a zone
forfeits the fare and pushes the rider to a competitor, while over-supply pays drivers to
idle. A forecast is therefore useful at one-hour horizon and zone granularity, and it is
only worth operating if it improves on the heuristic an organisation would otherwise use
for free.

## 2. Data and its constraints

The source is the NYC Taxi & Limousine Commission's response to a Freedom of Information Law
request, covering 4,534,327 dispatched pickups between
2014-04-01 and 2014-09-30. Each record carries a
minute-resolution timestamp, coordinates rounded to four decimal places, and a dispatch base
identifier. There are no missing values.

**The schema constrains the research question.** With no duration or fare column, any model
of trip duration on this dataset must first invent its target. We therefore forecast demand,
which is directly countable from the records.

Two properties of the raw data required decisions rather than defaults.

**Duplicate records.** 82,581 rows (1.82%)
are exact duplicates. The reflex is to drop them. That is wrong here: at minute resolution
and ~11 m coordinate rounding, two genuinely distinct pickups dispatched from the same base
in the same minute on the same block are indistinguishable in this schema. Midtown at 18:00
dispatches far more than one vehicle per minute. Critically, duplicate share *rises with
demand* — the signature of collision under coarse resolution rather than of a data fault.
Dropping them would systematically understate demand exactly where it is highest.

**Out-of-region coordinates.** Latitudes span 39.66–42.12
and longitudes -74.93–-72.07, reaching from
Philadelphia to eastern Long Island. 24,505 rows
(0.54%) fall outside a Manhattan-centred bounding box and were
excluded, since spatial clustering across that span would merge locations 100 km apart.

## 3. Feature construction and leakage control

Events were aggregated to a zone × hour panel. Zones were obtained by K-means over pickup
coordinates (k = 12) rather than by a uniform grid: over this bounding box a
uniform grid is mostly water and parkland, so most cells would carry near-zero demand while
a handful of Manhattan cells would aggregate very different neighbourhoods.

The panel was reindexed onto a complete hourly grid before any lag was computed, so that a
missing hour cannot silently cause `lag_24h` to reach twenty-five hours back.

21 features were constructed, all strictly backward-looking:

- Lags use groupby(zone).shift(k): a row can only see its own zone's past.
- Rolling windows are shift(1) before .rolling(), so the window ends at t-1 and an hour never enters its own rolling statistic.
- The panel is reindexed onto a complete hourly grid before lagging, so a missing hour cannot let lag_24h silently reach 25 hours back.
- 2,016 warm-up rows with incomplete lag history are dropped rather than imputed — imputing them would fabricate history.

The rolling-window control deserves emphasis. Computing `df.rolling(w).mean()` directly on
the target includes the observation being predicted in its own predictor. The resulting fit
looks excellent and collapses in production, and no metric reveals the problem. Shifting by
one period before aggregating makes the window strictly [t−w, t−1].

## 4. Evaluation protocol

Partitioning was chronological with an embargo. The final 28 days
form the test set (8,064 zone-hours); the
168 hours immediately preceding the boundary were discarded
from training (40,596 zone-hours remain).

The embargo width equals the widest feature lag. Without it, training rows within one week of
the boundary share lag history with the first test rows, and the two partitions are not
independent.

## 5. Results

| Model | Family | MAE | RMSE | R² | Skill vs seasonal naive |
|---|---|---:|---:|---:|---:|
| LightGBM | gradient boosting | 15.65 | 30.19 | 0.957 | +39.1% |
| Ridge regression | linear | 18.00 | 30.49 | 0.957 | +30.0% |
| Seasonal naive (t − 168h) | baseline | 25.70 | 53.68 | 0.865 | +0.0% |
| Naive (t − 24h) | baseline | 35.36 | 71.60 | 0.760 | -37.6% |

Three observations.

**The baseline ladder matters more than the winner.** Reporting "LightGBM achieves MAE
15.65" is not a result. Reporting it against 25.70 for a
zero-parameter baseline is.

**Weekly periodicity dominates daily.** The t−24h naive forecast is *worse* than t−168h
(35.36 vs 25.70). Feature importance confirms the mechanism:
`lag_168h` carries 35.6% of gain.

**R² is uninformative on this problem.** Ridge and gradient boosting both report
0.957 while differing by 2.35 MAE. Any model
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
| 80% | 80% | **71.2%** | -0.0878 | ±14.7 | ❌ under-covers |
| 90% | 90% | **87.2%** | -0.0281 | ±28.3 | ✅ calibrated |
| 95% | 95% | **93.1%** | -0.0191 | ±43.1 | ✅ calibrated |

**A negative result.** The 80% interval under-covers materially. Split
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

The browser scorer is a hour-of-week × zone surrogate, not the LightGBM model. Its agreement with the parent is reported here so the live demo is read as an approximation with a known gap, not as the trained model. Measured fidelity: R² = 0.8676
against the parent model on the same holdout.

## 8. Limitations

- Six months of a single year cannot express annual seasonality; the model must not be read as capturing winter demand.
- 2014 Uber volume grew steeply month over month. Trend is in-sample here, so forecasts beyond the observed window will extrapolate a growth rate that did not continue indefinitely.
- K-means assumes isotropic clusters in degrees; one degree of longitude is shorter than one of latitude at 40°N, so zones are mildly stretched east-west. At this scale the distortion is under 25% and does not affect ranking, but it is a real approximation.
- LightGBM cannot extrapolate beyond the demand levels seen in training. 2014 volume was growing steeply, so a forecast far past the window would saturate at the training maximum rather than continue the trend.
- Conformal coverage degrades at the narrowest band: the 80% interval realises 71.2% rather than its nominal level. Split conformal assumes the calibration and test residuals are exchangeable. They are not here: calibration comes from the chronologically last slice of training data, and demand grew substantially across the six months, so test-period residuals are systematically larger than calibration-period residuals. The wider 90% and 95% bands absorb that drift and remain calibrated; the tight band does not. This is a real limitation of the method under trend, not a tuning artefact, and the fix is periodic recalibration on recent data.
- Error concentrates at hour 18 (MAE 26.52 against mean demand 212.8) — the evening peak, where both demand and its variance are highest.
- Zone 9 carries the largest absolute error (MAE 45.37); high-volume zones dominate the aggregate figure, so a single citywide MAE understates performance in quiet zones and overstates it in busy ones.
- Model selection used a single chronological holdout. With one test window there is no estimate of variance across periods; a different four weeks would give a different number.
- The surrogate ignores autoregressive state, so it cannot react to a demand shock the way the parent model does. It reproduces the periodic structure only.

## 9. Reproduction

See [`pipeline/build.py`](./pipeline/build.py). All artifacts in
[`artifacts/`](./artifacts/) carry the commit, seed and library versions that produced them.

*Generated at commit `a90a660` · seed 42 · 43.12s · Python 3.11.15 · numpy 2.4.6 · pandas 3.0.5 · sklearn 1.9.1 · lightgbm 4.7.0*
