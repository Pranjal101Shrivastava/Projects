# No Free Lunch in Forecasting: A Four-Series Rolling-Origin Study

## 1. Problem with single-series benchmarks

A forecasting comparison on one series and one holdout cannot distinguish a genuinely better
method from one that happens to suit that series, nor a better method from a luckier split.
This study addresses both: four series with deliberately different structure, and
rolling-origin evaluation with refitting at each origin.

## 2. Error measure

Raw MAE cannot be averaged across series measuring passengers, degrees Celsius, prescription
volumes and sunspot area. MASE normalises by the in-sample mean absolute error of the
seasonal-naive forecast:

    MASE = mean(|y − ŷ|) / mean(|y_t − y_{t−m}|)

The denominator makes it unit-free; the reference point makes it interpretable. MASE = 1.0
means the forecast matched seasonal naive.

MAPE was rejected as undefined at zero, explosive near it, and asymmetric.

## 3. Protocol

Rolling origin: 8 refits, each forecasting 12 steps ahead from data strictly before the origin. No model sees any observation at or after its origin. Reported MASE is the mean across origins, so a single lucky cut cannot determine the ranking.

Direct multi-step was used for the ML forecaster: h separate models, one per horizon step.
Recursive one-step forecasting feeds a model its own predictions, so error compounds and the
last steps of the horizon are predictions of predictions.

## 4. Series characterisation

| Series | n | Trend strength | Seasonal strength | ADF p | KPSS p | Verdict |
|---|---:|---:|---:|---:|---:|---|
| International Airline Passengers | 144 | 0.996 | 0.975 | 0.9919 | 0.0100 | non-stationary (both tests agree) — differen |
| Melbourne Minimum Temperature (monthly mean) | 120 | 0.256 | 0.950 | 0.3357 | 0.1000 | conflicting: ADF cannot reject a unit root b |
| Australian Antidiabetic Drug Sales | 204 | 0.985 | 0.854 | 1.0000 | 0.0100 | non-stationary (both tests agree) — differen |
| Sunspot Area | 137 | 0.301 | 0.694 | 0.4767 | 0.0529 | conflicting: ADF cannot reject a unit root b |

Both stationarity tests were run because their null hypotheses are opposites. A single test
that fails to reject is ambiguous between "the null holds" and "insufficient power"; running
both distinguishes trend-stationary from difference-stationary series.

## 5. Results

### International Airline Passengers

*upward trend with multiplicative annual seasonality* · 144 observations · 1949-01-01 → 1960-12-01 ·
period 12

Trend strength 0.996 · seasonal strength 0.975 ·
ADF p = 0.9919 · KPSS p = 0.0100

**Stationarity:** non-stationary (both tests agree) — differencing required

| Model | MASE | ± std | Worst origin | Beats naive |
|---|---:|---:|---:|:---:|
| SARIMA(1,1,1)(1,1,1,m) | 0.648 | ±0.400 | 1.624 | ✅ |
| Holt-Winters (add/add) | 0.685 | ±0.236 | 1.201 | ✅ |
| LightGBM direct multi-step | 1.187 | ±0.162 | 1.488 | ✅ |
| Seasonal naive | 1.313 | ±0.494 | 1.959 | — |
| Drift | 1.813 | ±0.474 | 2.766 | — |
| Naive (last value) | 2.119 | ±0.537 | 3.196 | — |
### Melbourne Minimum Temperature (monthly mean)

*strong annual seasonality, no trend* · 120 observations · 1981-01-01 → 1990-12-01 ·
period 12

Trend strength 0.256 · seasonal strength 0.950 ·
ADF p = 0.3357 · KPSS p = 0.1000

**Stationarity:** conflicting: ADF cannot reject a unit root but KPSS cannot reject stationarity — the series is likely difference-stationary but the tests are underpowered here

| Model | MASE | ± std | Worst origin | Beats naive |
|---|---:|---:|---:|:---:|
| SARIMA(1,1,1)(1,1,1,m) | 0.894 | ±0.274 | 1.277 | ✅ |
| LightGBM direct multi-step | 0.920 | ±0.225 | 1.291 | ✅ |
| Holt-Winters (add/add) | 0.925 | ±0.271 | 1.331 | ✅ |
| Seasonal naive | 0.926 | ±0.207 | 1.227 | — |
| Drift | 2.687 | ±0.578 | 4.004 | — |
| Naive (last value) | 2.892 | ±0.591 | 4.132 | — |
### Australian Antidiabetic Drug Sales

*trend with a sharp December spike* · 204 observations · 1991-07-01 → 2008-06-01 ·
period 12

Trend strength 0.985 · seasonal strength 0.854 ·
ADF p = 1.0000 · KPSS p = 0.0100

**Stationarity:** non-stationary (both tests agree) — differencing required

| Model | MASE | ± std | Worst origin | Beats naive |
|---|---:|---:|---:|:---:|
| SARIMA(1,1,1)(1,1,1,m) | 1.116 | ±0.497 | 2.088 | ✅ |
| Holt-Winters (add/add) | 1.124 | ±0.470 | 2.136 | ✅ |
| LightGBM direct multi-step | 1.593 | ±0.284 | 1.994 | ✅ |
| Seasonal naive | 1.915 | ±0.711 | 3.129 | — |
| Drift | 2.345 | ±0.535 | 3.580 | — |
| Naive (last value) | 2.582 | ±0.556 | 3.926 | — |
### Sunspot Area

*~11 year cycle, not calendar-anchored* · 137 observations · 1875-01-01 → 2011-01-01 ·
period 11

Trend strength 0.301 · seasonal strength 0.694 ·
ADF p = 0.4767 · KPSS p = 0.0529

**Stationarity:** conflicting: ADF cannot reject a unit root but KPSS cannot reject stationarity — the series is likely difference-stationary but the tests are underpowered here

| Model | MASE | ± std | Worst origin | Beats naive |
|---|---:|---:|---:|:---:|
| Seasonal naive | 1.501 | ±0.349 | 2.255 | — |
| LightGBM direct multi-step | 1.692 | ±0.382 | 2.157 | — |
| SARIMA(1,1,1)(1,1,1,m) | 1.797 | ±0.656 | 2.997 | — |
| Holt-Winters (add/add) | 1.841 | ±0.642 | 3.218 | — |
| Naive (last value) | 2.408 | ±0.623 | 3.660 | — |
| Drift | 2.425 | ±0.630 | 3.688 | — |


## 6. Cross-series synthesis

| Model | Mean rank | Wins | Best | Worst |
|---|---:|---:|---:|---:|
| SARIMA(1,1,1)(1,1,1,m) | 1.50 | 3 | 1 | 3 |
| LightGBM direct multi-step | 2.50 | 0 | 2 | 3 |
| Holt-Winters (add/add) | 2.75 | 0 | 2 | 4 |
| Seasonal naive | 3.25 | 1 | 1 | 4 |
| Drift | 5.25 | 0 | 5 | 6 |
| Naive (last value) | 5.75 | 0 | 5 | 6 |

**2 distinct winners across four series.** The sunspot result is
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

- Model hyperparameters are fixed rather than tuned per series. Tuning would likely improve the ML forecaster most, so the comparison is mildly conservative towards it.
- SARIMA order is fixed at (1,1,1)(1,1,1,m) rather than selected per series by AIC. A per-series search would be fairer to SARIMA and is the obvious next step.
- Series lengths range from 141 to 3,650 observations, so the number of usable rolling origins differs and short-series estimates are noisier.

*Generated at commit `02a5f5a` · seed 42 · 32.98s · Python 3.11.15 · numpy 2.4.6 · pandas 3.0.5 · sklearn 1.9.1 · lightgbm 4.7.0*
