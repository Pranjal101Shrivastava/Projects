# Abstract — NYC Ride-Hail Demand Forecasting

**Objective.** Forecast hourly ride-hail pickup demand at zone granularity to support
vehicle positioning, and quantify the uncertainty of those forecasts honestly.

**Data.** 4,534,327 real dispatched pickups from the NYC Taxi & Limousine
Commission's Freedom of Information Law release, covering
2014-04-01 to 2014-09-30 across 5 dispatch
bases. The release contains timestamp, coordinates and base only; it has no trip duration or
fare field, which constrains the target to demand.

**Method.** Point events were aggregated to a 12-zone × hourly panel of
50,676 observations with 21 strictly backward-looking
features: autoregressive lags to one week, rolling statistics shifted by one period before
aggregation, cyclically encoded calendar terms and holiday flags. Zones were learned by
K-means over pickup coordinates. Evaluation used a chronological holdout of the final
28 days with a 168-hour embargo at
the boundary, matching the widest feature lag. Uncertainty was quantified by split-conformal
calibration on a held-out training slice.

**Results.** Gradient boosting achieved MAE 15.65 against 25.70 for
a seasonal-naive baseline, a 39.1% reduction. Daily
periodicity proved a weaker baseline than weekly (35.36 MAE), and
`lag_168h` carried 35.6% of model gain. Conformal intervals were
empirically calibrated at the 90% and 95% levels but under-covered at 80%, realising
71.2%.

**Conclusion.** The coverage failure is attributable to non-exchangeability under trend:
demand grew 82.9% across the observation window, so calibration residuals drawn from the
training period understate test-period error. Wider bands absorb the drift; narrow ones do
not. Periodic recalibration on recent data is required for conformal methods applied to
growing series.

**Keywords.** demand forecasting, gradient boosting, conformal prediction, temporal
validation, spatio-temporal analysis
