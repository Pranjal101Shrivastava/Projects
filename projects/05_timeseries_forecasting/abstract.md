# Abstract — Cross-Series Forecasting Evaluation with Rolling-Origin Backtests

**Objective.** Determine whether any forecasting method should be preferred as a default, and
quantify how far that answer depends on the structure of the series being forecast.

**Data.** Four real series chosen for structural diversity: international airline passengers
(trend with multiplicative annual seasonality), Melbourne minimum temperature (strong annual
seasonality, no trend), Australian antidiabetic drug sales (trend with a December spike), and
sunspot area (an approximately eleven-year cycle with no calendar anchor).

**Method.** Six forecasters spanning three families — naive baselines, classical statistical
methods (Holt-Winters, SARIMA) and machine learning (direct multi-step gradient boosting) —
were evaluated under rolling-origin backtesting with up to eight refits per series at horizon
12. Errors were reported as mean absolute scaled error, normalised by in-sample
seasonal-naive error to permit comparison across incommensurable units. Each series was
decomposed by STL and tested for stationarity by both ADF and KPSS.

**Results.** 2 distinct models took first place across four series.
SARIMA achieved the best mean rank (1.50) with
3 outright wins. On the sunspot series, no learned
model outperformed seasonal naive. Gradient boosting won no series but never ranked below
third, and was outperformed by classical statistical methods on three of four.

**Conclusion.** No forecaster dominates across structurally different series. Single-series
benchmarking is insufficient to support a general recommendation, and series whose periodicity
is not calendar-anchored defeat methods that assume it.

**Keywords.** time series forecasting, rolling-origin evaluation, MASE, SARIMA, stationarity
testing
