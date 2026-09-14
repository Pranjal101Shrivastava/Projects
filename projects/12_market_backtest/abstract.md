# Abstract — A Purged Walk-Forward Backtest with a Negative Result

**Objective.** Test whether daily-frequency machine learning models extract tradeable signal
from a single liquid equity under realistic evaluation, and report the outcome irrespective of
sign.

**Data.** 506 daily adjusted bars for AAPL, 2015-02-17 to 2017-02-16.
Adjusted closes differ from unadjusted by up to 3.9% over the
window; unadjusted series would inject spurious negative returns at dividend dates.
14 features were derived from lagged returns, momentum, realised volatility,
RSI, volume z-score and high-low range, yielding 479 modelled rows at a
5-day forward-return horizon.

**Method.** Ridge regression and gradient boosting were evaluated under purged walk-forward
cross-validation (5 expanding folds) with a purge band
removing the label-overlap region. Signal was assessed by information coefficient (rank
correlation between prediction and realised return) before any trading rule was applied.
Positions were simulated with 10 bps round-trip transaction costs and compared
against buy-and-hold. Absence of look-ahead was verified empirically by perturbing all future
price and volume bars by +50% from row 253 and confirming no earlier feature value
changed.

**Results.** Information coefficients were -0.0624 (p = 0.231), -0.0938 (p = 0.072)
— neither distinguishable from zero. Directional accuracy was
51.6%, 48.9%. Net Sharpe ratios were
-0.738, -0.719 against buy-and-hold's
-0.035. No strategy beat the benchmark on return or on a risk-adjusted basis.
The look-ahead perturbation test recorded a maximum drift of 0.0 across
14 features and 254 rows.

**Conclusion.** Two years of daily bars on one instrument contain little exploitable structure;
the models found none, and transaction costs would have consumed any marginal edge. The
methodological contribution is the evaluation protocol — purged splits, cost accounting, a
benchmark, an empirical look-ahead test, and signal measured before returns — under which a
negative result is both detectable and reportable.

**Keywords.** backtesting, purged cross-validation, information coefficient, transaction costs,
negative result, look-ahead bias
