# An Honest Backtest and Why It Found Nothing

*Purged walk-forward evaluation on 506 daily AAPL bars*

## 1. Problem

Backtests are unusually easy to get wrong in a direction that flatters the author. Look-ahead
bias, survivorship, overlapping labels, omitted transaction costs and missing benchmarks each
independently produce apparent edges. This study fixes the protocol in advance and reports
whatever comes out.

## 2. Data

506 adjusted daily bars, 2015-02-17 to 2017-02-16. Price moved
122.91 → 135.35, a buy-and-hold total return of
10.1% over the full sample.

Adjusted and unadjusted closes differ by up to 3.85% over this window. Using the unadjusted series would insert a fake negative return at every dividend date, which a momentum feature reads as signal.

Return distribution: mean 0.000191, standard deviation 0.0153, skew
-0.1922, excess kurtosis 3.1175, annualised volatility
24.3%.

## 3. The stationarity trap

| Model | R² |
|---|---:|
| Tomorrow's price from today's price | 0.97532 |
| Same model as a return forecast | -0.00013 |

Repeating yesterday's price 'predicts' tomorrow's price with R² = 0.9753. The identical model, restated as a return forecast of zero, scores R² = -0.0001. Same model, same data, same information content — which is none. The first number is an artefact of the target's non-stationarity, not evidence of skill, and it is the single most common way financial ML results are overstated.

ADF p-values: returns 0.0000, price
0.6848. ADF rejects a unit root in returns but not in price — the textbook result, and the reason the target is returns. Modelling price directly means modelling a non-stationary series with a regression that assumes otherwise.

## 4. Features and leakage controls

14 features: `ret_lag_1`, `ret_lag_2`, `ret_lag_3`, `ret_lag_5`, `ret_lag_10`, `mom_5`, `vol_5`, `mom_10`, `vol_10`, `mom_21`, `vol_21`, `rsi_14`, `volume_z`, `hl_range`.

Controls:

1. Every feature is shifted by at least one day before any rolling window, so no feature uses the bar it is predicting from.
2. The target is the only quantity computed forward in time.
3. Cross-validation purges 5 rows between train and test to remove the label-overlap band.
4. Adjusted prices throughout, so dividends do not appear as returns.

### 4.1 Empirical look-ahead test

Static rules cannot establish absence of look-ahead when a shift is applied across a `.pipe()`
boundary or inside a helper. The pipeline therefore perturbs the future and checks the past:

| Parameter | Value |
|---|---|
| Test | future-bar perturbation |
| Cut row | 253 |
| Perturbation | +50% to all price and volume columns from cut onward |
| Features checked | 14 |
| Earlier rows checked | 254 |
| Maximum drift | **0.0** |
| Result | passed |

Two features trip the static lookahead rule because the shift is applied outside the flagged expression — across a .pipe() boundary in one case and at the end of a helper in the other. This test verifies the composed result rather than the syntax, and is the evidence behind those acknowledgements.

## 5. Cross-validation

A 5-day forward return computed every day means consecutive rows share 4 days of outcome. Adjacent rows are therefore close to the same observation, which is why the CV below purges a band around every boundary rather than relying on chronological ordering alone.

| Model | Fold | Train | Test | MAE | Directional accuracy |
|---|---:|---:|---:|---:|---:|
| ridge | 0 | 79 | 74 | 0.04577 | 44.6% |
| ridge | 1 | 158 | 74 | 0.02914 | 54.0% |
| ridge | 2 | 237 | 74 | 0.02956 | 59.5% |
| ridge | 3 | 316 | 74 | 0.02875 | 35.1% |
| ridge | 4 | 395 | 74 | 0.02066 | 64.9% |
| lightgbm | 0 | 79 | 74 | 0.03038 | 58.1% |
| lightgbm | 1 | 158 | 74 | 0.03305 | 50.0% |
| lightgbm | 2 | 237 | 74 | 0.03696 | 40.5% |
| lightgbm | 3 | 316 | 74 | 0.03014 | 39.2% |
| lightgbm | 4 | 395 | 74 | 0.02392 | 56.8% |

## 6. Signal, measured before returns

| Model | IC | p | Significant | Directional accuracy | R² on returns |
|---|---:|---:|:---:|---:|---:|
| ridge | -0.0624 | 0.2308 | no | 51.6% | -0.2987 |
| lightgbm | -0.0938 | 0.0716 | no | 48.9% | -0.2915 |

## 7. Backtest results

Net of 10 bps:

| Strategy | Total | Annualised | Vol | Sharpe | Sortino | Max DD | Hit rate | Periods |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| buy & hold | -1.4% | -0.9% | 25.4% | **-0.035** | -0.046 | -32.0% | 50.9% | 395 |
| ridge | -20.2% | -14.2% | 19.3% | **-0.738** | -0.746 | -31.9% | 20.8% | 370 |
| lightgbm | -20.0% | -14.1% | 19.6% | **-0.719** | -0.720 | -30.7% | 21.3% | 370 |

Gross:

| Strategy | Total | Annualised | Vol | Sharpe | Sortino | Max DD | Hit rate | Periods |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| buy & hold | -1.3% | -0.8% | 25.4% | **-0.032** | -0.042 | -32.0% | 50.9% | 395 |
| ridge | -12.8% | -8.9% | 19.3% | **-0.463** | -0.422 | -29.8% | 21.6% | 370 |
| lightgbm | -12.2% | -8.5% | 19.5% | **-0.433** | -0.389 | -26.1% | 21.9% | 370 |

Cost drag: ridge 8.8%, lightgbm 9.3%.

At 10 bps round trip, a strategy switching position every other day trades ~126 times a year and must generate roughly 12.6% of gross annual alpha simply to break even. Costs are not a detail at daily frequency; they are usually the whole result.

## 8. Verdict

No strategy beat buy-and-hold on a risk-adjusted basis after costs. This is the expected outcome and it is reported as the result. Two years of daily bars on one instrument contains very little exploitable structure, the information coefficients are small and not statistically distinguishable from zero, and transaction costs consume what remains.

## 9. Limitations

- 506 trading days of a single instrument. Sharpe ratios on samples this short have very wide confidence intervals — a difference of 0.5 is not distinguishable from zero here.
- One stock over one period that happened to trend. Nothing here generalises to other instruments, regimes or horizons.
- Costs are modelled as a flat 10 bps. Real execution adds slippage that scales with size and widens sharply in volatile periods, so these net figures are optimistic.
- Feature and model choices were made by the author with knowledge of the dataset. Even with clean CV, that is a form of selection bias no backtest can remove — only out-of-sample deployment can.

*Generated at commit `442ebf8` · seed 42 · 0.76s · Python 3.11.15 · numpy 2.4.6 · pandas 3.0.5 · sklearn 1.9.1 · lightgbm 4.7.0 · torch 2.14.0+cu130*
