# 12 · Market Backtest — A Negative Result, Reported

A trading study built to be hard to fool yourself with: purged walk-forward cross-validation,
transaction costs, a buy-and-hold benchmark, and a look-ahead test that perturbs the future to
prove the features cannot see it. **0 of
2 strategies beat the benchmark.** That is the result, and it is
published as the result.

| Dataset | Kind | Size | Origin | Licence |
|---|:---:|---|---|---|
| [Apple Inc. Daily OHLCV (Feb 2015 - Feb 2017)](https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv) | 🟢 **REAL** | 506 trading days | Daily open/high/low/close/volume bars for AAPL with split- and dividend-adjusted closes, as distributed in Plotly's finance chart examples. | Open for educational use. |

## The trap this project exists to avoid

| Model | R² |
|---|---:|
| Predicting tomorrow's **price** from today's price | **0.9753** |
| The identical model, restated as a **return** forecast | **-0.0001** |

Repeating yesterday's price 'predicts' tomorrow's price with R² = 0.9753. The identical model, restated as a return forecast of zero, scores R² = -0.0001. Same model, same data, same information content — which is none. The first number is an artefact of the target's non-stationarity, not evidence of skill, and it is the single most common way financial ML results are overstated.

The statistical backing: an augmented Dickey-Fuller test rejects a unit root in returns
(p = 0.0000) and fails to reject it in price
(p = 0.685). ADF rejects a unit root in returns but not in price — the textbook result, and the reason the target is returns. Modelling price directly means modelling a non-stationary series with a regression that assumes otherwise.

## Step one: did the models predict anything?

Before any trading logic, before any equity curve — does the signal exist at all?

| Model | Information coefficient | p-value | Significant | Directional accuracy | R² on returns | Trades | Time in market |
|---|---:|---:|:---:|---:|---:|---:|---:|
| ridge | -0.0624 | 0.2308 | **no** | 51.6% | -0.2987 | 88 | 47.0% |
| lightgbm | -0.0938 | 0.0716 | **no** | 48.9% | -0.2915 | 93 | 46.5% |

Both information coefficients are small, negative, and carry p-values above 0.05. Directional
accuracy sits within a few points of a coin flip. **There is no signal here**, and everything
downstream follows from that.

Reporting a strategy's returns without first establishing that its *predictions* have
measurable skill is how backtests get published. The IC column is the check that comes first.

## Step two: the backtest anyway

Net of 10 bps round-trip costs:

| Strategy | Total return | Annualised | Volatility | Sharpe | Sortino | Max drawdown | Hit rate | Periods |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| buy & hold | -1.4% | -0.9% | 25.4% | **-0.035** | -0.046 | -32.0% | 50.9% | 395 |
| ridge | -20.2% | -14.2% | 19.3% | **-0.738** | -0.746 | -31.9% | 20.8% | 370 |
| lightgbm | -20.0% | -14.1% | 19.6% | **-0.719** | -0.720 | -30.7% | 21.3% | 370 |

Gross, for comparison:

| Strategy | Total return | Annualised | Volatility | Sharpe | Sortino | Max drawdown | Hit rate | Periods |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| buy & hold | -1.3% | -0.8% | 25.4% | **-0.032** | -0.042 | -32.0% | 50.9% | 395 |
| ridge | -12.8% | -8.9% | 19.3% | **-0.463** | -0.422 | -29.8% | 21.6% | 370 |
| lightgbm | -12.2% | -8.5% | 19.5% | **-0.433** | -0.389 | -26.1% | 21.9% | 370 |

At 10 bps round trip, a strategy switching position every other day trades ~126 times a year and must generate roughly 12.6% of gross annual alpha simply to break even. Costs are not a detail at daily frequency; they are usually the whole result.

Both models were already losing money **before** costs, so costs are not the reason this study
failed — but they are the reason a study that looks marginally profitable gross usually is not
worth running.

The benchmark covers 395 periods against the strategies'
370: a model needs training history before it can
trade, so each series is scored over its own periods rather than forced onto a common window
that would flatter one of them.

## Purged walk-forward folds

| Model | Fold | Train rows | Test rows | MAE | Directional accuracy |
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

Directional accuracy swings from 35.1% to 64.9% across folds of
74 rows. Any single fold could be quoted as a
triumph or a disaster; the average is the only reading that survives having fixed the window in
advance.

A 5-day forward return computed every day means consecutive rows share 4 days of outcome. Adjacent rows are therefore close to the same observation, which is why the CV below purges a band around every boundary rather than relying on chronological ordering alone.

## Proving there is no look-ahead

Static analysis flags suspicious shifts. It does not prove anything, so the pipeline runs an
empirical test: **future-bar perturbation**.

All price and volume columns are multiplied by 1.5 from row 253 onward, the
entire feature matrix is rebuilt, and every feature value at the 254
*earlier* rows is compared against the unperturbed build.

**Maximum drift: 0.0 across 14 features.**
Passed.

Two features trip the static lookahead rule because the shift is applied outside the flagged expression — across a .pipe() boundary in one case and at the end of a helper in the other. This test verifies the composed result rather than the syntax, and is the evidence behind those acknowledgements.

Leakage controls in force:

- Every feature is shifted by at least one day before any rolling window, so no feature uses the bar it is predicting from.
- The target is the only quantity computed forward in time.
- Cross-validation purges 5 rows between train and test to remove the label-overlap band.
- Adjusted prices throughout, so dividends do not appear as returns.

## The series itself

| Property | Value |
|---|---:|
| Window | 2015-02-17 → 2017-02-16 |
| Trading days | 506 |
| Price | 122.91 → 135.35 |
| Buy & hold, full sample | 10.1% |
| Annualised volatility | 24.3% |
| Skew | -0.1922 |
| Excess kurtosis | 3.1175 |
| Forecast horizon | 5 days |
| Features | 14 |
| Rows modelled | 479 |

Excess kurtosis 3.12 means fat tails — returns are not Gaussian, which
is one more reason to read maximum drawdown beside any Sharpe ratio.

Adjusted and unadjusted closes differ by up to 3.85% over this window. Using the unadjusted series would insert a fake negative return at every dividend date, which a momentum feature reads as signal.

## Conclusion

No strategy beat buy-and-hold on a risk-adjusted basis after costs. This is the expected outcome and it is reported as the result. Two years of daily bars on one instrument contains very little exploitable structure, the information coefficients are small and not statistically distinguishable from zero, and transaction costs consume what remains.

## Screens

![12 backtest](../../docs/screenshots/12_backtest.png)


## CRISP-DM record

> **Business question.** Can a model predict short-horizon equity returns well enough to beat holding the stock, once transaction costs and honest validation are applied?

### Business Understanding

A trading strategy is only worth running if it beats the passive alternative after costs. That comparison is what most forecasting write-ups omit, and omitting it is what allows a strategy with no edge to look successful. The benchmark here is buy-and-hold on the same instrument over the same window, net of the same cost model.

**What is the prediction target?**

- **Chose:** 5-day forward log return, not price.
- **Why:** Price is very close to a random walk, so predicting it yields R² above 0.99 for a model that does nothing but repeat the last value. That statistic is vacuous and is nonetheless reported as a result throughout the practitioner literature. Returns are approximately stationary and are what a position actually earns.
- *Rejected:* Predict the closing price — produces spectacular R² and zero information.
- *Rejected:* Predict direction only — discards magnitude, which is what determines whether a trade covers its costs.

**What must the strategy beat?**

- **Chose:** Buy-and-hold, net of identical transaction costs.
- **Why:** An absolute return figure is uninterpretable. If the instrument rose 30% over the window, a strategy returning 12% destroyed value while appearing profitable.

### Data Understanding

506 real daily bars, 2015-02-17 to 2017-02-16. Returns show excess kurtosis 3.12 — fat tails, so Gaussian risk assumptions understate drawdowns. ADF rejects a unit root in returns but not in price.

**Adjusted or unadjusted closing prices?**

- **Chose:** Adjusted, always.
- **Why:** Adjusted and unadjusted closes differ by up to 3.85% over this window. Using the unadjusted series would insert a fake negative return at every dividend date, which a momentum feature reads as signal.
- *Rejected:* Raw close — inserts a fabricated negative return at every dividend, which momentum features read as signal.

### Data Preparation

14 features, all shifted before any rolling window. Target is the 5-day forward log return — the only quantity computed forward in time.

**How is overlapping-label leakage handled?**

- **Chose:** Purged walk-forward CV with a 5-row embargo.
- **Why:** A 5-day forward return computed every day means consecutive rows share 4 days of outcome. Adjacent rows are therefore close to the same observation, which is why the CV below purges a band around every boundary rather than relying on chronological ordering alone.
- *Rejected:* Plain TimeSeriesSplit — chronological but leaves train and test touching, so overlapping labels span the boundary.
- *Rejected:* Random K-fold — trains on the future; the most common error in published financial ML.

### Modeling

Two deliberately small models — a heavily regularised ridge and a shallow gradient booster — evaluated out-of-fold under purged walk-forward CV, then converted into a long-only strategy and charged 10 bps per round trip.

**Why report information coefficient rather than R²?**

- **Chose:** Spearman IC, with its p-value.
- **Why:** R² on returns is near zero for every model, including good ones, so it cannot distinguish between them. IC measures rank agreement between forecast and outcome, which is what a position-sizing rule actually consumes. Its p-value is reported because a small IC on a short sample is usually noise.

**Why long-only rather than long/short?**

- **Chose:** Long-only.
- **Why:** A long/short result on two years of one instrument would be dominated by assumptions about borrow availability and cost that this data cannot support.

### Evaluation

No strategy beat buy-and-hold on a risk-adjusted basis after costs. This is the expected outcome and it is reported as the result. Two years of daily bars on one instrument contains very little exploitable structure, the information coefficients are small and not statistically distinguishable from zero, and transaction costs consume what remains.

**Limitations**

- 506 trading days of a single instrument. Sharpe ratios on samples this short have very wide confidence intervals — a difference of 0.5 is not distinguishable from zero here.
- One stock over one period that happened to trend. Nothing here generalises to other instruments, regimes or horizons.
- Costs are modelled as a flat 10 bps. Real execution adds slippage that scales with size and widens sharply in volatile periods, so these net figures are optimistic.
- Feature and model choices were made by the author with knowledge of the dataset. Even with clean CV, that is a form of selection bias no backtest can remove — only out-of-sample deployment can.

### Deployment

Equity curves, fold-level diagnostics and the cost decomposition ship so a reader can see gross and net side by side. Nothing here is deployable as a trading system and the page says so.


## Run it

```bash
git clone https://github.com/Pranjal101Shrivastava/Projects
cd Projects
pip install -r requirements.txt
export PYTHONPATH=lib

python3 projects/12_market_backtest/pipeline/build.py   # rebuilds every artifact below
python3 tools/audit.py                          # static leakage audit
```

Datasets download on first run into `.data/` and are verified against their recorded
SHA-256 on every run thereafter. The pipeline is seeded, so a rerun at the same commit
reproduces the same numbers.

---

**Live:** [https://pranjal101shrivastava.github.io/Projects/#/p/backtest](https://pranjal101shrivastava.github.io/Projects/#/p/backtest) *(requires GitHub Pages enabled)* ·
**Method:** [`pipeline/build.py`](./pipeline/build.py) ·
**Audit:** [`audit.md`](./audit.md) ·
**Artifacts:** [`artifacts/`](./artifacts/)

*Generated at commit `442ebf8` · seed 42 · 0.77s · Python 3.11.15 · numpy 2.4.6 · pandas 3.0.5 · sklearn 1.9.1 · lightgbm 4.7.0 · torch 2.14.0+cu130*
