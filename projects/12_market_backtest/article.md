# I built a trading model, tested it properly, and it lost money

*So I published it*

This is the project where nothing worked, which is exactly why it is here.

## The number that should scare you

Here is a model I could have led with:

> **R² = 0.9753** predicting tomorrow's AAPL
> price.

Impressive, right? It is repeating yesterday's price.

Here is the identical model, restated as a forecast of tomorrow's *return*:

> **R² = -0.0001**

Same model. Same data. Same information content, which is none.

Repeating yesterday's price 'predicts' tomorrow's price with R² = 0.9753. The identical model, restated as a return forecast of zero, scores R² = -0.0001. Same model, same data, same information content — which is none. The first number is an artefact of the target's non-stationarity, not evidence of skill, and it is the single most common way financial ML results are overstated.

If you have ever seen a financial ML post with a stunning R² and a chart of predicted-vs-actual
prices hugging a diagonal, this is what you were looking at.

## What I actually tested

14 features — lagged returns, momentum, realised volatility, RSI, volume
z-score — predicting the 5-day forward return on 506 days of
AAPL. Ridge and LightGBM. Purged walk-forward CV.

And before looking at a single equity curve, I asked the question that comes first: **do the
predictions correlate with what happened?**

| Model | Information coefficient | p-value |
|---|---:|---:|
| ridge | -0.0624 | 0.231 |
| lightgbm | -0.0938 | 0.072 |

Both negative. Both with p-values above 0.05. Directional accuracy:
51.6%, 48.9% — a coin flip scores
50%.

There was no signal. Everything after this point is arithmetic on noise.

## The backtest, run anyway

| Strategy | Net annualised | Net Sharpe | Max drawdown |
|---|---:|---:|---:|
| ridge | -14.2% | **-0.738** | -31.9% |
| lightgbm | -14.1% | **-0.719** | -30.7% |
| buy & hold | -0.9% | -0.035 | -32.0% |

Both strategies lost to doing nothing. Both lost money in absolute terms. Both were already
losing **before** I charged 10 bps of transaction costs.

At 10 bps round trip, a strategy switching position every other day trades ~126 times a year and must generate roughly 12.6% of gross annual alpha simply to break even. Costs are not a detail at daily frequency; they are usually the whole result.

## The test I'm proudest of

My own static leakage scanner flagged two features in this pipeline. I could have written a
comment saying "this is fine, trust me". Instead I made the pipeline prove it.

The test: take all the future price and volume bars from row 253 onward and
multiply them by 1.5. Rebuild the entire feature matrix. Then check every feature value at
every one of the 254 *earlier* rows.

If any of them moved, that feature is reading the future.

**Maximum drift: 0.0.** Nothing moved. Not one bit, across
14 features.

Two features trip the static lookahead rule because the shift is applied outside the flagged expression — across a .pipe() boundary in one case and at the end of a helper in the other. This test verifies the composed result rather than the syntax, and is the evidence behind those acknowledgements.

That is what "no look-ahead" should mean — a measurement, not an assurance.

## Why this is the honest outcome

Two years of daily data on one stock. Roughly 479 usable rows,
14 features, overlapping labels. If I had found a Sharpe of 1.5 in there, the
correct reaction would have been suspicion, not celebration.

The things that would have manufactured one are all well known: forget to purge overlapping
labels, model price instead of returns, drop transaction costs, omit the benchmark, or simply
try models until one works and report that one.

I did none of those, and got nothing. That is the system working.

## What I'd want you to take from this

The value here is not the models. It is the protocol:

1. Establish the target is stationary before modelling it.
2. Measure signal (IC) **before** looking at returns.
3. Purge overlapping labels out of the CV boundary.
4. Charge costs, and show gross beside net.
5. Compare against buy-and-hold, not against zero.
6. Prove there is no look-ahead by perturbing the future.
7. Publish the result you got.

Step seven is the hard one.

---

**Live demo:** [https://pranjal101shrivastava.github.io/Projects/#/p/backtest](https://pranjal101shrivastava.github.io/Projects/#/p/backtest)
**Code:** [github.com/Pranjal101Shrivastava/Projects](https://github.com/Pranjal101Shrivastava/Projects)
