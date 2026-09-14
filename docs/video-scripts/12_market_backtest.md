# Video 12 · Market Backtest

**Estimated runtime:** ~7:27 at a measured pace (1,006 spoken words)
**Live page:** https://pranjal101shrivastava.github.io/Projects/#/p/backtest
**Project directory:** [`projects/12_market_backtest/`](../../projects/12_market_backtest/)

> **The one sentence this video has to land:** Information coefficients of -0.0624, -0.0938 — no measurable signal, reported as the finding rather than tuned away.

## Before you record

- Open **https://pranjal101shrivastava.github.io/Projects/#/p/backtest**.
- Toggle gross/net on the equity curves on camera.
- Have the perturbation-test panel ready — it's the strongest part of the video.

Everything in plain text below is meant to be **spoken**. Lines marked *On screen* and
*Note to self* are directions — do not read them aloud.

---

## [0:00] Cold open

> **On screen —** the two R-squared cards side by side

Here's a model I could have led with.

**R-squared of 0.9753** predicting
tomorrow's Apple share price.

That looks publishable. It is repeating yesterday's price.

Here is the identical model, restated as a forecast of tomorrow's **return**:

**R-squared of -0.0001**.

Same model. Same data. Same information content — which is none.

This is the project where nothing worked, and that is exactly why it's here.

## [0:28] Why the first number is an artefact

> **On screen —** the stationarity figures

The reason the first number is meaningless is that prices are non-stationary. They wander.
Today's price is the best single guess for tomorrow's, so a model that just repeats it
explains almost all the variance — and predicts nothing.

The statistical version: an augmented Dickey-Fuller test rejects a unit root in *returns*,
with p of 0.0000, and fails to reject it in *price*,
p of 0.685.

So the target has to be returns. Modelling price directly means fitting a regression that
assumes stationarity to a series that isn't.

If you have ever seen a financial machine-learning post with a stunning R-squared and a
chart of predicted-versus-actual prices hugging the diagonal — that's what you were looking
at.

## [1:20] The setup

> **On screen —** the sample facts table

506 daily bars of Apple, 2015-02-17 to 2017-02-16.
14 features — lagged returns, momentum, realised volatility, RSI, a volume
z-score — predicting the 5-day forward return. Two models: ridge
regression and gradient boosting.

One detail that matters: I'm using **adjusted** closes. Adjusted and unadjusted differ by up
to 3.9% over this window, and using unadjusted prices would
insert a fake negative return at every dividend date — which a momentum feature reads as
signal.

## [1:52] Measure signal before you measure money

> **On screen —** the information coefficient table

Now the step that should come first in every study like this, and usually doesn't.

Before building any trading rule, before plotting any equity curve: **do the predictions
correlate with what actually happened?**

That's the information coefficient — the rank correlation between prediction and realised
return.

ridge: -0.0624, with a p-value of 0.231.
lightgbm: -0.0938, with a p-value of 0.072.

Both negative. Both with p-values above nought point nought five. Directional accuracy:
51.6%, 48.9% — against a coin
flip at fifty percent.

There is no signal here. Everything after this point is arithmetic on noise.

And that's the discipline: an equity curve built from predictions with no measurable skill
will *sometimes look excellent* by chance. Checking the IC first is what stops you
publishing that.

## [2:50] The backtest anyway

> **On screen —** the equity curves with the cost toggle

I ran the backtest regardless, because the protocol is the point.

Net of 10 basis points of round-trip costs:

ridge: annualised -14.2%, Sharpe -0.738.
lightgbm: annualised -14.1%, Sharpe -0.719.

Buy and hold over the same window: -0.9% annualised, Sharpe
-0.035.

Both strategies lost to doing nothing. Both lost money outright. And both were already losing
**before** I charged any costs — so costs aren't the reason this failed, though they're the
reason a study that looks marginally profitable gross usually isn't worth running.

Watch that toggle: gross to net. At 10 bps round trip, a strategy switching position every other day trades ~126 times a year and must generate roughly 12.6% of gross annual alpha simply to break even. Costs are not a detail at daily frequency; they are usually the whole result.

## [3:52] Purged cross-validation

> **On screen —** the fold tables

Two things about the validation.

First: the folds are chronological and expanding. Fold one trains on the earliest data and
tests on what comes next. No model ever sees its own future.

Second, and subtler: my target is a 5-day forward return, computed
every day. So consecutive rows share 4 days of outcome. Adjacent
rows are nearly the same observation.

Chronological ordering alone doesn't fix that. The row just before the test boundary overlaps
with the row just after it. So the cross-validation **purges** a band around every boundary
and throws those rows away.

And look at the fold-to-fold variation: directional accuracy swings from 35.1% to
64.9% across folds of the same model.

Any single fold could be quoted as a triumph or a disaster. That range is exactly why fixing
the protocol in advance matters — after seeing that spread, I could pick a story.

## [4:57] Proving there's no look-ahead

> **On screen —** the perturbation test panel

This is the part I'm most pleased with.

My own static leakage scanner flagged two features in this pipeline. I could have written a
comment saying "this is fine, trust me." Comments aren't evidence.

So the pipeline proves it instead. The test: take every future price and volume bar from row
253 onward and **multiply them by 1.5**. Rebuild the entire feature matrix from
scratch. Then check every feature value at every one of the 254 rows
*before* the cut.

If any of them moved, that feature is reading the future.

**Maximum drift: 0.0.** Nothing moved. Not one bit, across
14 features.

And that test runs inside the pipeline on every execution. So a future edit that introduces
look-ahead fails the run, instead of producing a better-looking Sharpe ratio.

That's what "no look-ahead" should mean: a measurement, not an assurance.

## [6:00] Close

> **On screen —** the limitations section

So: no signal, no edge, no profit. Published as the result.

And I want to be careful about what this does *not* establish. One instrument, one two-year
window. Finding nothing here is evidence about this sample — not about momentum, or machine
learning, or equities in general. A study claiming it had *found* an edge on this same sample
would be making the much stronger and far less defensible claim.

Apple was also chosen with hindsight — it exists today and had a liquid history — which makes
buy-and-hold harder to beat than a fair universe would.

The value here isn't the models. It's the protocol:

Check the target is stationary before modelling it. Measure signal before returns. Purge
overlapping labels. Charge costs and show gross beside net. Benchmark against doing nothing.
Prove the absence of look-ahead by perturbing the future.

And then publish the result you got.

That last step is the hard one — and it's the one that makes the other six worth doing.

That's the twelfth and last project. Everything is in the repository: the code, the
artifacts, the audit, and the numbers behind every claim in all twelve of these videos.
Thanks for watching.
---

*Generated by [`tools/video_scripts.py`](../../tools/video_scripts.py) from committed
artifacts. Every figure spoken here is the figure on screen.*
