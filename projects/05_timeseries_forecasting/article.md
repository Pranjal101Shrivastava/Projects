# I benchmarked six forecasters on four series and got four different answers

*Including one where a forty-year-old method lost to doing nothing*

If you benchmark forecasting models on one time series, you will learn which model fits that
series. You will not learn which model to reach for next time. Those are different questions,
and the literature routinely conflates them.

So I picked four real series that differ in *structure*, not just in subject:

- **International Airline Passengers** — upward trend with multiplicative annual seasonality
- **Melbourne Minimum Temperature (monthly mean)** — strong annual seasonality, no trend
- **Australian Antidiabetic Drug Sales** — trend with a sharp December spike
- **Sunspot Area** — ~11 year cycle, not calendar-anchored

Then I ran six forecasters on all four, refitting at every rolling origin.

## The headline is boring. The exception is not.

SARIMA won three of four series. Mean rank 1.50.
If I stopped there, the conclusion would be "use SARIMA" and the article would be forgettable.

Here is the fourth series.

| Sunspot area | MASE |
|---|---:|
| Seasonal naive | 1.501 |
| LightGBM direct multi-step | 1.692 |
| SARIMA(1,1,1)(1,1,1,m) | 1.797 |
| Holt-Winters (add/add) | 1.841 |
| Naive (last value) | 2.408 |
| Drift | 2.425 |

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

It won **0 series**.

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

**Live demo:** [https://pranjal101shrivastava.github.io/Projects/#/p/forecasting](https://pranjal101shrivastava.github.io/Projects/#/p/forecasting)
**Code:** [github.com/Pranjal101Shrivastava/Projects](https://github.com/Pranjal101Shrivastava/Projects)
