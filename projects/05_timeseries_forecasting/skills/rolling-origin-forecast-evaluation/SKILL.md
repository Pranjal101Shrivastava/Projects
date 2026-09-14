---
name: rolling-origin-forecast-evaluation
description: Compare forecasters across series without fooling yourself with one split
---

# Compare forecasters across series without fooling yourself with one split

*Derived from [`projects/05_timeseries_forecasting`](../../), which applied this procedure to: International Airline Passengers (1949-1960), Daily Minimum Temperatures, Melbourne (1981-1990), Antidiabetic Drug Sales, Australia (1991-2008), Sunspot Area (1875-2015).*

This is a **transferable procedure**, not instructions for rerunning the script in this
repository. It states what to do on a new dataset, where the judgement calls are, and which
failure modes this project actually hit.

## When to use this

You need to choose a forecasting method and want the choice to generalise beyond the one
series you tested on.

## Procedure

### 1. Use more than one series, chosen for structural diversity

One series tells you which model fits that series. Pick series that differ in *structure*:
trend with calendar seasonality, seasonality without trend, and — importantly — at least one
whose periodicity is **not calendar-anchored**.

Approximate, drifting cycles (solar, economic, epidemic, fashion) defeat every method that
assumes a fixed period. Without one in your set you will over-recommend seasonal models.

### 2. Use MASE, not MAPE or raw RMSE

    MASE = mean(|y − ŷ|) / mean(|y_t − y_{t−m}|)

Unit-free and anchored: 1.0 means "matched seasonal naive". This is the only sane way to
average across series with different units.

MAPE is undefined at zero, explosive near it, and asymmetric between over- and
under-forecasts.

### 3. Roll the origin; refit at each one

```python
for origin in origins:
    train, actual = values[:origin], values[origin:origin + h]
    prediction = model_fn(train, h)          # refit on this origin's data only
```

A single split reports where one arbitrary cut fell. Report the mean *and the spread* across
origins.

### 4. Use direct multi-step for ML forecasters

Recursive forecasting feeds a model its own predictions, so error compounds and the far
horizon becomes predictions of predictions. Train h separate models instead.

### 5. Run ADF and KPSS, and interpret them jointly

Their null hypotheses are opposites. A single test that fails to reject is ambiguous between
"the null holds" and "the test is underpowered". Together they distinguish trend-stationary
(detrend) from difference-stationary (difference).

Expect conflicts. Report them rather than picking whichever agreed with your expectation.

### 6. Report the series where nothing works

If a naive baseline wins on one of your series, that is the most informative row in the table.

## Decision points requiring judgement

**High-frequency seasonality.** A seasonal model at m=365 must estimate 365 seasonal lags —
intractable and unidentifiable. Aggregate to a coarser frequency that preserves the cycle.
Setting m=7 instead, silently, models a cycle that may not exist.

**Fixed hyperparameters favour classical methods.** Note it as a limitation rather than
claiming a fair fight.

---

## Reproduce the reference implementation

```bash
export PYTHONPATH=lib
python3 projects/05_timeseries_forecasting/pipeline/build.py
python3 tools/audit.py
```

Results, with their run provenance, land in
[`projects/05_timeseries_forecasting/artifacts/`](../../artifacts/).
