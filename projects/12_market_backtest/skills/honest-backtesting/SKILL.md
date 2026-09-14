---
name: honest-backtesting
description: Evaluate a trading or any forward-looking model so that a negative result survives
---

# Evaluate a trading or any forward-looking model so that a negative result survives

*Derived from [`projects/12_market_backtest`](../../), which applied this procedure to: Apple Inc. Daily OHLCV (Feb 2015 - Feb 2017).*

This is a **transferable procedure**, not instructions for rerunning the script in this
repository. It states what to do on a new dataset, where the judgement calls are, and which
failure modes this project actually hit.

## When to use this

You are evaluating a model whose predictions would drive sequential decisions over time —
trading is the canonical case, but this applies to any forward-looking model where being
wrong is expensive and the temptation to flatter yourself is strong.

## Procedure

### 1. Check stationarity before choosing a target

Run an ADF test on both the level and the difference. If the level has a unit root, modelling
it directly produces enormous R² values that measure nothing but the target's own persistence.
This is the single most common way results in this domain are overstated.

The diagnostic: restate your model as a forecast of the *change*. If R² collapses from 0.97 to
0.00, you were never predicting anything.

### 2. Measure signal before you measure money

Compute the information coefficient — rank correlation between prediction and realised
outcome — with a p-value, before building any decision rule. An equity curve derived from
predictions with no measurable signal is arithmetic on noise, and it will occasionally look
excellent by chance.

### 3. Purge overlapping labels out of the CV boundary

A k-day forward return computed daily means consecutive rows share k−1 days of outcome.
Adjacent rows are nearly the same observation. Chronological ordering alone does not fix this:
remove a band of rows around each train/test boundary.

### 4. Prove the absence of look-ahead, do not assert it

Static rules cannot see through a `.pipe()` boundary or a helper function. Perturb the future
instead: multiply all inputs from row N onward by some factor, rebuild the entire feature
matrix, and assert that no feature value at any row before N changed. Maximum drift should be
exactly zero.

Bake this into the pipeline so it runs on every execution, not once during development.

### 5. Charge costs, and report gross beside net

State the cost assumption explicitly and compute the break-even: a strategy trading n times a
year at c bps needs n·c of gross alpha before it earns anything. At daily frequency costs are
usually the whole result.

### 6. Benchmark against doing nothing

Not against zero. Buy-and-hold, or the constant prediction, or the current process. A strategy
that returns 8% in a year the benchmark returned 20% lost.

### 7. Publish the result you got

This is the hard step, and it is the one that makes every step above worth doing.

## Decision points requiring judgement

**How many models did you try?** Every additional model family and feature set is another
hypothesis. If something comes out significant, the p-value needs correcting for all of them —
including the ones you abandoned.

**Instrument selection is a hindsight decision.** Anything with a long liquid history survived
to be chosen. That biases the benchmark upward and makes it harder to beat, which is worth
stating explicitly rather than hoping nobody notices.

## Failure modes this project hit

**A stunning R² that was the target's non-stationarity.** Yesterday's price predicts today's
price with R² ≈ 0.98. The identical model as a return forecast scores ≈ 0.00.

**Fold-to-fold variance wide enough to support any conclusion.** Directional accuracy swung
across a 30-point range between folds of the same model. Any single fold could have been
quoted as a triumph. Fixing the protocol in advance is what makes the average meaningful.

---

## Reproduce the reference implementation

```bash
export PYTHONPATH=lib
python3 projects/12_market_backtest/pipeline/build.py
python3 tools/audit.py
```

Results, with their run provenance, land in
[`projects/12_market_backtest/artifacts/`](../../artifacts/).
