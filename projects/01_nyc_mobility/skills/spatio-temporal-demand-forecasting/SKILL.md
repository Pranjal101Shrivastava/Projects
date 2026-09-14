---
name: spatio-temporal-demand-forecasting
description: Forecast demand over a spatial grid and a time index, with honest uncertainty
---

# Forecast demand over a spatial grid and a time index, with honest uncertainty

*Derived from [`projects/01_nyc_mobility`](../../), which applied this procedure to: Uber NYC Pickups (TLC FOIL response, Apr-Sep 2014).*

This is a **transferable procedure**, not instructions for rerunning the script in this
repository. It states what to do on a new dataset, where the judgement calls are, and which
failure modes this project actually hit.

## When to use this

You have event records with a timestamp and a location, and you need to forecast how many
events will occur in each region in each time bucket.

## Procedure

### 1. Establish what the data can support before choosing a target

Read the schema before writing the research question. The columns present determine what can
be predicted; the columns you wish were present determine what cannot.

If the target you intended is absent, **do not synthesise it**. A model fitted to a generated
target measures how well it recovers the generator.

### 2. Partition space by where events actually are

K-means over event coordinates, not a uniform grid. A uniform grid over any real
metropolitan bounding box is mostly water, parkland and industrial land; most cells carry
near-zero counts while a handful aggregate wildly different neighbourhoods.

Check the coordinate range first. Real location data routinely contains points from entirely
different markets.

### 3. Build the panel, then reindex it before lagging

```python
panel = events.groupby(["zone", "hour"]).size().reset_index(name="demand")

# Critical: reindex onto a COMPLETE time grid before computing any lag.
full = pd.date_range(panel.hour.min(), panel.hour.max(), freq="h")
grid = pd.MultiIndex.from_product([zones, full], names=["zone", "hour"])
panel = panel.set_index(["zone", "hour"]).reindex(grid, fill_value=0).reset_index()
```

Without the reindex, a missing hour silently causes `lag_24h` to reach twenty-five hours back.

### 4. Shift before rolling. Always.

```python
grouped = panel.groupby("zone")["demand"]

# WRONG — the window includes the value being predicted
panel["roll"] = grouped.rolling(24).mean()

# RIGHT — the window ends at t-1
panel["roll"] = grouped.shift(1).rolling(24).mean()
```

This is the single most common silent leak in demand forecasting. The fit looks excellent and
collapses in production, and no metric reveals it.

### 5. Split chronologically, with an embargo

The embargo width should equal your widest feature lag. Training rows within one lag-width of
the boundary share history with the first test rows.

### 6. Build the baseline ladder before the model

For periodic human activity, **test weekly before daily**. Weekday-versus-weekend is usually
a larger effect than day-to-day drift.

| Rung | Why |
|---|---|
| Seasonal naive at the dominant period | The number every model must beat |
| A second naive at a shorter period | Establishes which periodicity dominates |
| Regularised linear | Is non-linearity earning its complexity? |
| Gradient boosting | Only now |

### 7. Do not report R² alone

On any series with a strong daily cycle, everything scores high R². Report the error metric
and the skill score against the naive baseline.

## Decision points requiring judgement

**Duplicate records.** Check whether duplicate share correlates with the event rate. If it
rises with volume, they are collisions under coarse timestamp or coordinate resolution, not
faults — dropping them understates demand exactly at peak.

**Conformal intervals under trend.** Split conformal assumes calibration and test residuals
are exchangeable. If the series has a trend, calibration residuals drawn from the training
period will be smaller than test residuals and narrow bands will under-cover. Measure realised
coverage; do not assume the nominal level holds.

---

## Reproduce the reference implementation

```bash
export PYTHONPATH=lib
python3 projects/01_nyc_mobility/pipeline/build.py
python3 tools/audit.py
```

Results, with their run provenance, land in
[`projects/01_nyc_mobility/artifacts/`](../../artifacts/).
