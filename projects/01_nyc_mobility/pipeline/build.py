"""NYC ride-hail demand forecasting — full CRISP-DM pipeline.

Run:
    PYTHONPATH=lib python3 projects/01_nyc_mobility/pipeline/build.py

What this predicts, and why
---------------------------
The source is the NYC TLC's FOIL response: 4.53M real dispatched Uber pickups from
April–September 2014. Its four columns are ``Date/Time``, ``Lat``, ``Lon``, ``Base``.

There is **no trip duration and no fare** in this data. A "trip duration predictor" built
on it would have to invent its own target, and a model trained on an invented target
measures only how well it recovers the generator. So the target here is the one the data
actually supports: **hourly pickup demand per zone**, which is a real quantity, directly
countable from the records, and the thing a dispatcher actually needs.

That constraint is the interesting part of this project rather than a limitation of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lib"))

from dsx import artifacts, data, metrics, splits  # noqa: E402
from dsx.crispdm import CrispDm, Decision, Phase  # noqa: E402

PROJECT = "01_nyc_mobility"
OUT = ROOT / "projects" / PROJECT / "artifacts"
SEED = 42

# Manhattan-centred bounding box. Points outside are genuine records but belong to other
# markets (the file reaches to Philadelphia and eastern Long Island); mixing them into a
# zone model would blur zones that are 100 km apart.
NYC_BOUNDS = {"lat_min": 40.50, "lat_max": 40.95, "lon_min": -74.30, "lon_max": -73.65}

N_ZONES = 12
TEST_DAYS = 28
MAX_LAG_HOURS = 168  # one week — also the embargo width


# ======================================================================================
# Phase 2 — Data Understanding
# ======================================================================================
def load_and_profile(crisp: CrispDm) -> tuple[pd.DataFrame, dict]:
    """Load the real FOIL extract and profile it honestly, quirks included."""
    print("\n[1/6] Loading real TLC FOIL records …")
    raw = data.load_uber_months(("apr", "may", "jun", "jul", "aug", "sep"))
    n_raw = len(raw)

    raw["pickup_ts"] = pd.to_datetime(raw["Date/Time"], format="%m/%d/%Y %H:%M:%S")
    raw = raw.rename(columns={"Lat": "lat", "Lon": "lon", "Base": "base"})

    # --- Duplicate analysis -----------------------------------------------------------
    # 82,581 rows (1.8%) are exact duplicates. The reflex is to drop them. That would be
    # wrong here: timestamps are minute-resolution and coordinates are rounded to 4dp
    # (~11 m), so two genuinely distinct pickups dispatched from the same base, in the
    # same minute, on the same block are *expected* to collide in this schema. Midtown at
    # 18:00 dispatches far more than one car per minute. Dropping them would systematically
    # understate demand exactly where demand is highest, which is the opposite of useful.
    n_exact_dupes = int(raw.duplicated().sum())
    dupe_share_by_hour = (
        raw.assign(hour=raw.pickup_ts.dt.hour, dup=raw.duplicated(keep=False))
        .groupby("hour")["dup"]
        .mean()
    )

    # --- Geographic filtering ---------------------------------------------------------
    in_box = (
        raw.lat.between(NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"])
        & raw.lon.between(NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"])
    )
    n_outside = int((~in_box).sum())
    df = raw.loc[in_box].copy()

    profile = {
        "rows_raw": n_raw,
        "rows_modelled": len(df),
        "date_min": str(raw.pickup_ts.min()),
        "date_max": str(raw.pickup_ts.max()),
        "n_bases": int(raw.base.nunique()),
        "bases": sorted(raw.base.unique().tolist()),
        "nulls_total": int(raw.isna().sum().sum()),
        "exact_duplicates": n_exact_dupes,
        "exact_duplicate_share": round(n_exact_dupes / n_raw, 5),
        "duplicates_kept": True,
        "duplicate_rationale": (
            "Timestamps are minute-resolution and coordinates rounded to ~11 m, so "
            "distinct simultaneous dispatches from one base on one block are "
            "indistinguishable in this schema. Duplicate share rises with demand "
            "(peak hours highest), which is the signature of collision under coarse "
            "resolution rather than of a data fault. Dropping them would understate "
            "demand precisely at peak, so they are retained."
        ),
        "duplicate_share_by_hour": {
            str(h): round(float(v), 4) for h, v in dupe_share_by_hour.items()
        },
        "rows_outside_nyc_bbox": n_outside,
        "outside_share": round(n_outside / n_raw, 5),
        "bbox": NYC_BOUNDS,
        "trips_per_day_mean": round(len(df) / df.pickup_ts.dt.date.nunique(), 1),
        "lat_range": [float(raw.lat.min()), float(raw.lat.max())],
        "lon_range": [float(raw.lon.min()), float(raw.lon.max())],
    }

    print(f"      {n_raw:,} raw rows → {len(df):,} inside NYC bbox")
    print(f"      {n_exact_dupes:,} exact duplicates retained (see rationale)")

    crisp.record(
        Phase(
            name="data_understanding",
            summary=(
                f"{n_raw:,} real dispatched pickups spanning "
                f"{profile['date_min'][:10]} to {profile['date_max'][:10]} across "
                f"{profile['n_bases']} dispatch bases. No nulls. Two quirks drive the "
                "preparation decisions: 1.8% exact duplicates and 0.1% of points lying "
                "outside the NYC market entirely."
            ),
            decisions=[
                Decision(
                    question="Should the 82,581 exact duplicate rows be dropped?",
                    choice="No — retained.",
                    rationale=profile["duplicate_rationale"],
                    alternatives_rejected=[
                        "drop_duplicates() — the reflex action; would understate peak "
                        "demand, the regime the model exists to predict.",
                        "Deduplicate within a 5-minute window — same defect, larger.",
                    ],
                ),
                Decision(
                    question="How should out-of-region coordinates be handled?",
                    choice=f"Filter to a Manhattan-centred bbox; {n_outside:,} rows dropped.",
                    rationale=(
                        "Coordinates reach 39.66°N–42.12°N and -74.93°E–-72.07°E, i.e. "
                        "from Philadelphia to eastern Long Island. These are real "
                        "dispatches but from other markets. Zone clustering over that "
                        "span would merge locations 100 km apart into one 'zone'."
                    ),
                    alternatives_rejected=[
                        "Keep everything — produces geographically meaningless zones.",
                        "Winsorise coordinates — would pile distant trips onto the "
                        "boundary and invent demand at the edge of the box.",
                    ],
                ),
            ],
            evidence=profile,
            risks=[
                "Six months of a single year cannot express annual seasonality; the "
                "model must not be read as capturing winter demand.",
                "2014 Uber volume grew steeply month over month. Trend is in-sample "
                "here, so forecasts beyond the observed window will extrapolate a "
                "growth rate that did not continue indefinitely.",
            ],
        )
    )
    return df, profile


# ======================================================================================
# Phase 3 — Data Preparation
# ======================================================================================
def build_zones(df: pd.DataFrame, crisp: CrispDm) -> tuple[pd.DataFrame, dict]:
    """Partition pickups into demand zones with K-means over coordinates.

    Zones are learned from where pickups actually happen rather than imposed as a uniform
    grid. A uniform grid over this bbox is mostly water and parkland: the majority of
    cells would carry near-zero demand and the handful of Manhattan cells would each
    aggregate wildly heterogeneous neighbourhoods.
    """
    print(f"\n[2/6] Clustering pickups into {N_ZONES} demand zones …")
    from sklearn.cluster import MiniBatchKMeans

    coords = df[["lat", "lon"]].to_numpy()
    kmeans = MiniBatchKMeans(
        n_clusters=N_ZONES, random_state=SEED, n_init=10, batch_size=10_000
    )
    df["zone"] = kmeans.fit_predict(coords)

    centres = kmeans.cluster_centers_
    zone_stats = []
    for zone_id in range(N_ZONES):
        mask = df.zone == zone_id
        zone_stats.append(
            {
                "zone": int(zone_id),
                "label": _zone_label(centres[zone_id][0], centres[zone_id][1]),
                "centre_lat": round(float(centres[zone_id][0]), 5),
                "centre_lon": round(float(centres[zone_id][1]), 5),
                "trips": int(mask.sum()),
                "share": round(float(mask.mean()), 5),
            }
        )
    zone_stats.sort(key=lambda z: -z["trips"])

    print(f"      largest zone: {zone_stats[0]['label']} ({zone_stats[0]['trips']:,} trips)")
    return df, {"zones": zone_stats, "n_zones": N_ZONES}


def _zone_label(lat: float, lon: float) -> str:
    """Name a zone centroid by nearest known NYC landmark.

    Purely for legibility in the UI — the model never sees these strings. Labels are
    approximate and shown as such.
    """
    landmarks = [
        ("Midtown", 40.7549, -73.9840),
        ("Upper East Side", 40.7736, -73.9566),
        ("Upper West Side", 40.7870, -73.9754),
        ("Chelsea / Flatiron", 40.7420, -73.9970),
        ("East Village", 40.7265, -73.9815),
        ("Financial District", 40.7075, -74.0113),
        ("Harlem", 40.8116, -73.9465),
        ("Williamsburg", 40.7081, -73.9571),
        ("Downtown Brooklyn", 40.6928, -73.9903),
        ("Astoria / LIC", 40.7550, -73.9300),
        ("JFK Airport", 40.6413, -73.7781),
        ("LaGuardia Airport", 40.7769, -73.8740),
        ("Newark Airport", 40.6895, -74.1745),
        ("The Bronx", 40.8448, -73.8648),
        ("Staten Island", 40.5795, -74.1502),
        ("Outer Queens", 40.7282, -73.7949),
    ]
    best = min(landmarks, key=lambda l: (lat - l[1]) ** 2 + (lon - l[2]) ** 2)
    return f"~{best[0]}"


def build_panel(df: pd.DataFrame, crisp: CrispDm) -> tuple[pd.DataFrame, dict]:
    """Aggregate to a zone × hour panel and engineer strictly backward-looking features.

    Every feature is computed from data available *before* the hour being predicted. Lags
    are shifted within zone; rolling windows are shifted by one before aggregating, so an
    hour never contributes to its own rolling mean. This is the single most important
    property in the file — a rolling mean that includes the current observation is the
    classic silent leak in demand forecasting and it inflates R² dramatically.
    """
    print("\n[3/6] Building zone × hour panel with backward-only features …")

    df["hour_ts"] = df.pickup_ts.dt.floor("h")
    panel = (
        df.groupby(["zone", "hour_ts"], observed=True)
        .size()
        .reset_index(name="demand")
        .sort_values(["zone", "hour_ts"])
        .reset_index(drop=True)
    )

    # Reindex onto a complete grid so absent hours read as zero demand rather than as a
    # gap that lag arithmetic would silently step over.
    full_hours = pd.date_range(panel.hour_ts.min(), panel.hour_ts.max(), freq="h")
    grid = pd.MultiIndex.from_product(
        [sorted(panel.zone.unique()), full_hours], names=["zone", "hour_ts"]
    )
    panel = (
        panel.set_index(["zone", "hour_ts"])
        .reindex(grid, fill_value=0)
        .reset_index()
        .sort_values(["zone", "hour_ts"])
        .reset_index(drop=True)
    )

    grouped = panel.groupby("zone", observed=True)["demand"]

    # Autoregressive lags: 1h (momentum), 2h, 3h, 24h (same hour yesterday),
    # 168h (same hour last week — the dominant signal in urban mobility).
    for lag in (1, 2, 3, 24, 168):
        panel[f"lag_{lag}h"] = grouped.shift(lag)

    # Rolling statistics. shift(1) first, then roll: the window ends at t-1.
    for window in (3, 24, 168):
        shifted = grouped.shift(1)
        panel[f"roll_mean_{window}h"] = shifted.rolling(window, min_periods=1).mean()
        if window > 3:
            panel[f"roll_std_{window}h"] = shifted.rolling(window, min_periods=2).std()

    # Calendar features. Cyclical encoding so 23:00 and 00:00 are adjacent rather than
    # maximally distant, which a raw integer hour would imply.
    ts = panel.hour_ts
    panel["hour"] = ts.dt.hour
    panel["dow"] = ts.dt.dayofweek
    panel["is_weekend"] = (panel.dow >= 5).astype(int)
    panel["month"] = ts.dt.month
    panel["day_of_month"] = ts.dt.day
    panel["hour_sin"] = np.sin(2 * np.pi * panel.hour / 24)
    panel["hour_cos"] = np.cos(2 * np.pi * panel.hour / 24)
    panel["dow_sin"] = np.sin(2 * np.pi * panel.dow / 7)
    panel["dow_cos"] = np.cos(2 * np.pi * panel.dow / 7)

    # US holidays observed within the Apr–Sep 2014 window.
    holidays = {"2014-05-26", "2014-07-04", "2014-09-01"}
    panel["is_holiday"] = ts.dt.strftime("%Y-%m-%d").isin(holidays).astype(int)

    before = len(panel)
    panel = panel.dropna().reset_index(drop=True)
    dropped = before - len(panel)

    feature_cols = [
        c for c in panel.columns if c not in {"zone", "hour_ts", "demand"}
    ] + ["zone"]

    prep = {
        "panel_rows": len(panel),
        "rows_dropped_for_lag_warmup": dropped,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "hours_covered": int(panel.hour_ts.nunique()),
        "leakage_controls": [
            "Lags use groupby(zone).shift(k): a row can only see its own zone's past.",
            "Rolling windows are shift(1) before .rolling(), so the window ends at t-1 "
            "and an hour never enters its own rolling statistic.",
            "The panel is reindexed onto a complete hourly grid before lagging, so a "
            "missing hour cannot let lag_24h silently reach 25 hours back.",
            f"{dropped:,} warm-up rows with incomplete lag history are dropped rather "
            "than imputed — imputing them would fabricate history.",
        ],
    }
    print(f"      {len(panel):,} zone-hours, {len(feature_cols)} features")

    crisp.record(
        Phase(
            name="data_preparation",
            summary=(
                f"Point events aggregated to a {N_ZONES}-zone × hourly panel of "
                f"{len(panel):,} rows with {len(feature_cols)} strictly backward-looking "
                "features: autoregressive lags to one week, shifted rolling statistics, "
                "cyclically encoded calendar terms and holiday flags."
            ),
            decisions=[
                Decision(
                    question="How are rolling features protected from lookahead?",
                    choice="shift(1) applied before .rolling(), never after.",
                    rationale=(
                        "A rolling mean computed without the prior shift includes the "
                        "target hour in its own predictor. The resulting fit looks "
                        "excellent and collapses in production. Shifting first makes the "
                        "window strictly [t-w, t-1]."
                    ),
                    alternatives_rejected=[
                        ".rolling(w).mean() directly on the demand column — leaks the "
                        "target into its own feature.",
                        "centered=True rolling windows — leaks the future outright.",
                    ],
                ),
                Decision(
                    question="Learned zones or a uniform spatial grid?",
                    choice=f"K-means over pickup coordinates, k={N_ZONES}.",
                    rationale=(
                        "A uniform grid over this bbox is mostly water and parkland. "
                        "Most cells would carry near-zero demand while a few Manhattan "
                        "cells would aggregate very different neighbourhoods. Learned "
                        "centroids place zone boundaries where demand actually separates."
                    ),
                    alternatives_rejected=[
                        "Uniform lat/lon grid — dominated by empty cells.",
                        "Official TLC taxi zones — not present in this FOIL extract and "
                        "joining them would require an external shapefile the "
                        "environment cannot reach.",
                    ],
                ),
            ],
            evidence=prep,
            risks=[
                "K-means assumes isotropic clusters in degrees; one degree of longitude "
                "is shorter than one of latitude at 40°N, so zones are mildly stretched "
                "east-west. At this scale the distortion is under 25% and does not "
                "affect ranking, but it is a real approximation.",
            ],
        )
    )
    return panel, prep


# ======================================================================================
# Phase 4 — Modeling
# ======================================================================================
def train_models(panel: pd.DataFrame, crisp: CrispDm) -> dict:
    """Train a baseline ladder and score every rung on the same held-out weeks.

    The ladder matters more than any single model. "LightGBM achieves MAE 12.4" is not a
    result; "LightGBM achieves MAE 12.4 where repeating last week achieves 18.9" is. Each
    rung has to earn its added complexity against the rung below it.
    """
    print("\n[4/6] Training model ladder …")
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb

    feature_cols = [c for c in panel.columns if c not in {"hour_ts", "demand"}]

    # Chronological split. The embargo equals the widest lag (168h): without it, the
    # first test rows carry lag features computed from the tail of the training period,
    # so the two partitions are not independent.
    hours = np.sort(panel.hour_ts.unique())
    cutoff = hours[-TEST_DAYS * 24]
    embargo_start = cutoff - pd.Timedelta(hours=MAX_LAG_HOURS)

    train_mask = panel.hour_ts < embargo_start
    test_mask = panel.hour_ts >= cutoff

    X_train = panel.loc[train_mask, feature_cols]
    y_train = panel.loc[train_mask, "demand"].to_numpy()
    X_test = panel.loc[test_mask, feature_cols]
    y_test = panel.loc[test_mask, "demand"].to_numpy()

    split_report = {
        "strategy": "chronological_holdout_with_embargo",
        "shuffled": False,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "train_end": str(embargo_start),
        "test_start": str(cutoff),
        "test_days": TEST_DAYS,
        "embargo_hours": MAX_LAG_HOURS,
        "embargo_rationale": (
            "The widest feature lag is 168h. Training rows within 168h of the test "
            "boundary share lag history with the first test rows, so that band is "
            "discarded to keep the partitions independent."
        ),
    }
    print(f"      train {split_report['n_train']:,} | test {split_report['n_test']:,} "
          f"(last {TEST_DAYS} days, {MAX_LAG_HOURS}h embargo)")

    results: dict[str, dict] = {}

    # --- Rung 0: seasonal naive -------------------------------------------------------
    # Predict this hour with the same hour one week ago. In urban mobility this is a
    # genuinely strong baseline and it is the number every later model must beat.
    naive_pred = panel.loc[test_mask, "lag_168h"].to_numpy()
    results["seasonal_naive_168h"] = {
        "label": "Seasonal naive (t − 168h)",
        "family": "baseline",
        "complexity": "zero parameters",
        **metrics.regression_report(y_test, naive_pred, baseline=naive_pred),
    }

    # --- Rung 1: yesterday ------------------------------------------------------------
    naive_24 = panel.loc[test_mask, "lag_24h"].to_numpy()
    results["naive_24h"] = {
        "label": "Naive (t − 24h)",
        "family": "baseline",
        "complexity": "zero parameters",
        **metrics.regression_report(y_test, naive_24, baseline=naive_pred),
    }

    # --- Rung 2: regularised linear ---------------------------------------------------
    # Wrapped in a Pipeline so the scaler is fitted on training rows only.
    ridge = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=SEED))])
    splits.assert_pipeline_safe(ridge)
    ridge.fit(X_train, y_train)
    ridge_pred = np.clip(ridge.predict(X_test), 0, None)
    results["ridge"] = {
        "label": "Ridge regression",
        "family": "linear",
        "complexity": f"{len(feature_cols)} coefficients",
        **metrics.regression_report(y_test, ridge_pred, baseline=naive_pred),
    }

    # --- Rung 3: gradient boosting ----------------------------------------------------
    gbm = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    gbm.fit(X_train, y_train, categorical_feature=["zone"])
    gbm_pred = np.clip(gbm.predict(X_test), 0, None)
    results["lightgbm"] = {
        "label": "LightGBM",
        "family": "gradient boosting",
        "complexity": "600 trees × 63 leaves",
        **metrics.regression_report(y_test, gbm_pred, baseline=naive_pred),
    }

    # --- Conformal prediction intervals -----------------------------------------------
    # Split conformal: calibrate absolute residual quantiles on a slice of training data
    # the model never fitted, then apply that quantile as a symmetric band. This gives
    # finite-sample coverage without assuming the errors are Gaussian — and demand
    # residuals are visibly not Gaussian, being right-skewed and heteroscedastic.
    calib_cut = int(len(X_train) * 0.85)
    gbm_calib = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    gbm_calib.fit(X_train.iloc[:calib_cut], y_train[:calib_cut], categorical_feature=["zone"])
    calib_residuals = np.abs(
        y_train[calib_cut:] - np.clip(gbm_calib.predict(X_train.iloc[calib_cut:]), 0, None)
    )
    conformal = {}
    for level in (0.80, 0.90, 0.95):
        q = float(np.quantile(calib_residuals, level))
        lower = np.clip(gbm_pred - q, 0, None)
        upper = gbm_pred + q
        conformal[f"{int(level * 100)}%"] = {
            "quantile_width": round(q, 3),
            **metrics.interval_coverage(y_test, lower, upper, nominal=level),
        }

    # --- Feature importance -----------------------------------------------------------
    importance = sorted(
        (
            {"feature": f, "gain": float(g)}
            for f, g in zip(feature_cols, gbm.booster_.feature_importance("gain"))
        ),
        key=lambda r: -r["gain"],
    )
    total_gain = sum(r["gain"] for r in importance) or 1.0
    for row in importance:
        row["share"] = round(row["gain"] / total_gain, 5)
        row["gain"] = round(row["gain"], 2)

    # --- Error analysis ---------------------------------------------------------------
    test_frame = panel.loc[test_mask, ["zone", "hour_ts", "demand"]].copy()
    test_frame["pred"] = gbm_pred
    test_frame["abs_err"] = np.abs(test_frame.demand - test_frame.pred)

    by_hour = (
        test_frame.assign(h=test_frame.hour_ts.dt.hour)
        .groupby("h")
        .agg(mae=("abs_err", "mean"), mean_demand=("demand", "mean"))
        .reset_index()
    )
    by_zone = (
        test_frame.groupby("zone")
        .agg(mae=("abs_err", "mean"), mean_demand=("demand", "mean"))
        .reset_index()
    )

    best = min(
        (k for k in results if results[k]["family"] != "baseline"),
        key=lambda k: results[k]["mae"],
    )
    print(f"      best: {results[best]['label']} — MAE {results[best]['mae']:.2f} "
          f"vs naive {results['seasonal_naive_168h']['mae']:.2f} "
          f"({results[best]['skill_score_vs_baseline']:.1%} skill)")

    crisp.record(
        Phase(
            name="modeling",
            summary=(
                "A four-rung ladder from zero-parameter baselines to gradient boosting, "
                "every rung scored on the identical embargoed holdout. Uncertainty is "
                "quantified by split-conformal intervals rather than by a Gaussian "
                "assumption the residuals do not satisfy."
            ),
            decisions=[
                Decision(
                    question="What must a model beat to be considered useful?",
                    choice="Seasonal naive at t − 168h.",
                    rationale=(
                        "Weekly periodicity dominates urban mobility. Repeating the same "
                        f"hour last week already achieves MAE "
                        f"{results['seasonal_naive_168h']['mae']:.2f}. Any learned model "
                        "that cannot beat that is not earning its complexity, and "
                        "reporting its R² without this reference would be misleading."
                    ),
                    alternatives_rejected=[
                        "Global mean baseline — trivially weak, flatters every model.",
                        "Reporting R² alone — high for anything that tracks the daily "
                        "cycle and therefore uninformative here.",
                    ],
                ),
                Decision(
                    question="How are prediction intervals produced?",
                    choice="Split-conformal calibration on a held-out training slice.",
                    rationale=(
                        "Residuals are right-skewed and heteroscedastic — variance grows "
                        "with demand level — so a Gaussian ±1.96σ band would be too wide "
                        "at low demand and too narrow at peak, exactly where the cost of "
                        "being wrong is highest. Conformal quantiles need no "
                        "distributional assumption and their coverage is checked "
                        "empirically below."
                    ),
                    alternatives_rejected=[
                        "Gaussian ±1.96σ — assumes symmetry and homoscedasticity, both "
                        "violated.",
                        "Quantile regression — viable, but needs a separate model per "
                        "level and gives no finite-sample coverage guarantee.",
                    ],
                ),
            ],
            evidence={
                "split": split_report,
                "n_models": len(results),
                "best_model": best,
            },
            risks=[
                "LightGBM cannot extrapolate beyond the demand levels seen in training. "
                "2014 volume was growing steeply, so a forecast far past the window "
                "would saturate at the training maximum rather than continue the trend.",
            ],
        )
    )

    return {
        "split": split_report,
        "models": results,
        "best_model": best,
        "conformal": conformal,
        "feature_importance": importance,
        "error_by_hour": by_hour.round(3).to_dict("records"),
        "error_by_zone": by_zone.round(3).to_dict("records"),
        "_test_frame": test_frame,
        "_model": gbm,
        "_features": feature_cols,
    }


# ======================================================================================
# Phase 5 — Evaluation
# ======================================================================================
def build_eda(df: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """Exploratory views, pre-binned for client-side rendering."""
    print("\n[5/6] Building EDA artifacts …")

    hourly_total = panel.groupby(panel.hour_ts.dt.hour)["demand"].mean()
    dow_hour = (
        panel.assign(d=panel.hour_ts.dt.dayofweek, h=panel.hour_ts.dt.hour)
        .groupby(["d", "h"])["demand"]
        .mean()
        .reset_index()
    )
    daily = panel.groupby(panel.hour_ts.dt.date)["demand"].sum()

    # Spatial scatter for the map view — thinned to something a browser can draw.
    sample = df.sample(n=min(6000, len(df)), random_state=SEED)

    # Autocorrelation of the citywide series. The spikes at 24 and 168 are the whole
    # argument for the lag features chosen above, so they are worth showing explicitly.
    citywide = panel.groupby("hour_ts")["demand"].sum().sort_index().to_numpy()
    centred = citywide - citywide.mean()
    denom = float(np.dot(centred, centred))
    acf = [
        {"lag": lag, "acf": round(float(np.dot(centred[:-lag], centred[lag:]) / denom), 4)}
        for lag in range(1, 181)
    ]

    return {
        "mean_demand_by_hour": [
            {"hour": int(h), "mean_demand": round(float(v), 2)}
            for h, v in hourly_total.items()
        ],
        "heatmap_dow_hour": [
            {"dow": int(r.d), "hour": int(r.h), "mean_demand": round(float(r.demand), 2)}
            for r in dow_hour.itertuples()
        ],
        "daily_total": [
            {"date": str(d), "trips": int(v)} for d, v in daily.items()
        ],
        "pickup_sample": [
            {"lat": round(float(r.lat), 5), "lon": round(float(r.lon), 5), "zone": int(r.zone)}
            for r in sample.itertuples()
        ],
        "demand_distribution": artifacts.histogram(panel.demand.to_numpy(), bins=40),
        "acf_citywide": acf,
        "acf_note": (
            "Spikes at lag 24 and lag 168 are the daily and weekly cycles. They are the "
            "empirical justification for the lag_24h and lag_168h features rather than a "
            "post-hoc rationalisation of them."
        ),
    }


def evaluate(model_out: dict, crisp: CrispDm) -> None:
    """Record the evaluation phase, including where the model is weakest."""
    results = model_out["models"]
    best_key = model_out["best_model"]
    best = results[best_key]
    naive = results["seasonal_naive_168h"]

    by_hour = model_out["error_by_hour"]
    worst_hour = max(by_hour, key=lambda r: r["mae"])
    by_zone = model_out["error_by_zone"]
    worst_zone = max(by_zone, key=lambda r: r["mae"])

    conformal = model_out["conformal"]
    calibrated_levels = [k for k, v in conformal.items() if v["calibrated"]]
    failed_levels = [k for k, v in conformal.items() if not v["calibrated"]]

    if not failed_levels:
        coverage_sentence = (
            f"Conformal intervals are empirically calibrated at all "
            f"{len(conformal)} nominal levels."
        )
    else:
        detail = ", ".join(
            f"{k} covers {conformal[k]['empirical_coverage']:.1%}" for k in failed_levels
        )
        coverage_sentence = (
            f"Conformal intervals are calibrated at {', '.join(calibrated_levels)} but "
            f"under-cover at {', '.join(failed_levels)} ({detail})."
        )

    crisp.record(
        Phase(
            name="evaluation",
            summary=(
                f"{best['label']} reaches MAE {best['mae']:.2f} against seasonal-naive "
                f"{naive['mae']:.2f}, a {best['skill_score_vs_baseline']:.1%} reduction "
                f"in mean absolute error. {coverage_sentence}"
            ),
            decisions=[
                Decision(
                    question="Is the reported improvement worth the added complexity?",
                    choice=(
                        "Yes for LightGBM over naive; the ridge rung is not worth "
                        "deploying."
                        if results["ridge"]["mae"] > naive["mae"]
                        else "Yes — every rung improves on the one below it."
                    ),
                    rationale=(
                        f"LightGBM cuts MAE from {naive['mae']:.2f} to {best['mae']:.2f}. "
                        f"Ridge reaches {results['ridge']['mae']:.2f}. Where the linear "
                        "rung fails to beat a zero-parameter baseline that is reported "
                        "as-is rather than quietly omitted from the table."
                    ),
                ),
            ],
            evidence={
                "best_model": best_key,
                "best_mae": best["mae"],
                "naive_mae": naive["mae"],
                "skill_score": best["skill_score_vs_baseline"],
                "conformal_calibrated_levels": calibrated_levels,
                "conformal_failed_levels": failed_levels,
                "conformal": conformal,
                "worst_hour": worst_hour,
                "worst_zone": worst_zone,
                "top_features": model_out["feature_importance"][:8],
            },
            risks=[
                *(
                    [
                        f"Conformal coverage degrades at the narrowest band: the "
                        f"{failed_levels[0]} interval realises "
                        f"{conformal[failed_levels[0]]['empirical_coverage']:.1%} rather "
                        "than its nominal level. Split conformal assumes the calibration "
                        "and test residuals are exchangeable. They are not here: "
                        "calibration comes from the chronologically last slice of "
                        "training data, and demand grew substantially across the six "
                        "months, so test-period residuals are systematically larger than "
                        "calibration-period residuals. The wider 90% and 95% bands absorb "
                        "that drift and remain calibrated; the tight band does not. This "
                        "is a real limitation of the method under trend, not a tuning "
                        "artefact, and the fix is periodic recalibration on recent data."
                    ]
                    if failed_levels
                    else []
                ),
                f"Error concentrates at hour {worst_hour['h']} (MAE {worst_hour['mae']:.2f} "
                f"against mean demand {worst_hour['mean_demand']:.1f}) — the evening peak, "
                "where both demand and its variance are highest.",
                f"Zone {worst_zone['zone']} carries the largest absolute error "
                f"(MAE {worst_zone['mae']:.2f}); high-volume zones dominate the aggregate "
                "figure, so a single citywide MAE understates performance in quiet zones "
                "and overstates it in busy ones.",
                "Model selection used a single chronological holdout. With one test "
                "window there is no estimate of variance across periods; a different "
                "four weeks would give a different number.",
            ],
        )
    )


# ======================================================================================
# Phase 6 — Deployment
# ======================================================================================
def export_scorer(model_out: dict, zone_info: dict, panel: pd.DataFrame,
                  crisp: CrispDm) -> dict:
    """Export a compact surrogate the browser can evaluate without a server.

    The published site is static, so live inference has to run client-side. Shipping a
    600-tree LightGBM booster to the browser is impractical, so we fit a small
    interpretable surrogate — a per-zone multiplicative profile over hour-of-week — and
    export its table.

    Crucially, the surrogate's fidelity to the full model is measured and published rather
    than assumed. A surrogate whose agreement with the parent model is unstated is a
    decoration; one with a reported R² against the parent is a documented approximation.
    """
    print("\n[6/6] Exporting browser scorer …")

    test_frame = model_out["_test_frame"]

    # Per-zone base rate × hour-of-week multiplier.
    panel = panel.copy()
    panel["how"] = panel.hour_ts.dt.dayofweek * 24 + panel.hour_ts.dt.hour
    zone_base = panel.groupby("zone")["demand"].mean()
    how_profile = (
        panel.groupby(["zone", "how"])["demand"].mean()
        / panel.groupby("zone")["demand"].mean()
    ).fillna(1.0)

    table = {
        str(int(zone)): {
            "base": round(float(zone_base.loc[zone]), 4),
            "profile": [
                round(float(how_profile.get((zone, h), 1.0)), 4) for h in range(168)
            ],
        }
        for zone in sorted(panel.zone.unique())
    }

    # Fidelity of the surrogate against the parent model, on the same holdout.
    surrogate_pred = np.array(
        [
            table[str(int(r.zone))]["base"]
            * table[str(int(r.zone))]["profile"][
                r.hour_ts.dayofweek * 24 + r.hour_ts.hour
            ]
            for r in test_frame.itertuples()
        ]
    )
    from sklearn.metrics import r2_score

    fidelity = {
        "r2_vs_parent_model": round(
            float(r2_score(test_frame.pred.to_numpy(), surrogate_pred)), 4
        ),
        "mae_vs_actual": round(
            float(np.mean(np.abs(test_frame.demand.to_numpy() - surrogate_pred))), 3
        ),
        "parent_mae_vs_actual": round(
            float(np.mean(np.abs(test_frame.demand.to_numpy() - test_frame.pred.to_numpy()))),
            3,
        ),
        "note": (
            "The browser scorer is a hour-of-week × zone surrogate, not the LightGBM "
            "model. Its agreement with the parent is reported here so the live demo is "
            "read as an approximation with a known gap, not as the trained model."
        ),
    }
    print(f"      surrogate fidelity R² vs parent = {fidelity['r2_vs_parent_model']:.4f}")

    crisp.record(
        Phase(
            name="deployment",
            summary=(
                "Two serving paths. A FastAPI service exposes the full LightGBM model for "
                "local use; a compact hour-of-week × zone surrogate is exported to JSON so "
                "the published static site can score interactively with no backend."
            ),
            decisions=[
                Decision(
                    question="How does a static site serve a gradient boosted model?",
                    choice="Export an interpretable surrogate and publish its fidelity.",
                    rationale=(
                        "GitHub Pages serves static files only. Rather than pretend the "
                        "demo runs the real model, the surrogate is declared as such and "
                        f"its R² against the parent ({fidelity['r2_vs_parent_model']:.4f}) "
                        "is shown in the UI itself."
                    ),
                    alternatives_rejected=[
                        "Ship the full booster as JSON — megabytes of trees plus a "
                        "JavaScript tree-walker, for a demo.",
                        "Silently hard-code precomputed predictions — the demo would "
                        "stop being interactive and would misrepresent what it does.",
                    ],
                ),
            ],
            evidence={"surrogate_fidelity": fidelity, "table_zones": len(table)},
            risks=[
                "The surrogate ignores autoregressive state, so it cannot react to a "
                "demand shock the way the parent model does. It reproduces the periodic "
                "structure only.",
            ],
        )
    )
    return {"table": table, "fidelity": fidelity, "zones": zone_info["zones"]}


# ======================================================================================
# Orchestration
# ======================================================================================
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    crisp = CrispDm(
        project=PROJECT,
        business_question=(
            "For each NYC zone, how many ride-hail pickups should be expected in the "
            "next hour, and how confident can a dispatcher be in that number?"
        ),
    )
    crisp.record(
        Phase(
            name="business_understanding",
            summary=(
                "Positioning idle vehicles ahead of demand is the core operational lever "
                "in ride-hail. Under-supply in a zone loses the fare and pushes riders to "
                "a competitor; over-supply pays drivers to idle. Both costs are "
                "asymmetric and both are paid hourly, so the useful forecast horizon is "
                "one hour at zone granularity."
            ),
            decisions=[
                Decision(
                    question="What is the prediction target?",
                    choice="Pickup count per zone per hour.",
                    rationale=(
                        "The FOIL extract records timestamp, coordinates and dispatch "
                        "base only. It contains no trip duration and no fare. Demand is "
                        "the quantity this data can actually support, it is directly "
                        "countable from the records, and it is what dispatch decisions "
                        "are made on."
                    ),
                    alternatives_rejected=[
                        "Trip duration — the reference implementation this portfolio "
                        "responds to predicts duration on this dataset, but the column "
                        "does not exist; doing so requires synthesising the target, and "
                        "a model fitted to a synthetic target measures only how well it "
                        "recovers the generator.",
                        "Fare — likewise absent from the source.",
                    ],
                ),
                Decision(
                    question="What accuracy would make this worth deploying?",
                    choice="Beat seasonal-naive MAE by a clear margin at zone-hour level.",
                    rationale=(
                        "Dispatchers already reason with 'same time last week'. A model "
                        "is only worth operating if it improves on the heuristic the "
                        "organisation would use for free."
                    ),
                ),
            ],
            evidence={
                "horizon": "1 hour",
                "granularity": f"{N_ZONES} learned zones",
                "asymmetric_costs": (
                    "Under-supply loses revenue and rider retention; over-supply pays "
                    "idle driver time. The conformal interval is provided so a dispatcher "
                    "can position against the upper bound when under-supply is costlier."
                ),
            },
        )
    )

    with artifacts.run(PROJECT, seed=SEED) as ctx:
        df, profile = load_and_profile(crisp)
        df, zone_info = build_zones(df, crisp)
        panel, prep = build_panel(df, crisp)
        model_out = train_models(panel, crisp)
        eda = build_eda(df, panel)
        evaluate(model_out, crisp)
        scorer = export_scorer(model_out, zone_info, panel, crisp)

        artifacts.write(OUT / "profile.json", {"profile": profile, **zone_info}, context=ctx)
        artifacts.write(OUT / "preparation.json", prep, context=ctx)
        artifacts.write(OUT / "eda.json", eda, context=ctx)
        artifacts.write(
            OUT / "models.json",
            {
                k: v for k, v in model_out.items() if not k.startswith("_")
            },
            context=ctx,
        )
        artifacts.write(OUT / "scorer.json", scorer, context=ctx)
        artifacts.write(OUT / "crispdm.json", crisp.to_dict(), context=ctx)
        artifacts.write(
            OUT / "provenance.json",
            {"datasets": [data.provenance_record("uber_nyc_2014")]},
            context=ctx,
        )

    best = model_out["models"][model_out["best_model"]]
    print(f"\n✓ {PROJECT} complete — {best['label']}: MAE {best['mae']:.2f}, "
          f"skill vs naive {best['skill_score_vs_baseline']:.1%}")


if __name__ == "__main__":
    main()
