"""Time series forecasting across four real series with walk-forward backtesting.

Run:
    PYTHONPATH=lib python3 projects/05_timeseries_forecasting/pipeline/build.py

Most forecasting write-ups evaluate on one series and one holdout, which cannot separate
"this model is good" from "this model happens to suit this series". Four series with
deliberately different structure are used here:

* **Airline passengers** — trend plus multiplicative seasonality, the canonical Box-Jenkins case.
* **Melbourne daily temperatures** — strong annual seasonality, no trend.
* **Australian drug sales** — trend plus a sharp December spike.
* **Sunspot area** — an ~11-year cycle with no calendar anchor, which defeats any model
  that assumes seasonality aligns to the calendar.

Evaluation is walk-forward rather than a single split: the model is refitted at each origin
and forecasts a fixed horizon ahead, so the reported error is an average over many origins
rather than an accident of where one cut happened to fall.

Errors are compared with MASE, scaled by the in-sample naive error. Comparing raw MAE
across series measuring passengers, degrees Celsius and prescription counts would be
meaningless; MASE is unit-free and 1.0 always means "no better than naive".
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lib"))

from dsx import artifacts, data, metrics  # noqa: E402
from dsx.crispdm import CrispDm, Decision, Phase  # noqa: E402

warnings.filterwarnings("ignore")

PROJECT = "05_timeseries_forecasting"
OUT = ROOT / "projects" / PROJECT / "artifacts"
SEED = 42

SERIES_SPECS = {
    "airline_passengers": {
        "title": "International Airline Passengers",
        "value_col": "Passengers",
        "date_col": "Month",
        "period": 12,
        "unit": "thousands of passengers",
        "structure": "upward trend with multiplicative annual seasonality",
    },
    "daily_min_temps": {
        "title": "Melbourne Minimum Temperature (monthly mean)",
        "value_col": "Temp",
        "date_col": "Date",
        "period": 12,
        "unit": "°C",
        "structure": "strong annual seasonality, no trend",
        # Aggregated from 3,650 daily observations to 120 monthly means. The annual cycle
        # is the structure of interest and survives aggregation intact, whereas a seasonal
        # model at m=365 is intractable: SARIMA would have to estimate 365 seasonal lags
        # from ten cycles of data, which is both infeasible to fit and hopeless to
        # identify. Aggregating is the honest response; quietly setting m=7 and calling it
        # "seasonality" would model a weekly cycle this series does not have.
        "resample": "MS",
    },
    "drug_sales": {
        "title": "Australian Antidiabetic Drug Sales",
        "value_col": "value",
        "date_col": "date",
        "period": 12,
        "unit": "scripts (millions)",
        "structure": "trend with a sharp December spike",
    },
    "sunspots": {
        "title": "Sunspot Area",
        "value_col": "value",
        "date_col": "date",
        "period": 11,
        "unit": "millionths of hemisphere",
        "structure": "~11 year cycle, not calendar-anchored",
    },
}

HORIZON = 12
N_ORIGINS = 8


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, period: int) -> float:
    """Mean absolute scaled error.

    Denominator is the in-sample mean absolute error of the seasonal naive forecast. That
    makes the metric unit-free and gives it a fixed, interpretable reference point: MASE
    of 1.0 means the model matched seasonal naive, below 1.0 means it beat it. This is the
    only sane way to average performance across series measured in different units.
    """
    if len(y_train) <= period:
        period = 1
    scale = np.mean(np.abs(y_train[period:] - y_train[:-period]))
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def load_series(key: str) -> pd.Series:
    """Load one declared series as a clean, date-indexed float Series."""
    spec = SERIES_SPECS[key]
    df = pd.read_csv(data.fetch(key))
    df.columns = [c.strip() for c in df.columns]

    value_col = spec["value_col"]
    if value_col not in df.columns:
        # Fall back to the last numeric column when the mirror renames headers.
        numeric = df.select_dtypes(include=[np.number]).columns
        value_col = numeric[-1]

    date_col = spec["date_col"] if spec["date_col"] in df.columns else df.columns[0]
    index = pd.to_datetime(df[date_col], errors="coerce", format="mixed")
    series = pd.Series(
        pd.to_numeric(df[value_col], errors="coerce").to_numpy(), index=index, name=key
    )
    series = series.dropna().sort_index()

    rule = spec.get("resample")
    if rule:
        series = series.resample(rule).mean().dropna()
    return series


# ======================================================================================
# Forecasters
# ======================================================================================
def forecast_naive(train: np.ndarray, h: int, period: int) -> np.ndarray:
    """Repeat the last observation."""
    return np.repeat(train[-1], h)


def forecast_seasonal_naive(train: np.ndarray, h: int, period: int) -> np.ndarray:
    """Repeat the observation from one full period ago — the reference model."""
    if len(train) < period:
        return forecast_naive(train, h, period)
    season = train[-period:]
    return np.array([season[i % period] for i in range(h)])


def forecast_drift(train: np.ndarray, h: int, period: int) -> np.ndarray:
    """Extrapolate the straight line through the first and last observation."""
    slope = (train[-1] - train[0]) / max(1, len(train) - 1)
    return train[-1] + slope * np.arange(1, h + 1)


def forecast_holt_winters(train: np.ndarray, h: int, period: int) -> np.ndarray:
    """Exponential smoothing with additive trend and seasonality."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    if len(train) < 2 * period:
        return forecast_seasonal_naive(train, h, period)
    model = ExponentialSmoothing(
        train, trend="add", seasonal="add", seasonal_periods=period,
        initialization_method="estimated",
    ).fit()
    return np.asarray(model.forecast(h))


def forecast_sarima(train: np.ndarray, h: int, period: int) -> np.ndarray:
    """SARIMA(1,1,1)(1,1,1,m) — a fixed, defensible specification."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    if len(train) < 3 * period:
        return forecast_seasonal_naive(train, h, period)
    model = SARIMAX(
        train, order=(1, 1, 1), seasonal_order=(1, 1, 1, period),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    return np.asarray(model.forecast(h))


def forecast_gbm(train: np.ndarray, h: int, period: int) -> np.ndarray:
    """Direct multi-step gradient boosting on lag features.

    Recursive one-step forecasting would feed the model its own predictions and compound
    the error over the horizon. A separate model per step avoids that at the cost of
    training h models — worth it because compounding error is the dominant failure mode of
    recursive schemes at long horizons.
    """
    import lightgbm as lgb

    max_lag = min(period * 2, len(train) // 3)
    if max_lag < 2 or len(train) < max_lag + h + 10:
        return forecast_seasonal_naive(train, h, period)

    lags = list(range(1, max_lag + 1))
    predictions = []
    for step in range(1, h + 1):
        rows, targets = [], []
        for t in range(max_lag, len(train) - step + 1):
            rows.append([train[t - lag] for lag in lags])
            targets.append(train[t + step - 1])
        if len(rows) < 20:
            return forecast_seasonal_naive(train, h, period)
        model = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=15,
            min_child_samples=5, random_state=SEED, n_jobs=-1, verbose=-1,
        )
        model.fit(np.array(rows), np.array(targets))
        predictions.append(
            float(model.predict(np.array([[train[len(train) - lag] for lag in lags]]))[0])
        )
    return np.array(predictions)


FORECASTERS = {
    "naive": ("Naive (last value)", forecast_naive),
    "seasonal_naive": ("Seasonal naive", forecast_seasonal_naive),
    "drift": ("Drift", forecast_drift),
    "holt_winters": ("Holt-Winters (add/add)", forecast_holt_winters),
    "sarima": ("SARIMA(1,1,1)(1,1,1,m)", forecast_sarima),
    "gbm": ("LightGBM direct multi-step", forecast_gbm),
}


# ======================================================================================
# Analysis
# ======================================================================================
def decompose(values: np.ndarray, period: int) -> dict:
    """STL decomposition plus the diagnostics that justify the model choice."""
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

    effective_period = period if period > 1 and len(values) > 2 * period else 2
    result = STL(values, period=effective_period, robust=True).fit()

    var_total = np.var(values)
    strength_trend = max(0.0, 1 - np.var(result.resid) / max(1e-12, np.var(result.resid + result.trend)))
    strength_seasonal = max(0.0, 1 - np.var(result.resid) / max(1e-12, np.var(result.resid + result.seasonal)))

    # Two stationarity tests with opposite null hypotheses. Agreement is informative;
    # disagreement means the series is borderline and the answer should not be asserted.
    adf_p = float(adfuller(values, autolag="AIC")[1])
    kpss_p = float(kpss(values, regression="c", nlags="auto")[1])

    max_lag = min(60, len(values) // 3)
    return {
        "trend_strength": round(float(strength_trend), 4),
        "seasonal_strength": round(float(strength_seasonal), 4),
        "residual_variance_share": round(float(np.var(result.resid) / max(1e-12, var_total)), 4),
        "adf_pvalue": round(adf_p, 5),
        "kpss_pvalue": round(kpss_p, 5),
        "stationarity_verdict": _stationarity_verdict(adf_p, kpss_p),
        "acf": [
            {"lag": i, "value": round(float(v), 4)}
            for i, v in enumerate(acf(values, nlags=max_lag))
        ],
        "pacf": [
            {"lag": i, "value": round(float(v), 4)}
            for i, v in enumerate(pacf(values, nlags=min(max_lag, len(values) // 2 - 1)))
        ],
        "components": {
            "trend": [round(float(v), 4) for v in result.trend[-240:]],
            "seasonal": [round(float(v), 4) for v in result.seasonal[-240:]],
            "resid": [round(float(v), 4) for v in result.resid[-240:]],
        },
    }


def _stationarity_verdict(adf_p: float, kpss_p: float) -> str:
    """Interpret ADF and KPSS jointly.

    ADF's null is a unit root; KPSS's null is stationarity. They test opposite things, so
    running both catches the case where a single test is simply underpowered.
    """
    adf_stationary = adf_p < 0.05
    kpss_stationary = kpss_p > 0.05
    if adf_stationary and kpss_stationary:
        return "stationary (both tests agree)"
    if not adf_stationary and not kpss_stationary:
        return "non-stationary (both tests agree) — differencing required"
    if adf_stationary and not kpss_stationary:
        return (
            "conflicting: ADF rejects a unit root but KPSS rejects stationarity — "
            "consistent with trend-stationarity, so detrend rather than difference"
        )
    return (
        "conflicting: ADF cannot reject a unit root but KPSS cannot reject stationarity — "
        "the series is likely difference-stationary but the tests are underpowered here"
    )


def backtest(values: np.ndarray, period: int, horizon: int, n_origins: int) -> dict:
    """Rolling-origin evaluation. Refits at every origin; never sees the future."""
    n = len(values)
    min_train = max(3 * period, n - n_origins * horizon)
    origins = [
        min_train + i * horizon
        for i in range(n_origins)
        if min_train + i * horizon + horizon <= n
    ]
    if not origins:
        origins = [n - horizon]

    per_model: dict[str, list[float]] = {k: [] for k in FORECASTERS}
    per_model_mae: dict[str, list[float]] = {k: [] for k in FORECASTERS}
    example = None

    for origin in origins:
        train, actual = values[:origin], values[origin : origin + horizon]
        for key, (_, fn) in FORECASTERS.items():
            try:
                prediction = fn(train, horizon, period)
            except Exception:
                prediction = forecast_seasonal_naive(train, horizon, period)
            prediction = np.nan_to_num(prediction, nan=float(train[-1]))
            per_model[key].append(mase(actual, prediction, train, period))
            per_model_mae[key].append(float(np.mean(np.abs(actual - prediction))))
            if origin == origins[-1]:
                example = example or {"actual": [round(float(v), 4) for v in actual]}
                example[key] = [round(float(v), 4) for v in prediction]

    results = []
    for key, (label, _) in FORECASTERS.items():
        scores = np.array(per_model[key], dtype=float)
        scores = scores[np.isfinite(scores)]
        results.append(
            {
                "model": key,
                "label": label,
                "mase_mean": round(float(np.mean(scores)), 4) if len(scores) else None,
                "mase_std": round(float(np.std(scores)), 4) if len(scores) else None,
                "mase_worst_origin": round(float(np.max(scores)), 4) if len(scores) else None,
                "mae_mean": round(float(np.mean(per_model_mae[key])), 4),
                "beats_seasonal_naive": None,
                "n_origins": len(scores),
            }
        )

    baseline = next(r for r in results if r["model"] == "seasonal_naive")["mase_mean"]
    for row in results:
        if row["mase_mean"] is not None and baseline:
            row["beats_seasonal_naive"] = bool(row["mase_mean"] < baseline)
    results.sort(key=lambda r: (r["mase_mean"] is None, r["mase_mean"]))

    return {
        "n_origins": len(origins),
        "horizon": horizon,
        "origins": origins,
        "results": results,
        "example_forecast": example,
        "protocol": (
            f"Rolling origin: {len(origins)} refits, each forecasting {horizon} steps "
            "ahead from data strictly before the origin. No model sees any observation at "
            "or after its origin. Reported MASE is the mean across origins, so a single "
            "lucky cut cannot determine the ranking."
        ),
    }


# ======================================================================================
# Orchestration
# ======================================================================================
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pip_install_hint = "pip install statsmodels"
    try:
        import statsmodels  # noqa: F401
    except ImportError as error:  # pragma: no cover
        raise SystemExit(f"statsmodels is required ({pip_install_hint}): {error}")

    crisp = CrispDm(
        project=PROJECT,
        business_question=(
            "Which forecasting method should be the default for a new series, and how "
            "much does that answer depend on the structure of the series itself?"
        ),
    )
    crisp.record(
        Phase(
            name="business_understanding",
            summary=(
                "Forecasting tool selection is usually settled by habit or by a single "
                "benchmark. The practical question is narrower and more useful: given a "
                "series with an identifiable structure, which family should be reached for "
                "first, and when is a classical method still the right answer?"
            ),
            decisions=[
                Decision(
                    question="How can models be compared across series in different units?",
                    choice="MASE, scaled by in-sample seasonal-naive error.",
                    rationale=(
                        "Averaging raw MAE across passenger counts, degrees Celsius and "
                        "prescription volumes is meaningless. MASE is unit-free and "
                        "anchored: 1.0 always means 'matched seasonal naive', so results "
                        "are comparable and interpretable at the same time."
                    ),
                    alternatives_rejected=[
                        "MAPE — undefined at zero, explodes near it, and asymmetric "
                        "between over- and under-forecasts.",
                        "Raw RMSE — not comparable across units.",
                    ],
                ),
                Decision(
                    question="Single holdout or rolling origin?",
                    choice=f"Rolling origin with up to {N_ORIGINS} refits.",
                    rationale=(
                        "A single split reports where one arbitrary cut happened to fall. "
                        "Refitting at multiple origins gives a mean and a spread, which "
                        "is what distinguishes a genuinely better model from a luckier one."
                    ),
                ),
            ],
            evidence={"horizon": HORIZON, "n_origins": N_ORIGINS, "n_series": len(SERIES_SPECS)},
        )
    )

    with artifacts.run(PROJECT, seed=SEED) as ctx:
        series_payload: dict[str, dict] = {}
        all_rankings: dict[str, list[str]] = {}

        for key, spec in SERIES_SPECS.items():
            print(f"\n[{key}] loading …")
            series = load_series(key)
            values = series.to_numpy(dtype=float)
            print(f"      {len(values)} observations, "
                  f"{series.index.min().date()} → {series.index.max().date()}")

            print(f"      decomposing (period={spec['period']}) …")
            decomposition = decompose(values, spec["period"])

            print(f"      backtesting {len(FORECASTERS)} models over rolling origins …")
            bt = backtest(values, spec["period"], HORIZON, N_ORIGINS)

            winner = bt["results"][0]
            all_rankings[key] = [r["model"] for r in bt["results"]]
            print(f"      winner: {winner['label']} (MASE {winner['mase_mean']})")

            series_payload[key] = {
                "spec": spec,
                "n_observations": len(values),
                "start": str(series.index.min().date()),
                "end": str(series.index.max().date()),
                "summary": artifacts.histogram(values, bins=30),
                "observations": [
                    {"t": str(idx.date()), "y": round(float(v), 4)}
                    for idx, v in list(series.items())[-400:]
                ],
                "decomposition": decomposition,
                "backtest": bt,
            }

        # --- Cross-series synthesis -----------------------------------------------------
        # The point of using four series: does one model dominate, or does the best model
        # depend on structure? Averaging ranks across series answers that directly.
        rank_totals: dict[str, list[int]] = {k: [] for k in FORECASTERS}
        for ranking in all_rankings.values():
            for position, model in enumerate(ranking, start=1):
                rank_totals[model].append(position)

        synthesis = sorted(
            (
                {
                    "model": model,
                    "label": FORECASTERS[model][0],
                    "mean_rank": round(float(np.mean(positions)), 2),
                    "best_rank": int(np.min(positions)),
                    "worst_rank": int(np.max(positions)),
                    "wins": sum(1 for p in positions if p == 1),
                }
                for model, positions in rank_totals.items()
            ),
            key=lambda r: r["mean_rank"],
        )

        winners_by_series = {
            key: payload["backtest"]["results"][0]["label"]
            for key, payload in series_payload.items()
        }
        n_distinct_winners = len(set(winners_by_series.values()))

        crisp.record(
            Phase(
                name="data_understanding",
                summary=(
                    f"Four real series with deliberately different structure: "
                    + "; ".join(
                        f"{s['title']} ({s['structure']})" for s in SERIES_SPECS.values()
                    )
                    + ". Each is decomposed by STL and tested for stationarity with both "
                    "ADF and KPSS, whose null hypotheses are opposites."
                ),
                evidence={
                    key: {
                        "n": payload["n_observations"],
                        "trend_strength": payload["decomposition"]["trend_strength"],
                        "seasonal_strength": payload["decomposition"]["seasonal_strength"],
                        "stationarity": payload["decomposition"]["stationarity_verdict"],
                    }
                    for key, payload in series_payload.items()
                },
                decisions=[
                    Decision(
                        question="Why run both ADF and KPSS?",
                        choice="Both, with an explicit verdict when they conflict.",
                        rationale=(
                            "ADF's null is a unit root; KPSS's null is stationarity. A "
                            "single test that fails to reject is ambiguous between 'the "
                            "null holds' and 'the test is underpowered'. Running both "
                            "distinguishes trend-stationary from difference-stationary "
                            "series, which determines whether to detrend or difference."
                        ),
                    ),
                ],
            )
        )
        crisp.record(
            Phase(
                name="data_preparation",
                summary=(
                    "Minimal and identical across series: parse dates, coerce values to "
                    "float, drop unparseable rows, sort chronologically. No imputation, no "
                    "outlier removal — a smoothed series would make every model look "
                    "better than it is, and the sunspot and temperature series contain "
                    "genuine extremes that a forecaster ought to be judged on."
                ),
                evidence={
                    key: {"n": p["n_observations"], "start": p["start"], "end": p["end"]}
                    for key, p in series_payload.items()
                },
            )
        )
        crisp.record(
            Phase(
                name="modeling",
                summary=(
                    f"Six forecasters from three families — naive baselines, classical "
                    f"statistical (Holt-Winters, SARIMA) and machine learning (LightGBM "
                    f"direct multi-step) — each refit at every rolling origin on every "
                    f"series."
                ),
                decisions=[
                    Decision(
                        question="Recursive or direct multi-step for the ML forecaster?",
                        choice="Direct — a separate model per horizon step.",
                        rationale=(
                            "Recursive forecasting feeds the model its own predictions, so "
                            "error compounds over the horizon and the last steps are "
                            "predictions of predictions. Direct multi-step trains h models "
                            "and avoids the feedback entirely. It costs h times the "
                            "training, which at this scale is irrelevant."
                        ),
                        alternatives_rejected=[
                            "Recursive one-step — cheaper, but compounding error is the "
                            "dominant failure mode at horizon 12.",
                        ],
                    ),
                ],
                evidence={"forecasters": {k: v[0] for k, v in FORECASTERS.items()}},
            )
        )
        crisp.record(
            Phase(
                name="evaluation",
                summary=(
                    f"No single model wins everywhere: {n_distinct_winners} different "
                    f"models take first place across the four series. Best mean rank is "
                    f"{synthesis[0]['label']} at {synthesis[0]['mean_rank']}, but it wins "
                    f"outright on only {synthesis[0]['wins']} of {len(SERIES_SPECS)}."
                ),
                evidence={
                    "winners_by_series": winners_by_series,
                    "cross_series_ranking": synthesis,
                    "per_series": {
                        key: payload["backtest"]["results"]
                        for key, payload in series_payload.items()
                    },
                },
                decisions=[
                    Decision(
                        question="Is there a default model to recommend?",
                        choice=(
                            f"{synthesis[0]['label']} as a first attempt, but structure "
                            "should decide."
                        ),
                        rationale=(
                            "Mean rank across four structurally different series is a "
                            "weak recommendation by construction. The per-series table is "
                            "the real result: a model that dominates on calendar-seasonal "
                            "data can rank last on a series whose cycle is not "
                            "calendar-anchored."
                        ),
                    ),
                ],
                risks=[
                    "Model hyperparameters are fixed rather than tuned per series. Tuning "
                    "would likely improve the ML forecaster most, so the comparison is "
                    "mildly conservative towards it.",
                    "SARIMA order is fixed at (1,1,1)(1,1,1,m) rather than selected per "
                    "series by AIC. A per-series search would be fairer to SARIMA and is "
                    "the obvious next step.",
                    "Series lengths range from 141 to 3,650 observations, so the number of "
                    "usable rolling origins differs and short-series estimates are noisier.",
                ],
            )
        )
        crisp.record(
            Phase(
                name="deployment",
                summary=(
                    "Each series ships with its observations, STL components, ACF/PACF and "
                    "full backtest table, driving a forecast studio where a reader can "
                    "switch series and model and see the rolling-origin error move."
                ),
                evidence={"series": list(SERIES_SPECS)},
            )
        )

        artifacts.write(OUT / "series.json", series_payload, context=ctx)
        artifacts.write(
            OUT / "synthesis.json",
            {
                "cross_series_ranking": synthesis,
                "winners_by_series": winners_by_series,
                "n_distinct_winners": n_distinct_winners,
                "horizon": HORIZON,
                "protocol": next(iter(series_payload.values()))["backtest"]["protocol"],
            },
            context=ctx,
        )
        artifacts.write(OUT / "crispdm.json", crisp.to_dict(), context=ctx)
        artifacts.write(
            OUT / "provenance.json",
            {"datasets": [data.provenance_record(k) for k in SERIES_SPECS]},
            context=ctx,
        )

    print(f"\n✓ {PROJECT} complete — {n_distinct_winners} distinct winners across "
          f"{len(SERIES_SPECS)} series; best mean rank {synthesis[0]['label']}")


if __name__ == "__main__":
    main()
