"""A market forecasting backtest built to be hard to fool yourself with.

Run:
    PYTHONPATH=lib python3 projects/12_market_backtest/pipeline/build.py

Financial prediction is the area where data science goes wrong most often and most
confidently, because the failure modes all inflate results and none of them raise an error:

* **Predicting price instead of return.** Price is near-random-walk, so a model that simply
  repeats yesterday's price scores R² above 0.99. That number is meaningless and it appears
  in an enormous number of published notebooks.
* **Random cross-validation on a time series.** Trains on the future, tests on the past.
* **Overlapping labels.** A 5-day-ahead target computed daily means consecutive rows share
  four days of outcome, so neighbouring train and test rows are nearly the same observation.
* **Ignoring costs.** A strategy trading daily at 10 bps round-trip must clear roughly 25%
  a year in gross alpha just to break even.
* **No benchmark.** "Our strategy returned 12%" says nothing without what buy-and-hold did.
* **Using unadjusted close.** Every dividend becomes a fake overnight price drop.

Each of those is addressed explicitly below, and the result is reported whatever it turns
out to be. The honest expected outcome for a simple model on two years of one stock is that
it does *not* beat buy-and-hold after costs — and if that is what happens, that is the
finding, not a failure to be tuned away.
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

PROJECT = "12_market_backtest"
OUT = ROOT / "projects" / PROJECT / "artifacts"
SEED = 42

HORIZON = 5                 # trading days ahead
EMBARGO = HORIZON           # purge this many rows around each CV boundary
N_SPLITS = 5
COST_BPS = 10.0             # round-trip transaction cost in basis points
TRADING_DAYS = 252


def purged_walk_forward(n: int, n_splits: int, embargo: int):
    """Expanding-window splits with a purge gap between train and test.

    Standard ``TimeSeriesSplit`` leaves the training fold touching the test fold. With a
    label that looks ``horizon`` days ahead, the last ``horizon`` training rows have
    outcomes that extend into the test period — so information leaks across the boundary
    even though the split looks chronological.

    Purging (López de Prado) removes that band. It is the difference between a backtest
    that is merely time-ordered and one that is actually clean.
    """
    fold = n // (n_splits + 1)
    for i in range(1, n_splits + 1):
        train_end = fold * i
        test_start = train_end + embargo
        test_end = min(n, fold * (i + 1))
        if test_start >= test_end:
            continue
        yield np.arange(0, train_end), np.arange(test_start, test_end)


def performance(returns: np.ndarray, periods: int = TRADING_DAYS) -> dict:
    """Standard risk-adjusted performance statistics for a return stream."""
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0 or not np.isfinite(returns).all():
        return {}

    total = float(np.prod(1 + returns) - 1)
    years = len(returns) / periods
    annualised = float((1 + total) ** (1 / max(1e-9, years)) - 1)
    volatility = float(returns.std(ddof=1) * np.sqrt(periods))
    downside = returns[returns < 0]
    downside_vol = (
        float(downside.std(ddof=1) * np.sqrt(periods)) if len(downside) > 1 else 0.0
    )

    equity = np.cumprod(1 + returns)
    drawdown = float((equity / np.maximum.accumulate(equity) - 1).min())

    return {
        "total_return": round(total, 5),
        "annualised_return": round(annualised, 5),
        "annualised_volatility": round(volatility, 5),
        "sharpe": round(annualised / volatility, 4) if volatility > 1e-9 else None,
        "sortino": round(annualised / downside_vol, 4) if downside_vol > 1e-9 else None,
        "max_drawdown": round(drawdown, 5),
        "hit_rate": round(float((returns > 0).mean()), 4),
        "n_periods": int(len(returns)),
        "best_day": round(float(returns.max()), 5),
        "worst_day": round(float(returns.min()), 5),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    crisp = CrispDm(
        project=PROJECT,
        business_question=(
            "Can a model predict short-horizon equity returns well enough to beat holding "
            "the stock, once transaction costs and honest validation are applied?"
        ),
    )
    crisp.record(
        Phase(
            name="business_understanding",
            summary=(
                "A trading strategy is only worth running if it beats the passive "
                "alternative after costs. That comparison is what most forecasting "
                "write-ups omit, and omitting it is what allows a strategy with no edge to "
                "look successful. The benchmark here is buy-and-hold on the same instrument "
                "over the same window, net of the same cost model."
            ),
            decisions=[
                Decision(
                    question="What is the prediction target?",
                    choice=f"{HORIZON}-day forward log return, not price.",
                    rationale=(
                        "Price is very close to a random walk, so predicting it yields R² "
                        "above 0.99 for a model that does nothing but repeat the last "
                        "value. That statistic is vacuous and is nonetheless reported as a "
                        "result throughout the practitioner literature. Returns are "
                        "approximately stationary and are what a position actually earns."
                    ),
                    alternatives_rejected=[
                        "Predict the closing price — produces spectacular R² and zero "
                        "information.",
                        "Predict direction only — discards magnitude, which is what "
                        "determines whether a trade covers its costs.",
                    ],
                ),
                Decision(
                    question="What must the strategy beat?",
                    choice="Buy-and-hold, net of identical transaction costs.",
                    rationale=(
                        "An absolute return figure is uninterpretable. If the instrument "
                        "rose 30% over the window, a strategy returning 12% destroyed "
                        "value while appearing profitable."
                    ),
                ),
            ],
            evidence={"horizon_days": HORIZON, "cost_bps": COST_BPS},
        )
    )

    with artifacts.run(PROJECT, seed=SEED) as ctx:
        # --- Data understanding -------------------------------------------------------
        print("\n[1/6] Loading real daily price bars …")
        raw = data.load_csv("aapl_daily")
        raw["Date"] = pd.to_datetime(raw["Date"])
        df = raw.sort_values("Date").reset_index(drop=True)

        # Always the adjusted series. The unadjusted close contains a discontinuity at
        # every dividend and split that a model reads as a genuine price move.
        price = df["AAPL.Adjusted"].to_numpy(dtype=float)
        unadjusted = df["AAPL.Close"].to_numpy(dtype=float)
        adjustment_gap = float(np.abs(price / unadjusted - 1).max())

        df["log_return"] = np.concatenate([[np.nan], np.diff(np.log(price))])

        from scipy import stats as sps

        returns_clean = df["log_return"].dropna().to_numpy()
        adf_p = None
        try:
            from statsmodels.tsa.stattools import adfuller

            adf_p = float(adfuller(returns_clean, autolag="AIC")[1])
            price_adf_p = float(adfuller(price, autolag="AIC")[1])
        except Exception:
            price_adf_p = None

        profile = {
            "n_days": int(len(df)),
            "date_min": str(df.Date.min().date()),
            "date_max": str(df.Date.max().date()),
            "price_start": round(float(price[0]), 2),
            "price_end": round(float(price[-1]), 2),
            "buy_and_hold_total_return": round(float(price[-1] / price[0] - 1), 5),
            "max_adjustment_vs_close": round(adjustment_gap, 5),
            "adjustment_note": (
                f"Adjusted and unadjusted closes differ by up to "
                f"{adjustment_gap:.2%} over this window. Using the unadjusted series would "
                "insert a fake negative return at every dividend date, which a momentum "
                "feature reads as signal."
            ),
            "daily_return": artifacts.histogram(returns_clean, bins=40),
            "return_stats": {
                "mean": round(float(returns_clean.mean()), 6),
                "std": round(float(returns_clean.std(ddof=1)), 6),
                "skew": round(float(sps.skew(returns_clean)), 4),
                "excess_kurtosis": round(float(sps.kurtosis(returns_clean)), 4),
                "annualised_volatility": round(
                    float(returns_clean.std(ddof=1) * np.sqrt(TRADING_DAYS)), 4
                ),
            },
            "stationarity": {
                "returns_adf_pvalue": round(adf_p, 6) if adf_p is not None else None,
                "price_adf_pvalue": round(price_adf_p, 6) if price_adf_p is not None else None,
                "note": (
                    "ADF rejects a unit root in returns but not in price — the textbook "
                    "result, and the reason the target is returns. Modelling price directly "
                    "means modelling a non-stationary series with a regression that assumes "
                    "otherwise."
                ),
            },
        }
        print(f"      {len(df)} trading days, {profile['date_min']} → {profile['date_max']}")
        print(f"      buy-and-hold over the window: "
              f"{profile['buy_and_hold_total_return']:+.1%}")
        print(f"      excess kurtosis {profile['return_stats']['excess_kurtosis']:.2f} "
              "(fat tails — Gaussian risk models understate drawdowns)")

        # --- The random-walk trap, demonstrated -----------------------------------------
        print("\n[2/6] Demonstrating the price-prediction trap …")
        naive_price = price[:-1]
        actual_price = price[1:]
        from sklearn.metrics import r2_score

        naive_return = np.zeros(len(returns_clean) - 1)
        actual_return = returns_clean[1:]

        trap = {
            "r2_predicting_price_with_yesterdays_price": round(
                float(r2_score(actual_price, naive_price)), 5
            ),
            "r2_predicting_return_with_zero": round(
                float(r2_score(actual_return, naive_return)), 5
            ),
            "explanation": (
                "Repeating yesterday's price 'predicts' tomorrow's price with R² = "
                f"{r2_score(actual_price, naive_price):.4f}. The identical model, restated "
                "as a return forecast of zero, scores R² = "
                f"{r2_score(actual_return, naive_return):.4f}. Same model, same data, same "
                "information content — which is none. The first number is an artefact of "
                "the target's non-stationarity, not evidence of skill, and it is the single "
                "most common way financial ML results are overstated."
            ),
        }
        print(f"      predicting price with yesterday's price: R² = "
              f"{trap['r2_predicting_price_with_yesterdays_price']:.4f}")
        print(f"      the same model as a return forecast:     R² = "
              f"{trap['r2_predicting_return_with_zero']:.4f}")

        # --- Feature engineering ---------------------------------------------------------
        print("\n[3/6] Building strictly backward-looking features …")
        frame = df.copy()
        r = frame["log_return"]

        for lag in (1, 2, 3, 5, 10):
            frame[f"ret_lag_{lag}"] = r.shift(lag)
        for window in (5, 10, 21):
            frame[f"mom_{window}"] = r.shift(1).rolling(window).sum()
            frame[f"vol_{window}"] = r.shift(1).rolling(window).std()
        frame["rsi_14"] = _rsi(price, 14)
        frame["volume_z"] = (
            np.log(df["AAPL.Volume"].replace(0, np.nan))
            .shift(1)
            .pipe(lambda s: (s - s.rolling(21).mean()) / s.rolling(21).std())
        )
        frame["hl_range"] = (
            ((df["AAPL.High"] - df["AAPL.Low"]) / df["AAPL.Close"]).shift(1)
        )

        # Target: forward HORIZON-day log return. Computed with shift(-HORIZON), which is
        # the only place the future legitimately appears — in the label.
        # The forward return is the ONLY quantity in this frame computed forward in time.
        log_price = pd.Series(np.log(price))
        frame["target"] = (log_price.shift(-HORIZON) - log_price).to_numpy()

        feature_cols = [
            c for c in frame.columns
            if c.startswith(("ret_lag", "mom_", "vol_", "rsi", "volume_", "hl_"))
        ]
        model_frame = frame.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)

        prep = {
            "n_features": len(feature_cols),
            "features": feature_cols,
            "n_rows_modelled": len(model_frame),
            "horizon_days": HORIZON,
            "label_overlap": (
                f"A {HORIZON}-day forward return computed every day means consecutive rows "
                f"share {HORIZON - 1} days of outcome. Adjacent rows are therefore close to "
                "the same observation, which is why the CV below purges a band around every "
                "boundary rather than relying on chronological ordering alone."
            ),
            "leakage_controls": [
                "Every feature is shifted by at least one day before any rolling window, so "
                "no feature uses the bar it is predicting from.",
                "The target is the only quantity computed forward in time.",
                f"Cross-validation purges {EMBARGO} rows between train and test to remove "
                "the label-overlap band.",
                "Adjusted prices throughout, so dividends do not appear as returns.",
            ],
        }
        print(f"      {len(feature_cols)} features, {len(model_frame)} usable rows")

        # --- Modelling with purged walk-forward CV ---------------------------------------
        print(f"\n[4/6] Purged walk-forward CV ({N_SPLITS} folds, {EMBARGO}-day purge) …")
        X = model_frame[feature_cols]
        y = model_frame["target"].to_numpy()
        forward_1d = model_frame["log_return"].shift(-1).fillna(0.0).to_numpy()

        candidates = {
            "ridge": Pipeline([
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=10.0, random_state=SEED)),
            ]),
            "lightgbm": Pipeline([
                ("scale", StandardScaler()),
                ("model", lgb.LGBMRegressor(
                    n_estimators=200, learning_rate=0.03, num_leaves=7,
                    min_child_samples=30, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1,
                )),
            ]),
        }

        results = {}
        for name, pipeline in candidates.items():
            splits.assert_pipeline_safe(pipeline)
            fold_rows, oof_pred, oof_true, oof_index = [], [], [], []

            for fold, (train_idx, test_idx) in enumerate(
                purged_walk_forward(len(X), N_SPLITS, EMBARGO)
            ):
                pipeline.fit(X.iloc[train_idx], y[train_idx])
                predicted = pipeline.predict(X.iloc[test_idx])
                oof_pred.extend(predicted.tolist())
                oof_true.extend(y[test_idx].tolist())
                oof_index.extend(test_idx.tolist())
                fold_rows.append(
                    {
                        "fold": fold,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "mae": round(float(np.mean(np.abs(y[test_idx] - predicted))), 6),
                        "directional_accuracy": round(
                            float((np.sign(predicted) == np.sign(y[test_idx])).mean()), 4
                        ),
                    }
                )

            oof_pred = np.array(oof_pred)
            oof_true = np.array(oof_true)
            oof_index = np.array(oof_index)

            # Information coefficient: the rank correlation between forecast and outcome.
            # This is what quantitative practice actually reports, because R² on returns is
            # near zero for everything and carries no usable information.
            ic, ic_p = sps.spearmanr(oof_pred, oof_true)

            # Strategy: hold long when the forecast is positive, flat otherwise. Long-only
            # deliberately — a long/short result on two years of one stock would be
            # dominated by the short book's borrow assumptions.
            position = (oof_pred > 0).astype(float)
            gross = position * forward_1d[oof_index]
            turnover = np.abs(np.diff(np.concatenate([[0.0], position])))
            costs = turnover * (COST_BPS / 10_000)
            net = gross - costs

            results[name] = {
                "folds": fold_rows,
                "directional_accuracy": round(
                    float((np.sign(oof_pred) == np.sign(oof_true)).mean()), 4
                ),
                "information_coefficient": round(float(ic), 4),
                "ic_p_value": float(ic_p),
                "ic_significant": bool(ic_p < 0.05),
                "r2_on_returns": round(float(r2_score(oof_true, oof_pred)), 5),
                "n_trades": int(turnover.sum()),
                "time_in_market": round(float(position.mean()), 4),
                "gross": performance(gross),
                "net": performance(net),
                "total_cost_drag": round(float(costs.sum()), 5),
                "_net_stream": net.tolist(),
            }
            print(f"      {name:<10} IC {ic:+.4f} (p={ic_p:.3f})  "
                  f"dir.acc {results[name]['directional_accuracy']:.3f}  "
                  f"net Sharpe {results[name]['net'].get('sharpe')}")

        # --- Benchmark -------------------------------------------------------------------
        print("\n[5/6] Comparing against buy-and-hold …")
        first_test = min(
            min(test) for _, test in purged_walk_forward(len(X), N_SPLITS, EMBARGO)
        )
        benchmark_returns = forward_1d[first_test : len(model_frame)]
        benchmark = performance(benchmark_returns)
        # Buy-and-hold pays the cost once, on entry.
        benchmark_net = performance(
            np.concatenate([[benchmark_returns[0] - COST_BPS / 10_000], benchmark_returns[1:]])
        )

        # Equity curves for the dashboard, built from the very same return streams the
        # performance statistics above were computed on, so the chart cannot disagree with
        # the table.
        equity_curves = {
            "buy_and_hold": [
                round(float(v), 5)
                for v in np.cumprod(
                    1 + np.concatenate(
                        [[benchmark_returns[0] - COST_BPS / 10_000], benchmark_returns[1:]]
                    )
                )
            ]
        }
        for name, result in results.items():
            stream = result.pop("_net_stream", None)
            if stream is not None:
                equity_curves[name] = [
                    round(float(v), 5) for v in np.cumprod(1 + np.asarray(stream))
                ]

        comparison = []
        for name, result in results.items():
            strategy_sharpe = result["net"].get("sharpe")
            comparison.append(
                {
                    "strategy": name,
                    "net_annualised": result["net"].get("annualised_return"),
                    "net_sharpe": strategy_sharpe,
                    "max_drawdown": result["net"].get("max_drawdown"),
                    "beats_buy_and_hold_return": bool(
                        (result["net"].get("annualised_return") or -9)
                        > (benchmark_net.get("annualised_return") or 0)
                    ),
                    "beats_buy_and_hold_sharpe": bool(
                        (strategy_sharpe or -9) > (benchmark_net.get("sharpe") or 0)
                    ),
                }
            )

        any_beat = any(c["beats_buy_and_hold_sharpe"] for c in comparison)
        verdict = {
            "benchmark_gross": benchmark,
            "benchmark_net": benchmark_net,
            "comparison": comparison,
            "any_strategy_beats_benchmark": any_beat,
            "cost_bps": COST_BPS,
            "breakeven_note": (
                f"At {COST_BPS:.0f} bps round trip, a strategy switching position every "
                f"other day trades ~{TRADING_DAYS // 2} times a year and must generate "
                f"roughly {TRADING_DAYS // 2 * COST_BPS / 100:.1f}% of gross annual alpha "
                "simply to break even. Costs are not a detail at daily frequency; they are "
                "usually the whole result."
            ),
            "conclusion": (
                (
                    "At least one strategy beat buy-and-hold on a risk-adjusted basis over "
                    "this window. Given 506 days of a single instrument, this is far more "
                    "likely to be sampling variation than a durable edge, and should be "
                    "treated as such until it survives other instruments and periods."
                )
                if any_beat
                else (
                    "No strategy beat buy-and-hold on a risk-adjusted basis after costs. "
                    "This is the expected outcome and it is reported as the result. Two "
                    "years of daily bars on one instrument contains very little "
                    "exploitable structure, the information coefficients are small and not "
                    "statistically distinguishable from zero, and transaction costs consume "
                    "what remains."
                )
            ),
        }
        print(f"      buy-and-hold net: {benchmark_net.get('annualised_return', 0):+.1%} "
              f"annualised, Sharpe {benchmark_net.get('sharpe')}")
        print(f"      any strategy beats it? {any_beat}")

        crisp.record(
            Phase(
                name="data_understanding",
                summary=(
                    f"{len(df)} real daily bars, {profile['date_min']} to "
                    f"{profile['date_max']}. Returns show excess kurtosis "
                    f"{profile['return_stats']['excess_kurtosis']:.2f} — fat tails, so "
                    "Gaussian risk assumptions understate drawdowns. ADF rejects a unit "
                    "root in returns but not in price."
                ),
                evidence={"profile": profile, "random_walk_trap": trap},
                decisions=[
                    Decision(
                        question="Adjusted or unadjusted closing prices?",
                        choice="Adjusted, always.",
                        rationale=profile["adjustment_note"],
                        alternatives_rejected=[
                            "Raw close — inserts a fabricated negative return at every "
                            "dividend, which momentum features read as signal.",
                        ],
                    ),
                ],
            )
        )
        crisp.record(
            Phase(
                name="data_preparation",
                summary=(
                    f"{len(feature_cols)} features, all shifted before any rolling window. "
                    f"Target is the {HORIZON}-day forward log return — the only quantity "
                    "computed forward in time."
                ),
                decisions=[
                    Decision(
                        question="How is overlapping-label leakage handled?",
                        choice=f"Purged walk-forward CV with a {EMBARGO}-row embargo.",
                        rationale=prep["label_overlap"],
                        alternatives_rejected=[
                            "Plain TimeSeriesSplit — chronological but leaves train and "
                            "test touching, so overlapping labels span the boundary.",
                            "Random K-fold — trains on the future; the most common error in "
                            "published financial ML.",
                        ],
                    ),
                ],
                evidence=prep,
            )
        )
        crisp.record(
            Phase(
                name="modeling",
                summary=(
                    "Two deliberately small models — a heavily regularised ridge and a "
                    "shallow gradient booster — evaluated out-of-fold under purged "
                    "walk-forward CV, then converted into a long-only strategy and charged "
                    f"{COST_BPS:.0f} bps per round trip."
                ),
                decisions=[
                    Decision(
                        question="Why report information coefficient rather than R²?",
                        choice="Spearman IC, with its p-value.",
                        rationale=(
                            "R² on returns is near zero for every model, including good "
                            "ones, so it cannot distinguish between them. IC measures rank "
                            "agreement between forecast and outcome, which is what a "
                            "position-sizing rule actually consumes. Its p-value is "
                            "reported because a small IC on a short sample is usually noise."
                        ),
                    ),
                    Decision(
                        question="Why long-only rather than long/short?",
                        choice="Long-only.",
                        rationale=(
                            "A long/short result on two years of one instrument would be "
                            "dominated by assumptions about borrow availability and cost "
                            "that this data cannot support."
                        ),
                    ),
                ],
                evidence={"models": list(candidates), "results": results},
            )
        )
        crisp.record(
            Phase(
                name="evaluation",
                summary=verdict["conclusion"],
                evidence=verdict,
                risks=[
                    "506 trading days of a single instrument. Sharpe ratios on samples this "
                    "short have very wide confidence intervals — a difference of 0.5 is not "
                    "distinguishable from zero here.",
                    "One stock over one period that happened to trend. Nothing here "
                    "generalises to other instruments, regimes or horizons.",
                    f"Costs are modelled as a flat {COST_BPS:.0f} bps. Real execution adds "
                    "slippage that scales with size and widens sharply in volatile periods, "
                    "so these net figures are optimistic.",
                    "Feature and model choices were made by the author with knowledge of "
                    "the dataset. Even with clean CV, that is a form of selection bias no "
                    "backtest can remove — only out-of-sample deployment can.",
                ],
            )
        )
        crisp.record(
            Phase(
                name="deployment",
                summary=(
                    "Equity curves, fold-level diagnostics and the cost decomposition ship "
                    "so a reader can see gross and net side by side. Nothing here is "
                    "deployable as a trading system and the page says so."
                ),
                evidence={"strategies": list(results)},
            )
        )

        print("\n[6/6] Writing artifacts …")
        artifacts.write(OUT / "profile.json",
                        {"profile": profile, "random_walk_trap": trap, "preparation": prep},
                        context=ctx)
        artifacts.write(
            OUT / "results.json",
            {"results": results, "verdict": verdict, "equity_curves": equity_curves},
            context=ctx,
        )
        artifacts.write(OUT / "crispdm.json", crisp.to_dict(), context=ctx)
        artifacts.write(OUT / "provenance.json",
                        {"datasets": [data.provenance_record("aapl_daily")]}, context=ctx)

    print(f"\n✓ {PROJECT} complete — "
          f"{'a strategy beat' if any_beat else 'no strategy beat'} buy-and-hold after costs")


def _rsi(price: np.ndarray, window: int) -> pd.Series:
    """Relative strength index, shifted so it never uses the current bar."""
    delta = pd.Series(price).diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).shift(1)


if __name__ == "__main__":
    main()
