"""AutoML tournament, with a controlled demonstration of target leakage.

Run:
    PYTHONPATH=lib python3 projects/06_automl_tournament/pipeline/build.py

The UCI Bank Marketing dataset ships with a documented trap. Its ``duration`` column
records how long the sales call lasted — a quantity that does not exist until the call is
over and the outcome effectively decided. A zero-second call is necessarily a rejection.
UCI's own documentation states the column "should be discarded if the intention is to have
a realistic predictive model".

It is included in most published notebooks on this dataset anyway, because including it
makes every model look dramatically better.

This project runs the whole tournament twice — once with the leaking feature and once
without — under identical conditions. The gap between the two leaderboards is the cost of
one careless column, measured rather than asserted. That measurement is the point of the
project; the AutoML machinery is the vehicle for it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lib"))

from dsx import artifacts, data, metrics, splits  # noqa: E402
from dsx.crispdm import CrispDm, Decision, Phase  # noqa: E402

PROJECT = "06_automl_tournament"
OUT = ROOT / "projects" / PROJECT / "artifacts"
SEED = 42
N_SEARCH_ITER = 12
N_FOLDS = 4


def build_search_space(seed: int) -> dict:
    """Candidate model families and their hyperparameter distributions.

    Deliberately spans different inductive biases rather than tuning one family harder:
    a linear model, a bagged tree ensemble, a boosted ensemble, and a distance-free
    probabilistic baseline. An AutoML system that searches only boosting depths is really
    a hyperparameter tuner.
    """
    from scipy.stats import loguniform, randint, uniform
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB

    return {
        "logistic": {
            "estimator": LogisticRegression(max_iter=3000, random_state=seed),
            "space": {
                "model__C": loguniform(1e-3, 1e2),
                "model__class_weight": [None, "balanced"],
            },
            "bias": "linear decision boundary in the encoded space",
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=seed, n_jobs=-1),
            "space": {
                "model__n_estimators": randint(200, 600),
                "model__max_depth": randint(4, 24),
                "model__min_samples_leaf": randint(1, 40),
                "model__max_features": uniform(0.2, 0.7),
                "model__class_weight": [None, "balanced"],
            },
            "bias": "bagged axis-aligned partitions, variance reduction",
        },
        "hist_gradient_boosting": {
            "estimator": HistGradientBoostingClassifier(random_state=seed),
            "space": {
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__max_iter": randint(100, 500),
                "model__max_leaf_nodes": randint(8, 64),
                "model__min_samples_leaf": randint(10, 80),
                "model__l2_regularization": loguniform(1e-6, 1e1),
            },
            "bias": "sequential residual fitting, bias reduction",
        },
        "gaussian_nb": {
            "estimator": GaussianNB(),
            "space": {"model__var_smoothing": loguniform(1e-11, 1e-6)},
            "bias": "conditional independence assumption — deliberately naive reference",
        },
    }


def run_tournament(X: pd.DataFrame, y: np.ndarray, numeric: list[str],
                   categorical: list[str], label: str) -> dict:
    """Search, fit and score every family under one identical protocol."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    print(f"\n  --- tournament: {label} ---")
    train_idx, test_idx, split_report = splits.stratified_split(y, test_size=0.25, seed=SEED)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Preprocessing lives inside the pipeline, so every fold refits it on that fold's
    # training rows only. This is the structural defence against preprocessing leakage:
    # scaling on the full dataset before CV would leak test statistics into training and
    # nothing in the metrics would show it.
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                              min_frequency=0.01),
                categorical,
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    leaderboard = []
    fitted: dict[str, object] = {}

    for name, config in build_search_space(SEED).items():
        pipeline = Pipeline([("prep", preprocessor), ("model", config["estimator"])])
        splits.assert_pipeline_safe(pipeline)

        started = time.perf_counter()
        search = RandomizedSearchCV(
            pipeline,
            config["space"],
            n_iter=N_SEARCH_ITER,
            scoring="average_precision",  # PR-AUC: the positive class is 11%
            cv=cv,
            random_state=SEED,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - started

        probabilities = search.predict_proba(X_test)[:, 1]
        report = metrics.classification_report(y_test, probabilities)
        fitted[name] = search.best_estimator_

        leaderboard.append(
            {
                "model": name,
                "inductive_bias": config["bias"],
                "cv_pr_auc_mean": round(float(search.best_score_), 5),
                "cv_pr_auc_std": round(
                    float(search.cv_results_["std_test_score"][search.best_index_]), 5
                ),
                "test_pr_auc": report["pr_auc"],
                "test_roc_auc": report["roc_auc"],
                "test_brier": report["brier"],
                "test_f1": report["f1"],
                "search_seconds": round(elapsed, 1),
                "best_params": {
                    k.replace("model__", ""): (
                        round(float(v), 6) if isinstance(v, float) else v
                    )
                    for k, v in search.best_params_.items()
                },
                "generalisation_gap": round(
                    float(search.best_score_) - report["pr_auc"], 5
                ),
            }
        )
        print(f"      {name:<26} CV {search.best_score_:.4f} → "
              f"test {report['pr_auc']:.4f}  ({elapsed:.0f}s)")

    leaderboard.sort(key=lambda r: -r["test_pr_auc"])

    # --- Stacked ensemble ---------------------------------------------------------------
    # Meta-learner trained on cross-validated out-of-fold predictions, never on in-fold
    # predictions. Using in-fold predictions would let the meta-learner see base-model
    # outputs on rows those models had memorised, which is leakage one level up and a
    # standard way stacking ensembles get silently overfitted.
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression

    top_three = [row["model"] for row in leaderboard[:3]]
    stack = StackingClassifier(
        estimators=[(name, fitted[name]) for name in top_three],
        final_estimator=LogisticRegression(max_iter=2000, random_state=SEED),
        cv=cv,
        n_jobs=-1,
    )
    started = time.perf_counter()
    stack.fit(X_train, y_train)
    stack_report = metrics.classification_report(y_test, stack.predict_proba(X_test)[:, 1])
    leaderboard.append(
        {
            "model": "stacked_ensemble",
            "inductive_bias": f"logistic meta-learner over out-of-fold predictions of {', '.join(top_three)}",
            "cv_pr_auc_mean": None,
            "cv_pr_auc_std": None,
            "test_pr_auc": stack_report["pr_auc"],
            "test_roc_auc": stack_report["roc_auc"],
            "test_brier": stack_report["brier"],
            "test_f1": stack_report["f1"],
            "search_seconds": round(time.perf_counter() - started, 1),
            "best_params": {"base_models": top_three},
            "generalisation_gap": None,
        }
    )
    leaderboard.sort(key=lambda r: -r["test_pr_auc"])
    print(f"      {'stacked_ensemble':<26} test {stack_report['pr_auc']:.4f}")

    best = leaderboard[0]
    return {
        "label": label,
        "features_used": {"numeric": numeric, "categorical": categorical},
        "n_features": len(numeric) + len(categorical),
        "split": split_report.to_dict(),
        "leaderboard": leaderboard,
        "best": best,
        "prevalence": round(float(y_test.mean()), 6),
        "search_protocol": (
            f"{N_SEARCH_ITER} randomised hyperparameter draws per family, "
            f"{N_FOLDS}-fold stratified CV, scored by average precision. Preprocessing is "
            "inside the pipeline so it is refitted per fold."
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    crisp = CrispDm(
        project=PROJECT,
        business_question=(
            "Which model should the bank use to prioritise term-deposit calls — and how "
            "much of any reported performance is real rather than leaked?"
        ),
    )
    crisp.record(
        Phase(
            name="business_understanding",
            summary=(
                "The operational goal is ranking prospects so a fixed calling budget "
                "reaches the most likely subscribers. The methodological goal, which "
                "matters more here, is establishing what the model's performance actually "
                "is once a documented target leak is removed."
            ),
            decisions=[
                Decision(
                    question="Should 'duration' be used as a feature?",
                    choice="No — and the tournament is run both ways to price the mistake.",
                    rationale=(
                        "Call duration is unknown until the call is over, by which point "
                        "the outcome is effectively determined: a zero-second call is a "
                        "rejection. Using it produces a model that cannot be deployed, "
                        "because at scoring time — before dialling — the value does not "
                        "exist. UCI's documentation says so explicitly, and most published "
                        "work on this dataset uses it anyway."
                    ),
                    alternatives_rejected=[
                        "Include duration for 'benchmark purposes' — the resulting number "
                        "is not a forecast of anything obtainable in production.",
                    ],
                ),
            ],
            evidence={"leak_column": "duration", "protocol": "identical tournament run twice"},
        )
    )

    with artifacts.run(PROJECT, seed=SEED) as ctx:
        print("\n[1/3] Loading real bank marketing contacts …")
        df = pd.read_csv(data.fetch("bank_marketing"), sep=";")
        y = (df["y"] == "yes").astype(int).to_numpy()

        numeric_all = [
            c for c in df.select_dtypes(include=[np.number]).columns if c != "y"
        ]
        categorical = [
            c for c in df.select_dtypes(include=["object", "str"]).columns if c != "y"
        ]
        numeric_clean = [c for c in numeric_all if c != "duration"]

        print(f"      {len(df):,} contacts, positive rate {y.mean():.4%}")

        # How strongly does the leaking column alone predict the target? This is the
        # diagnostic that identifies a leak before any model is trained.
        from sklearn.metrics import roc_auc_score

        duration_alone_auc = float(roc_auc_score(y, df["duration"]))
        leak_diagnostic = {
            "column": "duration",
            "univariate_roc_auc": round(duration_alone_auc, 4),
            "correlation_with_target": round(
                float(np.corrcoef(df["duration"], y)[0, 1]), 4
            ),
            "mean_duration_subscribed": round(float(df.loc[y == 1, "duration"].mean()), 1),
            "mean_duration_declined": round(float(df.loc[y == 0, "duration"].mean()), 1),
            "zero_duration_rows": int((df["duration"] == 0).sum()),
            "zero_duration_subscriptions": int(((df["duration"] == 0) & (y == 1)).sum()),
            "why_this_is_a_leak": (
                f"A single column achieving ROC-AUC {duration_alone_auc:.3f} on its own is "
                "a red flag, not a feature. Subscribers average "
                f"{df.loc[y == 1, 'duration'].mean():.0f}s on the call against "
                f"{df.loc[y == 0, 'duration'].mean():.0f}s for those who decline — but "
                "that is a consequence of the outcome, not a predictor of it. The value "
                "cannot be known before the call is placed, which is exactly when a "
                "prioritisation model must score."
            ),
        }
        print(f"      leak diagnostic: 'duration' alone reaches ROC-AUC "
              f"{duration_alone_auc:.4f}")

        crisp.record(
            Phase(
                name="data_understanding",
                summary=(
                    f"{len(df):,} contacts, {y.mean():.2%} positive. The 'duration' column "
                    f"alone achieves ROC-AUC {duration_alone_auc:.3f} against the target — "
                    "the signature of a leak rather than of a strong feature."
                ),
                evidence={"leak_diagnostic": leak_diagnostic},
                decisions=[
                    Decision(
                        question="How is a leaking feature identified in the first place?",
                        choice="Univariate predictive power plus a causal-timing check.",
                        rationale=(
                            "Two questions catch most leaks. First, does any single column "
                            "predict the target implausibly well on its own? Second, and "
                            "decisively: would this value exist at the moment the "
                            "prediction has to be made? 'duration' fails the second "
                            "outright — it is generated by the very event being predicted."
                        ),
                    ),
                ],
            )
        )

        # --- Two tournaments under identical conditions ---------------------------------
        print("\n[2/3] Running tournament twice — with and without the leak …")
        X_all = df.drop(columns=["y"])
        clean = run_tournament(X_all, y, numeric_clean, categorical, "leak_free")
        leaked = run_tournament(X_all, y, numeric_all, categorical, "with_leak")

        inflation = {
            "clean_best_pr_auc": clean["best"]["test_pr_auc"],
            "leaked_best_pr_auc": leaked["best"]["test_pr_auc"],
            "absolute_inflation": round(
                leaked["best"]["test_pr_auc"] - clean["best"]["test_pr_auc"], 5
            ),
            "relative_inflation": round(
                leaked["best"]["test_pr_auc"] / max(1e-9, clean["best"]["test_pr_auc"]) - 1, 4
            ),
            "clean_best_model": clean["best"]["model"],
            "leaked_best_model": leaked["best"]["model"],
            "per_model": [
                {
                    "model": c["model"],
                    "clean_pr_auc": c["test_pr_auc"],
                    "leaked_pr_auc": next(
                        (l["test_pr_auc"] for l in leaked["leaderboard"] if l["model"] == c["model"]),
                        None,
                    ),
                }
                for c in clean["leaderboard"]
            ],
            "verdict": (
                f"Including one column that cannot exist at scoring time raises the best "
                f"PR-AUC from {clean['best']['test_pr_auc']:.4f} to "
                f"{leaked['best']['test_pr_auc']:.4f} — a "
                f"{(leaked['best']['test_pr_auc'] / max(1e-9, clean['best']['test_pr_auc']) - 1):.1%} "
                "relative gain that would evaporate entirely in production. Nothing in the "
                "cross-validation, the confusion matrix or the calibration curve flags it. "
                "Only reasoning about when each value becomes known does."
            ),
        }
        print(f"\n      LEAK INFLATION: {clean['best']['test_pr_auc']:.4f} → "
              f"{leaked['best']['test_pr_auc']:.4f} "
              f"({inflation['relative_inflation']:+.1%})")

        crisp.record(
            Phase(
                name="data_preparation",
                summary=(
                    "Median imputation and standardisation for numerics, one-hot encoding "
                    "with rare-level folding for categoricals — all inside a "
                    "ColumnTransformer within the pipeline, so each CV fold refits the "
                    "transforms on its own training rows. The only difference between the "
                    "two runs is the presence of one column."
                ),
                evidence={
                    "leak_free_features": clean["n_features"],
                    "with_leak_features": leaked["n_features"],
                },
            )
        )
        crisp.record(
            Phase(
                name="modeling",
                summary=(
                    f"Four model families with genuinely different inductive biases, "
                    f"{N_SEARCH_ITER} randomised hyperparameter draws each under "
                    f"{N_FOLDS}-fold stratified CV, plus a stacked ensemble over the top "
                    "three trained on out-of-fold predictions."
                ),
                decisions=[
                    Decision(
                        question="Why search families rather than tune one harder?",
                        choice="Four families spanning different inductive biases.",
                        rationale=(
                            "A search over boosting depths is a hyperparameter tuner, not "
                            "AutoML. Including a linear model and a deliberately naive "
                            "Bayes reference shows how much of the final score comes from "
                            "model capacity and how much was available from any reasonable "
                            "baseline."
                        ),
                    ),
                    Decision(
                        question="How is the stacking meta-learner trained?",
                        choice="On cross-validated out-of-fold predictions only.",
                        rationale=(
                            "Fitting the meta-learner on in-fold predictions lets it see "
                            "base-model outputs for rows those models effectively "
                            "memorised. The ensemble then learns to trust an accuracy that "
                            "does not exist out of sample — leakage one level up, and a "
                            "common way stacking silently overfits."
                        ),
                    ),
                ],
                evidence={"clean": clean, "with_leak": leaked},
            )
        )
        crisp.record(
            Phase(
                name="evaluation",
                summary=inflation["verdict"],
                evidence={"leak_inflation": inflation},
                risks=[
                    "The leak-free leaderboard is the deployable one. Any comparison "
                    "against published results on this dataset that include 'duration' is "
                    "not like-for-like.",
                    "Macroeconomic columns are retained in the leak-free run. They are "
                    "knowable at scoring time, but they make the model partly a function "
                    "of the economic cycle, so performance will drift as conditions change.",
                    f"Randomised search with {N_SEARCH_ITER} draws per family is a light "
                    "budget. The ranking between close families should be read as "
                    "provisional.",
                ],
            )
        )
        crisp.record(
            Phase(
                name="deployment",
                summary=(
                    "Both leaderboards ship so the comparison is inspectable, with the "
                    "leak-free run marked as the deployable result and the leaked run "
                    "labelled as a demonstration."
                ),
                evidence={"deployable_leaderboard": "leak_free"},
            )
        )

        print("\n[3/3] Writing artifacts …")
        artifacts.write(OUT / "leak_demonstration.json",
                        {"diagnostic": leak_diagnostic, "inflation": inflation}, context=ctx)
        artifacts.write(OUT / "tournament_clean.json", clean, context=ctx)
        artifacts.write(OUT / "tournament_leaked.json", leaked, context=ctx)
        artifacts.write(OUT / "crispdm.json", crisp.to_dict(), context=ctx)
        artifacts.write(OUT / "provenance.json",
                        {"datasets": [data.provenance_record("bank_marketing")]}, context=ctx)

    print(f"\n✓ {PROJECT} complete — leak inflates PR-AUC by "
          f"{inflation['relative_inflation']:+.1%}")


if __name__ == "__main__":
    main()
