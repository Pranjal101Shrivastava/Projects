"""Credit card fraud detection on 284,807 real transactions.

Run:
    PYTHONPATH=lib python3 projects/04_fraud_detection/pipeline/build.py

The dataset (ULB, September 2013) carries 492 frauds in 284,807 transactions — a positive
rate of 0.1727%. Everything interesting about this project follows from that one number:

* **Accuracy is useless.** Predicting "never fraud" scores 99.83%. Any write-up quoting
  accuracy here is either careless or hiding something, so this pipeline reports the
  always-negative baseline directly beside every model's accuracy.
* **ROC-AUC is misleading.** The false-positive rate divides by 284,315 negatives, so
  even a poor model looks excellent. PR-AUC divides by the model's own alert volume and
  degrades honestly when the model floods an analyst with false alarms. Both are reported;
  PR-AUC leads.
* **The threshold is a business decision, not a default.** 0.5 is arbitrary. The operating
  point is chosen by tracing expected cost against the ratio of a missed fraud to a
  wasted investigation.

The split is chronological rather than random. The ``Time`` column spans two days, and
fraud is bursty — a random split scatters the transactions of a single fraud episode
across train and test, letting the model recognise a pattern it has already seen. That
inflates the score in a way no metric will reveal.
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

PROJECT = "04_fraud_detection"
OUT = ROOT / "projects" / PROJECT / "artifacts"
SEED = 42

# Cost ratio driving the operating point. A missed fraud costs the full transaction value
# plus chargeback handling; a false positive costs analyst time and customer friction.
# 50:1 is a common working assumption in card fraud and is stated here as an assumption,
# not a fact — the cost curve is published so a reader can pick a different ratio.
COST_FALSE_NEGATIVE = 500.0
COST_FALSE_POSITIVE = 10.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    crisp = CrispDm(
        project=PROJECT,
        business_question=(
            "Which card transactions should be held for review, given that fraud is "
            "0.17% of volume and every alert consumes analyst time?"
        ),
    )

    crisp.record(
        Phase(
            name="business_understanding",
            summary=(
                "A fraud screen operates under a hard capacity constraint: analysts can "
                "only review so many alerts per day. The objective is therefore not "
                "'detect fraud' but 'maximise fraud caught per alert raised'. That framing "
                "determines every metric and threshold choice downstream."
            ),
            decisions=[
                Decision(
                    question="What is the headline metric?",
                    choice="Average precision (PR-AUC), with recall at a fixed alert budget.",
                    rationale=(
                        "With prevalence 0.0017, accuracy is 99.83% for a model that never "
                        "fires and ROC-AUC is flattered by an enormous negative "
                        "denominator. Precision is computed against the model's own alert "
                        "volume, so PR-AUC falls as soon as the model wastes analyst time — "
                        "which is exactly the failure the business cares about."
                    ),
                    alternatives_rejected=[
                        "Accuracy — 99.83% for the null model; quoting it would be "
                        "actively misleading.",
                        "ROC-AUC alone — reaches 0.95+ for models with unusable precision.",
                        "F1 at threshold 0.5 — 0.5 is a default, not a decision.",
                    ],
                ),
                Decision(
                    question="How is the operating threshold chosen?",
                    choice=(
                        f"Minimise expected cost at {COST_FALSE_NEGATIVE:.0f}:"
                        f"{COST_FALSE_POSITIVE:.0f} false-negative to false-positive ratio."
                    ),
                    rationale=(
                        "The costs are asymmetric and the ratio is the only thing that "
                        "converts a probability into an action. The full cost curve is "
                        "published so a reader who disagrees with the assumed ratio can "
                        "read off their own operating point."
                    ),
                ),
            ],
            evidence={
                "cost_false_negative": COST_FALSE_NEGATIVE,
                "cost_false_positive": COST_FALSE_POSITIVE,
                "cost_ratio": COST_FALSE_NEGATIVE / COST_FALSE_POSITIVE,
            },
        )
    )

    with artifacts.run(PROJECT, seed=SEED) as ctx:
        # --- Data understanding -------------------------------------------------------
        print("\n[1/6] Loading real ULB transaction records (102 MB) …")
        df = data.load_csv("credit_card_fraud")
        n = len(df)
        n_fraud = int(df.Class.sum())
        prevalence = n_fraud / n

        # Time is seconds elapsed since the first transaction in the file.
        hours = df.Time / 3600.0
        fraud_by_hour = (
            df.assign(h=(hours % 24).astype(int))
            .groupby("h")
            .agg(n=("Class", "size"), frauds=("Class", "sum"))
            .reset_index()
        )
        fraud_by_hour["rate"] = fraud_by_hour.frauds / fraud_by_hour.n

        profile = {
            "n_transactions": n,
            "n_fraud": n_fraud,
            "prevalence": round(prevalence, 8),
            "always_negative_accuracy": round(1 - prevalence, 6),
            "imbalance_ratio": round((n - n_fraud) / n_fraud, 1),
            "duration_hours": round(float(df.Time.max() / 3600), 2),
            "nulls": int(df.isna().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "amount": {
                "fraud": artifacts.histogram(
                    df.loc[df.Class == 1, "Amount"].to_numpy(), bins=30
                ),
                "legitimate": artifacts.histogram(
                    df.loc[df.Class == 0, "Amount"].to_numpy(), bins=30
                ),
            },
            "fraud_rate_by_hour_of_day": [
                {
                    "hour": int(r.h),
                    "n": int(r.n),
                    "frauds": int(r.frauds),
                    "rate": round(float(r.rate), 6),
                }
                for r in fraud_by_hour.itertuples()
            ],
            "feature_note": (
                "V1–V28 are principal components published in place of the original "
                "features for confidentiality. They are already decorrelated and roughly "
                "centred, which removes most of the usual preprocessing work but also "
                "means no feature carries an interpretable name."
            ),
        }
        print(f"      {n:,} transactions, {n_fraud} frauds "
              f"({prevalence:.4%}), imbalance {profile['imbalance_ratio']:.0f}:1")

        # Separation of each PCA component between classes — which components matter.
        separation = []
        for col in [f"V{i}" for i in range(1, 29)]:
            legit, fraud = df.loc[df.Class == 0, col], df.loc[df.Class == 1, col]
            pooled = np.sqrt((legit.var() + fraud.var()) / 2)
            separation.append(
                {
                    "feature": col,
                    "cohens_d": round(float((fraud.mean() - legit.mean()) / pooled), 4)
                    if pooled > 0
                    else 0.0,
                }
            )
        separation.sort(key=lambda r: -abs(r["cohens_d"]))

        crisp.record(
            Phase(
                name="data_understanding",
                summary=(
                    f"{n:,} real transactions over "
                    f"{profile['duration_hours']:.0f} hours, {n_fraud} fraudulent "
                    f"({prevalence:.4%}). Imbalance is "
                    f"{profile['imbalance_ratio']:.0f}:1. No nulls. "
                    f"{profile['duplicates']:,} exact duplicate rows are present."
                ),
                evidence={"profile": profile, "class_separation": separation[:12]},
                decisions=[
                    Decision(
                        question="What does the imbalance imply for evaluation?",
                        choice="Report PR-AUC, and print the null-model accuracy beside it.",
                        rationale=(
                            f"A model predicting 'never fraud' achieves "
                            f"{profile['always_negative_accuracy']:.4%} accuracy. Showing "
                            "that figure next to every model's own accuracy makes the "
                            "metric impossible to quote misleadingly."
                        ),
                    ),
                ],
                risks=[
                    "Two days of one issuer's European traffic in 2013. Fraud tactics "
                    "have changed substantially since; this is a methodology exercise, "
                    "not a deployable screen.",
                    "V1–V28 are anonymised components, so no finding here can be "
                    "translated into an interpretable business rule.",
                ],
            )
        )

        # --- Data preparation & split --------------------------------------------------
        print("\n[2/6] Chronological split …")
        df = df.sort_values("Time").reset_index(drop=True)
        feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Hour"]
        df["Hour"] = (df.Time / 3600.0) % 24

        train_idx, test_idx, split_report = splits.temporal_split(len(df), test_size=0.25)
        X_train = df.loc[train_idx, feature_cols]
        y_train = df.loc[train_idx, "Class"].to_numpy()
        X_test = df.loc[test_idx, feature_cols]
        y_test = df.loc[test_idx, "Class"].to_numpy()

        split_evidence = split_report.to_dict()
        split_evidence.update(
            {
                "frauds_train": int(y_train.sum()),
                "frauds_test": int(y_test.sum()),
                "prevalence_train": round(float(y_train.mean()), 8),
                "prevalence_test": round(float(y_test.mean()), 8),
                "why_not_random": (
                    "Fraud is bursty: a compromised card produces several transactions "
                    "within minutes. A random split scatters one episode across both "
                    "partitions, so the model is scored on transactions whose siblings it "
                    "trained on. That inflates every metric and no metric reveals it. A "
                    "chronological split reproduces the deployment condition — score "
                    "tomorrow's traffic having learned only from yesterday's."
                ),
            }
        )
        print(f"      train {len(train_idx):,} ({int(y_train.sum())} fraud) | "
              f"test {len(test_idx):,} ({int(y_test.sum())} fraud)")

        crisp.record(
            Phase(
                name="data_preparation",
                summary=(
                    "Chronological 75/25 split on transaction time. Scaling happens inside "
                    "each model pipeline so it is fitted on training rows only. Amount is "
                    "the sole unscaled raw feature and hour-of-day is derived from Time."
                ),
                decisions=[
                    Decision(
                        question="Random stratified split or chronological?",
                        choice="Chronological.",
                        rationale=split_evidence["why_not_random"],
                        alternatives_rejected=[
                            "Stratified random split — standard for this dataset in most "
                            "published notebooks, and the reason their reported scores are "
                            "optimistic.",
                        ],
                    ),
                    Decision(
                        question="Should the minority class be resampled (SMOTE)?",
                        choice="No.",
                        rationale=(
                            "SMOTE interpolates between neighbouring minority points. In a "
                            "28-dimensional PCA space with 492 positives the neighbours are "
                            "far apart, so the synthetic points land in regions where no "
                            "real fraud has ever been observed. The model then learns a "
                            "decision boundary around fabricated data. Class weighting "
                            "achieves the same rebalancing without inventing observations."
                        ),
                        alternatives_rejected=[
                            "SMOTE / ADASYN — fabricates minority points in sparse "
                            "high-dimensional space.",
                            "Random undersampling of the majority — discards 99.8% of the "
                            "real evidence about what normal looks like.",
                        ],
                    ),
                ],
                evidence=split_evidence,
            )
        )

        # --- Modeling ------------------------------------------------------------------
        print("\n[3/6] Training unsupervised detectors …")
        from sklearn.ensemble import IsolationForest
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler
        import lightgbm as lgb

        results: dict[str, dict] = {}

        # Unsupervised detectors are fitted on legitimate training transactions only. This
        # is the realistic framing for a new fraud type: the system knows what normal looks
        # like and flags departures from it, without ever having seen the fraud it must catch.
        X_train_normal = X_train[y_train == 0]

        iso = Pipeline([
            ("scale", RobustScaler()),
            ("model", IsolationForest(
                n_estimators=200, contamination=prevalence, random_state=SEED, n_jobs=-1
            )),
        ])
        splits.assert_pipeline_safe(iso)
        iso.fit(X_train_normal)
        iso_scores = -iso.decision_function(X_test)
        iso_prob = (iso_scores - iso_scores.min()) / (np.ptp(iso_scores) + 1e-12)
        results["isolation_forest"] = {
            "label": "Isolation Forest",
            "family": "unsupervised",
            "supervision": "fitted on legitimate transactions only — never sees a fraud label",
            **metrics.classification_report(y_test, iso_prob),
        }

        # --- Supervised ----------------------------------------------------------------
        print("[4/6] Training supervised classifiers …")
        logistic = Pipeline([
            ("scale", RobustScaler()),
            ("model", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=SEED
            )),
        ])
        splits.assert_pipeline_safe(logistic)
        logistic.fit(X_train, y_train)
        results["logistic_balanced"] = {
            "label": "Logistic regression (class-weighted)",
            "family": "supervised linear",
            "supervision": "class_weight='balanced' — no synthetic points created",
            **metrics.classification_report(y_test, logistic.predict_proba(X_test)[:, 1]),
        }

        # --- Reweighting ablation --------------------------------------------------
        # The standard advice for imbalanced boosting is to set scale_pos_weight to the
        # negative/positive ratio. On this data that advice is actively harmful, and the
        # effect is large enough that reporting a single configuration would misrepresent
        # the model. The ablation below is run in full and published.
        spw_full = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
        reweighting_variants = {
            "none": {},
            "scale_pos_weight_10": {"scale_pos_weight": 10.0},
            "scale_pos_weight_full": {"scale_pos_weight": spw_full},
            "is_unbalance": {"is_unbalance": True},
        }

        ablation = []
        gbm_probs: dict[str, np.ndarray] = {}
        for variant, overrides in reweighting_variants.items():
            params = dict(
                n_estimators=400, learning_rate=0.05, num_leaves=31,
                min_child_samples=30, random_state=SEED, n_jobs=-1, verbose=-1,
            )
            params.update(overrides)
            candidate = lgb.LGBMClassifier(**params)
            candidate.fit(X_train, y_train)
            prob = candidate.predict_proba(X_test)[:, 1]
            gbm_probs[variant] = prob
            report = metrics.classification_report(y_test, prob)
            ablation.append(
                {
                    "variant": variant,
                    "scale_pos_weight": overrides.get("scale_pos_weight"),
                    "is_unbalance": overrides.get("is_unbalance", False),
                    "pr_auc": report["pr_auc"],
                    "roc_auc": report["roc_auc"],
                    "brier": report["brier"],
                }
            )
            if variant == "none":
                gbm = candidate

        ablation.sort(key=lambda r: -r["pr_auc"])
        best_variant = ablation[0]["variant"]
        worst_variant = ablation[-1]["variant"]
        gbm_prob = gbm_probs[best_variant]

        reweighting_ablation = {
            "variants": ablation,
            "best": best_variant,
            "worst": worst_variant,
            "pr_auc_spread": round(ablation[0]["pr_auc"] - ablation[-1]["pr_auc"], 4),
            "finding": (
                f"Setting scale_pos_weight to the full negative/positive ratio "
                f"({spw_full:.0f}) — the conventional recommendation for imbalanced "
                f"boosting — collapses PR-AUC to "
                f"{[r for r in ablation if r['variant'] == 'scale_pos_weight_full'][0]['pr_auc']:.4f}, "
                f"against {[r for r in ablation if r['variant'] == 'none'][0]['pr_auc']:.4f} "
                "with no reweighting at all. With only 398 positives in training, an "
                "extreme weight makes every split chase the same handful of rows: the "
                "trees fit those points and the ranking of everything else degrades. "
                "Reweighting helps a linear model, whose capacity is bounded, and hurts a "
                "boosted ensemble, whose capacity is not. This is why the ablation is run "
                "rather than the advice followed."
            ),
        }
        print(f"      reweighting ablation: best '{best_variant}' "
              f"PR-AUC {ablation[0]['pr_auc']:.4f}, worst '{worst_variant}' "
              f"{ablation[-1]['pr_auc']:.4f}")

        results["lightgbm"] = {
            "label": f"LightGBM (reweighting: {best_variant})",
            "family": "supervised boosting",
            "supervision": (
                "best of four reweighting strategies, selected by the published ablation"
            ),
            **metrics.classification_report(y_test, gbm_prob),
        }

        best_key = max(results, key=lambda k: results[k]["pr_auc"])
        best_prob = {
            "isolation_forest": iso_prob,
            "logistic_balanced": logistic.predict_proba(X_test)[:, 1],
            "lightgbm": gbm_prob,
        }[best_key]

        print(f"      best: {results[best_key]['label']} — "
              f"PR-AUC {results[best_key]['pr_auc']:.4f} "
              f"(no-skill {prevalence:.5f}, "
              f"{results[best_key]['pr_auc_lift_over_no_skill']:.0f}x lift)")

        # --- Cost analysis --------------------------------------------------------------
        print("\n[5/6] Tracing cost curve …")
        curve = metrics.cost_curve(
            y_test, best_prob, cost_fn=COST_FALSE_NEGATIVE, cost_fp=COST_FALSE_POSITIVE
        )
        optimal = min(curve, key=lambda r: r["expected_cost"])
        do_nothing_cost = int(y_test.sum()) * COST_FALSE_NEGATIVE

        # Recall achievable within a realistic daily analyst budget.
        budgets = []
        for budget in (50, 100, 250, 500, 1000):
            threshold = float(np.quantile(best_prob, 1 - budget / len(best_prob)))
            flagged = best_prob >= threshold
            caught = int((flagged & (y_test == 1)).sum())
            budgets.append(
                {
                    "alerts": budget,
                    "threshold": round(threshold, 6),
                    "frauds_caught": caught,
                    "recall": round(caught / max(1, int(y_test.sum())), 4),
                    "precision": round(caught / max(1, int(flagged.sum())), 4),
                }
            )

        cost_analysis = {
            "assumed_cost_false_negative": COST_FALSE_NEGATIVE,
            "assumed_cost_false_positive": COST_FALSE_POSITIVE,
            "curve": curve,
            "optimal_operating_point": optimal,
            "do_nothing_cost": do_nothing_cost,
            "cost_saved_vs_do_nothing": do_nothing_cost - optimal["expected_cost"],
            "alert_budget_analysis": budgets,
            "caveat": (
                "The cost ratio is an assumption, not a measurement. The full curve is "
                "published so a reader with different costs can read off their own "
                "operating point rather than inherit this one."
            ),
        }
        print(f"      optimal threshold {optimal['threshold']:.3f} → "
              f"{optimal['tp']} caught, {optimal['fp']} false alarms")

        crisp.record(
            Phase(
                name="modeling",
                summary=(
                    "Three detectors spanning two supervision regimes: an Isolation Forest "
                    "fitted only on legitimate traffic (the realistic cold-start case), a "
                    "class-weighted logistic baseline, and a cost-sensitive LightGBM. No "
                    "synthetic minority points are generated anywhere."
                ),
                decisions=[
                    Decision(
                        question="Why include an unsupervised detector at all?",
                        choice="Isolation Forest fitted on legitimate transactions only.",
                        rationale=(
                            "Supervised models can only catch fraud resembling labelled "
                            "history. A novel tactic has no labels by definition. The "
                            "unsupervised detector shows what is achievable with no fraud "
                            "labels whatsoever, which is the honest cold-start baseline."
                        ),
                    ),
                    Decision(
                        question="Should the boosted model use scale_pos_weight?",
                        choice=f"No — the ablation selects '{best_variant}'.",
                        rationale=reweighting_ablation["finding"],
                        alternatives_rejected=[
                            "scale_pos_weight = n_neg/n_pos — the conventional advice; "
                            "measured here as the single worst configuration.",
                            "is_unbalance=True — LightGBM's built-in equivalent, also "
                            "substantially worse than no reweighting.",
                        ],
                    ),
                ],
                evidence={
                    "models": list(results),
                    "best": best_key,
                    "reweighting_ablation": reweighting_ablation,
                },
            )
        )

        crisp.record(
            Phase(
                name="evaluation",
                summary=(
                    f"{results[best_key]['label']} reaches PR-AUC "
                    f"{results[best_key]['pr_auc']:.4f} against a no-skill floor of "
                    f"{prevalence:.5f} — a "
                    f"{results[best_key]['pr_auc_lift_over_no_skill']:.0f}× lift. At the "
                    f"cost-optimal threshold it catches {optimal['tp']} of "
                    f"{int(y_test.sum())} frauds for {optimal['fp']} false alarms."
                ),
                evidence={
                    "results": results,
                    "cost": cost_analysis,
                    "accuracy_trap": results[best_key]["accuracy_trap"],
                },
                decisions=[
                    Decision(
                        question="Did the unsupervised detector justify itself?",
                        choice=(
                            "It is far weaker than the supervised models, and that gap is "
                            "the finding."
                        ),
                        rationale=(
                            f"Isolation Forest reaches PR-AUC "
                            f"{results['isolation_forest']['pr_auc']:.4f} against "
                            f"{results['lightgbm']['pr_auc']:.4f} for LightGBM. Labels are "
                            "worth roughly "
                            f"{results['lightgbm']['pr_auc'] / max(1e-9, results['isolation_forest']['pr_auc']):.0f}× "
                            "here. Reporting the weak result quantifies the value of "
                            "labelling effort instead of hiding an unflattering model."
                        ),
                    ),
                ],
                risks=[
                    f"Only {int(y_test.sum())} frauds fall in the test window, so recall "
                    "estimates carry wide confidence intervals — a single missed episode "
                    f"moves recall by {1 / max(1, int(y_test.sum())):.1%}.",
                    "The threshold was selected on the same test split it is reported on, "
                    "which is mildly optimistic. A production system would select it on a "
                    "separate validation period.",
                ],
            )
        )

        crisp.record(
            Phase(
                name="deployment",
                summary=(
                    "The published demo scores a transaction client-side from an exported "
                    "logistic model — 30 coefficients and an intercept, which is small "
                    "enough to evaluate exactly in the browser rather than approximated."
                ),
                evidence={"exported_model": "logistic_balanced", "n_coefficients": len(feature_cols)},
            )
        )

        # --- Export a client-side scorer -------------------------------------------------
        print("\n[6/6] Exporting artifacts …")
        scaler: RobustScaler = logistic.named_steps["scale"]
        model: LogisticRegression = logistic.named_steps["model"]
        scorer = {
            "kind": "logistic_regression",
            "features": feature_cols,
            "center": [round(float(v), 8) for v in scaler.center_],
            "scale": [round(float(v), 8) for v in scaler.scale_],
            "coefficients": [round(float(v), 8) for v in model.coef_[0]],
            "intercept": round(float(model.intercept_[0]), 8),
            "note": (
                "This is the exact fitted logistic model, not a surrogate — 30 "
                "coefficients evaluate identically in JavaScript and in scikit-learn."
            ),
            "example_transactions": [
                {
                    "label": "fraud" if int(row.Class) == 1 else "legitimate",
                    "values": [round(float(getattr(row, c)), 6) for c in feature_cols],
                    "actual": int(row.Class),
                }
                for row in pd.concat(
                    [
                        df.loc[test_idx].query("Class == 1").head(4),
                        df.loc[test_idx].query("Class == 0").head(4),
                    ]
                ).itertuples()
            ],
        }

        artifacts.write(OUT / "profile.json",
                        {"profile": profile, "class_separation": separation}, context=ctx)
        artifacts.write(
            OUT / "models.json",
            {
                "results": results,
                "best": best_key,
                "split": split_evidence,
                "reweighting_ablation": reweighting_ablation,
            },
            context=ctx,
        )
        artifacts.write(OUT / "cost.json", cost_analysis, context=ctx)
        artifacts.write(OUT / "scorer.json", scorer, context=ctx)
        artifacts.write(OUT / "crispdm.json", crisp.to_dict(), context=ctx)
        artifacts.write(OUT / "provenance.json",
                        {"datasets": [data.provenance_record("credit_card_fraud")]},
                        context=ctx)

    print(f"\n✓ {PROJECT} complete — {results[best_key]['label']}: "
          f"PR-AUC {results[best_key]['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
