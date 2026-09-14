"""Evaluation metrics, chosen to be hard to fool.

The recurring theme here is that the default metric is often the wrong one. Accuracy on a
0.17% positive rate is 99.83% for a model that predicts "never fraud". R-squared on a
trending series is high for a model that just repeats yesterday. Each helper therefore
reports a *baseline alongside the score*, so a number can be read as "better than doing
nothing by this much" rather than as a bare figure that sounds impressive.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)


def classification_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float | None = None,
) -> dict:
    """Score a binary classifier, leading with PR-AUC rather than accuracy.

    For a rare positive class, ROC-AUC is optimistic: the false-positive rate has a huge
    denominator, so even a poor model looks strong. Average precision (PR-AUC) uses
    precision instead, whose denominator is the model's own alert volume, and so degrades
    honestly when the model floods the operator with false alarms.

    ``prevalence`` is included because it *is* the no-skill PR-AUC. A model scoring 0.80
    against a prevalence of 0.0017 has done something; one scoring 0.80 against a
    prevalence of 0.75 has not.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    prevalence = float(y_true.mean())

    if threshold is None:
        threshold = _best_f1_threshold(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "pr_auc_no_skill": prevalence,
        "pr_auc_lift_over_no_skill": (
            float(average_precision_score(y_true, y_prob) / prevalence)
            if prevalence
            else None
        ),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision),
        "recall": float(recall),
        "threshold": float(threshold),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "prevalence": prevalence,
        "n": int(len(y_true)),
        "accuracy_trap": {
            "model_accuracy": float((y_pred == y_true).mean()),
            "always_negative_accuracy": float(1 - prevalence),
            "note": (
                "Both figures are shown so accuracy cannot be quoted as evidence of "
                "skill. A model that never fires matches the second number."
            ),
        },
    }


def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold maximising F1 on the supplied data.

    Note the caveat: choosing a threshold on the same data you report is itself a mild
    form of optimism. Callers that care select the threshold on a validation split and
    pass it in explicitly.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    denom = precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    if len(thresholds) == 0:
        return 0.5
    return float(thresholds[max(0, int(np.argmax(f1)) - 1)])


def cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    cost_fn: float,
    cost_fp: float,
    n_points: int = 100,
) -> list[dict]:
    """Expected cost as a function of alert threshold.

    A fraud model is not deployed at "the best F1". It is deployed wherever the cost of a
    missed fraud and the cost of investigating a false alert balance. This traces that
    trade-off explicitly so the operating point is an argued choice rather than a default.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    out = []
    for threshold in np.linspace(0.0, 1.0, n_points):
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        out.append(
            {
                "threshold": round(float(threshold), 4),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "expected_cost": float(fn * cost_fn + fp * cost_fp),
                "alerts": int(tp + fp),
            }
        )
    return out


def regression_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    baseline: np.ndarray | None = None,
) -> dict:
    """Score a regressor against an explicit baseline.

    ``baseline`` should be the naive prediction a stakeholder would make without a model
    - the training mean, or last-observed-value for a time series. Skill score is the
    fractional reduction in MAE against it, which is the number that actually answers
    "was the model worth building".
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    result = {
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "mape": _safe_mape(y_true, y_pred),
        "n": int(len(y_true)),
    }

    if baseline is not None:
        baseline = np.asarray(baseline, dtype=float)
        baseline_mae = float(mean_absolute_error(y_true, baseline))
        result["baseline_mae"] = baseline_mae
        result["baseline_rmse"] = float(np.sqrt(mean_squared_error(y_true, baseline)))
        result["skill_score_vs_baseline"] = (
            float(1 - mae / baseline_mae) if baseline_mae else None
        )
    return result


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    """MAPE, or ``None`` where the series crosses zero.

    MAPE is undefined at zero and explodes near it. Returning ``None`` is more honest
    than clipping the denominator and reporting a number that looks finite but is not
    meaningful.
    """
    mask = np.abs(y_true) > 1e-9
    if mask.sum() < len(y_true) * 0.95:
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    nominal: float,
) -> dict:
    """Empirical coverage of a prediction interval against its nominal level.

    A 90% interval that covers 62% of outcomes is not a 90% interval. Reporting realised
    coverage next to the claimed level is the only way an interval can be falsified.
    """
    y_true = np.asarray(y_true, dtype=float)
    inside = (y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))
    covered = float(inside.mean())
    return {
        "nominal_coverage": nominal,
        "empirical_coverage": covered,
        "coverage_gap": round(covered - nominal, 4),
        "mean_interval_width": float(np.mean(np.asarray(upper) - np.asarray(lower))),
        "calibrated": abs(covered - nominal) <= 0.05,
    }


def clustering_report(X: np.ndarray, labels: np.ndarray) -> dict:
    """Internal cluster validity indices.

    All three disagree often, which is the point: clustering has no ground truth, so a
    single index is easy to over-read. Silhouette rewards separation, Calinski-Harabasz
    rewards compactness relative to spread, Davies-Bouldin penalises overlap (lower is
    better). A partition that wins on all three is genuinely defensible.
    """
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    n_clusters = len(set(labels.tolist())) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return {"n_clusters": n_clusters, "note": "Fewer than two clusters; indices undefined."}

    return {
        "n_clusters": int(n_clusters),
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "cluster_sizes": {
            str(label): int((labels == label).sum()) for label in sorted(set(labels.tolist()))
        },
    }
