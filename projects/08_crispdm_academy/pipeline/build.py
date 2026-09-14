"""CRISP-DM Academy — interactive teaching material computed from real data.

Run:
    PYTHONPATH=lib python3 projects/08_crispdm_academy/pipeline/build.py

Statistics teaching usually illustrates its concepts with generated data, because generated
data behaves. The cost is that students learn what a concept looks like when the
assumptions hold, and then meet real data where they do not.

Everything here is computed from the real datasets used elsewhere in this portfolio. That
means some of the illustrations are *messier* than a textbook figure — the Naive Bayes
independence assumption is measurably violated on Titanic, and the bias-variance curve does
not form a clean U. Those departures are the teaching content, not a defect in it.

Six modules, each exporting the data behind a live widget:

1. **Bayes and independence** — Naive Bayes on Titanic, with the independence assumption
   quantified rather than assumed.
2. **Threshold and cost** — confusion matrices, ROC and PR curves traced from real fraud
   model scores, showing why the two curves disagree under imbalance.
3. **Gradient descent** — a real loss surface with trajectories at several learning rates,
   including one that diverges.
4. **Backpropagation** — analytic gradients checked against finite differences, showing
   the chain rule is arithmetic rather than magic.
5. **Bias and variance** — polynomial degree sweep with the decomposition measured over
   bootstrap resamples.
6. **Sampling and uncertainty** — the central limit theorem demonstrated on a real,
   strongly skewed distribution.
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

PROJECT = "08_crispdm_academy"
OUT = ROOT / "projects" / PROJECT / "artifacts"
SEED = 42


# ======================================================================================
# Module 1 — Bayes and the independence assumption
# ======================================================================================
def module_bayes() -> dict:
    """Naive Bayes on real Titanic data, with the naive assumption measured.

    Naive Bayes assumes features are conditionally independent given the class. That is
    almost never true, and on Titanic it is visibly false: fare and passenger class are
    strongly dependent. The interesting pedagogical point is that the classifier still
    works reasonably well despite the violated assumption, because ranking only needs the
    posterior ordering to be right, not its calibration.
    """
    df = data.load_csv("titanic")
    df = df[["Survived", "Pclass", "Sex", "Age", "Fare", "Embarked", "SibSp", "Parch"]].copy()
    df["Age"] = df.Age.fillna(df.Age.median())
    df["Embarked"] = df.Embarked.fillna("S")
    df = df.dropna()

    y = df.Survived.to_numpy()
    prior = {
        "survived": round(float(y.mean()), 4),
        "died": round(float(1 - y.mean()), 4),
    }

    # Conditional distributions for the interactive calculator.
    likelihoods = {}
    for feature in ("Pclass", "Sex", "Embarked"):
        table = {}
        for value in sorted(df[feature].unique(), key=str):
            mask = df[feature] == value
            table[str(value)] = {
                "p_given_survived": round(
                    float((mask & (y == 1)).sum() / max(1, (y == 1).sum())), 5
                ),
                "p_given_died": round(
                    float((mask & (y == 0)).sum() / max(1, (y == 0).sum())), 5
                ),
                "count": int(mask.sum()),
            }
        likelihoods[feature] = table

    # How badly is conditional independence violated? Cramér's V between feature pairs,
    # computed within each class.
    from scipy.stats import chi2_contingency

    def cramers_v(a: pd.Series, b: pd.Series) -> float:
        table = pd.crosstab(a, b)
        if table.shape[0] < 2 or table.shape[1] < 2:
            return 0.0
        chi2 = chi2_contingency(table)[0]
        n = table.to_numpy().sum()
        return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))

    df["FareBand"] = pd.qcut(df.Fare, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    df["AgeBand"] = pd.cut(df.Age, [0, 12, 20, 40, 60, 100],
                           labels=["child", "teen", "adult", "middle", "senior"])

    pairs = [("Pclass", "FareBand"), ("Pclass", "Sex"), ("Sex", "AgeBand"),
             ("Pclass", "Embarked"), ("FareBand", "Embarked")]
    dependence = []
    for a, b in pairs:
        within_class = [
            cramers_v(df.loc[y == c, a], df.loc[y == c, b]) for c in (0, 1)
        ]
        dependence.append(
            {
                "pair": f"{a} × {b}",
                "cramers_v_given_died": round(within_class[0], 4),
                "cramers_v_given_survived": round(within_class[1], 4),
                "violates_independence": bool(max(within_class) > 0.2),
            }
        )
    dependence.sort(key=lambda r: -max(r["cramers_v_given_died"], r["cramers_v_given_survived"]))

    # Does the violation actually hurt? Compare against logistic regression, which makes
    # no independence assumption.
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import CategoricalNB
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

    features = ["Pclass", "Sex", "Embarked", "AgeBand", "FareBand"]
    X = df[features].astype(str)
    train_idx, test_idx, split = splits.stratified_split(y, test_size=0.3, seed=SEED)

    nb = Pipeline([
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ("model", CategoricalNB(alpha=1.0, min_categories=10)),
    ])
    nb.fit(X.iloc[train_idx], y[train_idx])
    nb_report = metrics.classification_report(
        y[test_idx], nb.predict_proba(X.iloc[test_idx])[:, 1]
    )

    lr = Pipeline([
        ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])
    lr.fit(X.iloc[train_idx], y[train_idx])
    lr_report = metrics.classification_report(
        y[test_idx], lr.predict_proba(X.iloc[test_idx])[:, 1]
    )

    violated = sum(1 for d in dependence if d["violates_independence"])
    return {
        "title": "Bayes' theorem and the independence assumption",
        "dataset": "Titanic (891 real passengers)",
        "prior": prior,
        "likelihoods": likelihoods,
        "dependence_check": dependence,
        "n_pairs_violating": violated,
        "comparison": {
            "naive_bayes": {"roc_auc": nb_report["roc_auc"], "brier": nb_report["brier"]},
            "logistic_regression": {"roc_auc": lr_report["roc_auc"], "brier": lr_report["brier"]},
        },
        "lesson": (
            f"Naive Bayes assumes features are independent given the class. On this data "
            f"{violated} of {len(dependence)} tested pairs violate that assumption "
            f"materially — passenger class and fare band are near-deterministic in each "
            f"other. The classifier still reaches ROC-AUC {nb_report['roc_auc']:.3f} "
            f"against {lr_report['roc_auc']:.3f} for logistic regression, which makes no "
            f"such assumption. The reason is that ranking depends only on the *ordering* "
            f"of posteriors, and dependence tends to distort their magnitudes more than "
            f"their order. That is also why Naive Bayes probabilities should not be "
            f"trusted as probabilities: its Brier score is {nb_report['brier']:.4f} "
            f"against {lr_report['brier']:.4f}."
        ),
        "quiz": [
            {
                "question": (
                    "Naive Bayes assumes P(A,B|C) = P(A|C)·P(B|C). On this data, Pclass "
                    "and FareBand are strongly dependent given survival. What is the most "
                    "likely consequence?"
                ),
                "options": [
                    "The classifier's ranking collapses to random",
                    "Predicted probabilities become over-confident, but ranking largely survives",
                    "The model cannot be fitted at all",
                    "Nothing — the assumption is only needed for training speed",
                ],
                "answer": 1,
                "explanation": (
                    "Dependent features contribute correlated evidence that the model "
                    "counts as though it were independent, so posteriors are pushed toward "
                    "0 and 1. Ordering is largely preserved, which is why ROC-AUC stays "
                    "respectable while the Brier score degrades."
                ),
            },
            {
                "question": (
                    f"The prior P(survived) is {prior['survived']:.3f}. A passenger is "
                    "female in first class. Which quantity does Bayes' theorem use to "
                    "convert that evidence into a posterior?"
                ),
                "options": [
                    "The likelihood P(evidence | survived) and the marginal P(evidence)",
                    "Only the prior",
                    "The accuracy of the classifier",
                    "The number of features",
                ],
                "answer": 0,
                "explanation": (
                    "Posterior ∝ prior × likelihood. The marginal P(evidence) normalises "
                    "so the posteriors across classes sum to one."
                ),
            },
        ],
    }


# ======================================================================================
# Module 2 — Thresholds, ROC and cost
# ======================================================================================
def module_threshold() -> dict:
    """ROC and PR curves from a real, severely imbalanced problem.

    Traced from actual fraud model scores rather than an idealised curve. The point is
    that ROC and PR give sharply different impressions of the *same* model when the
    positive class is rare, and only one of them tracks what an operator experiences.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve, roc_curve
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler

    df = data.load_csv("credit_card_fraud").sort_values("Time").reset_index(drop=True)
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    y = df.Class.to_numpy()

    train_idx, test_idx, split = splits.temporal_split(len(df), test_size=0.25)
    model = Pipeline([
        ("scale", RobustScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    ])
    splits.assert_pipeline_safe(model)
    model.fit(df.loc[train_idx, feature_cols], y[train_idx])
    scores = model.predict_proba(df.loc[test_idx, feature_cols])[:, 1]
    y_test = y[test_idx]

    fpr, tpr, roc_thresholds = roc_curve(y_test, scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_test, scores)

    step_roc = max(1, len(fpr) // 200)
    step_pr = max(1, len(precision) // 200)

    confusion_at = []
    for threshold in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
        predicted = (scores >= threshold).astype(int)
        tp = int(((predicted == 1) & (y_test == 1)).sum())
        fp = int(((predicted == 1) & (y_test == 0)).sum())
        fn = int(((predicted == 0) & (y_test == 1)).sum())
        tn = int(((predicted == 0) & (y_test == 0)).sum())
        confusion_at.append(
            {
                "threshold": threshold,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": round(tp / max(1, tp + fp), 4),
                "recall": round(tp / max(1, tp + fn), 4),
                "accuracy": round((tp + tn) / len(y_test), 6),
                "alerts": tp + fp,
            }
        )

    prevalence = float(y_test.mean())
    report = metrics.classification_report(y_test, scores)
    return {
        "title": "Thresholds, ROC, precision-recall and cost",
        "dataset": "ULB credit card fraud (71,202 held-out transactions)",
        "prevalence": round(prevalence, 6),
        "roc_curve": [
            {"fpr": round(float(fpr[i]), 5), "tpr": round(float(tpr[i]), 5)}
            for i in range(0, len(fpr), step_roc)
        ],
        "pr_curve": [
            {"recall": round(float(recall[i]), 5), "precision": round(float(precision[i]), 5)}
            for i in range(0, len(precision), step_pr)
        ],
        "roc_auc": report["roc_auc"],
        "pr_auc": report["pr_auc"],
        "no_skill_pr": round(prevalence, 6),
        "confusion_at_thresholds": confusion_at,
        "cost_curve": metrics.cost_curve(y_test, scores, cost_fn=500, cost_fp=10, n_points=60),
        "lesson": (
            f"The same model scores ROC-AUC {report['roc_auc']:.4f} and PR-AUC "
            f"{report['pr_auc']:.4f}. Both are correct; they answer different questions. "
            f"ROC's x-axis is false-positive *rate*, whose denominator is all "
            f"{int((y_test == 0).sum()):,} negatives, so thousands of false alarms barely "
            f"move it. Precision's denominator is the model's own alert volume, which is "
            f"what an analyst's queue actually contains. At prevalence "
            f"{prevalence:.4%}, a no-skill model has PR-AUC {prevalence:.5f} but ROC-AUC "
            f"0.5 — so the PR floor moves with the base rate and the ROC floor does not."
        ),
        "quiz": [
            {
                "question": (
                    "A fraud model reports 99.9% accuracy on data with 0.17% fraud. What "
                    "does that tell you?"
                ),
                "options": [
                    "The model is excellent",
                    "Almost nothing — predicting 'never fraud' scores 99.83%",
                    "The model has high recall",
                    "The threshold is well calibrated",
                ],
                "answer": 1,
                "explanation": (
                    "The always-negative baseline scores 99.83%. A 99.9% figure is 0.07 "
                    "points above doing nothing, and is consistent with catching almost no "
                    "fraud."
                ),
            },
            {
                "question": "Raising the decision threshold generally does what?",
                "options": [
                    "Raises precision, lowers recall",
                    "Raises both precision and recall",
                    "Lowers precision, raises recall",
                    "Has no effect on either",
                ],
                "answer": 0,
                "explanation": (
                    "A higher bar means fewer, more confident alerts: a larger share are "
                    "correct (precision up) but more true positives are missed (recall "
                    "down). The threshold selects a point on that trade-off; the cost "
                    "ratio should decide which point."
                ),
            },
        ],
    }


# ======================================================================================
# Module 3 — Gradient descent on a real loss surface
# ======================================================================================
def module_gradient_descent() -> dict:
    """Trace optimiser trajectories across a real two-parameter loss surface.

    Fitting mean temperature from month-of-year on the Melbourne data gives a genuine
    convex surface in two parameters, so the whole thing can be drawn. Four learning rates
    are traced, one of which is deliberately too large and diverges — divergence is the
    single most instructive behaviour and is usually omitted from teaching figures.
    """
    df = pd.read_csv(data.fetch("daily_min_temps"))
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df.Date)
    monthly = df.groupby(df.Date.dt.month)["Temp"].mean()

    # Standardised so the surface is well conditioned and the learning rates are readable.
    x = (monthly.index.to_numpy(dtype=float) - 6.5) / 3.45
    y_raw = monthly.to_numpy(dtype=float)
    y = (y_raw - y_raw.mean()) / y_raw.std()

    def loss(w: float, b: float) -> float:
        return float(np.mean((w * x + b - y) ** 2))

    def gradient(w: float, b: float) -> tuple[float, float]:
        error = w * x + b - y
        return float(2 * np.mean(error * x)), float(2 * np.mean(error))

    # Closed-form optimum, for reference.
    design = np.column_stack([x, np.ones_like(x)])
    w_star, b_star = np.linalg.lstsq(design, y, rcond=None)[0]

    grid_w = np.linspace(-2.5, 2.5, 45)
    grid_b = np.linspace(-2.5, 2.5, 45)
    surface = [
        {"w": round(float(w), 3), "b": round(float(b), 3), "loss": round(loss(w, b), 4)}
        for w in grid_w
        for b in grid_b
    ]

    trajectories = {}
    for lr in (0.01, 0.1, 0.5, 1.05):
        w, b = -2.0, 2.0
        path = []
        diverged = False
        for step in range(60):
            current = loss(w, b)
            if not np.isfinite(current) or current > 1e6:
                diverged = True
                break
            path.append(
                {"step": step, "w": round(w, 4), "b": round(b, 4), "loss": round(current, 5)}
            )
            gw, gb = gradient(w, b)
            w -= lr * gw
            b -= lr * gb
        # Classify by what the trajectory actually did, not by the learning rate value.
        # A run that exits the loop before the 1e6 guard can still be diverging, so the
        # test is whether the loss ended above where it started.
        final_loss = path[-1]["loss"] if path else float("inf")
        initial_loss = path[0]["loss"] if path else float("inf")
        optimal_loss = loss(w_star, b_star)
        diverged = diverged or final_loss > initial_loss
        converged = final_loss < 1.05 * optimal_loss

        trajectories[str(lr)] = {
            "learning_rate": lr,
            "path": path,
            "diverged": bool(diverged),
            "converged": bool(converged),
            "final_loss": final_loss,
            "loss_ratio_to_optimum": round(final_loss / max(1e-12, optimal_loss), 2),
            "steps_to_converge": next(
                (p["step"] for p in path if p["loss"] < 1.05 * optimal_loss), None
            ),
            "behaviour": (
                "diverges — each step overshoots by more than it corrects, so the loss grows"
                if diverged
                else "converges, but slowly — needs far more than 60 steps"
                if not converged
                else "converges efficiently"
            ),
        }

    return {
        "title": "Gradient descent and the learning rate",
        "dataset": "Melbourne monthly mean temperature (real, standardised)",
        "surface": surface,
        "optimum": {"w": round(float(w_star), 4), "b": round(float(b_star), 4),
                    "loss": round(loss(w_star, b_star), 5)},
        "trajectories": trajectories,
        "lesson": (
            "The learning rate is the whole story on a convex surface. At 0.01 the "
            "trajectory crawls; at 0.1–0.5 it converges cleanly; at 1.05 each step "
            "overshoots by more than it corrects and the loss diverges to infinity. The "
            "threshold is not arbitrary — for a quadratic loss, gradient descent diverges "
            "once the learning rate exceeds 2/λ_max, where λ_max is the largest eigenvalue "
            "of the Hessian. That is why the same learning rate can be fine on one problem "
            "and catastrophic on another: it depends on the curvature, not on the number."
        ),
        "quiz": [
            {
                "question": "Why does a learning rate of 1.05 diverge here?",
                "options": [
                    "The loss surface is non-convex",
                    "Each step overshoots by more than it corrects, so the error grows",
                    "There is not enough data",
                    "The gradient is computed incorrectly",
                ],
                "answer": 1,
                "explanation": (
                    "The surface is convex with a single minimum. Divergence happens when "
                    "the step size exceeds 2/λ_max: the update jumps past the minimum to a "
                    "point of higher loss, and the next gradient is larger still."
                ),
            }
        ],
    }


# ======================================================================================
# Module 4 — Backpropagation is the chain rule
# ======================================================================================
def module_backprop() -> dict:
    """Verify analytic gradients against finite differences on a real two-layer network.

    Backpropagation is often taught as though it were a special algorithm. It is the chain
    rule applied in reverse order. The demonstration here is a gradient check: if the
    analytic gradient matches a numerical derivative to ~1e-7, the derivation is right, and
    that check is the single most useful debugging tool when it is not.
    """
    rng = np.random.default_rng(SEED)

    df = data.load_csv("titanic")
    features = df[["Pclass", "Age", "Fare"]].copy()
    features["Age"] = features.Age.fillna(features.Age.median())
    X = ((features - features.mean()) / features.std()).to_numpy()[:64]
    y = df.Survived.to_numpy()[:64].reshape(-1, 1).astype(float)

    n_hidden = 5
    W1 = rng.normal(0, 0.5, (X.shape[1], n_hidden))
    b1 = np.zeros((1, n_hidden))
    W2 = rng.normal(0, 0.5, (n_hidden, 1))
    b2 = np.zeros((1, 1))

    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -50, 50)))

    def forward(W1, b1, W2, b2):
        z1 = X @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        a2 = sigmoid(z2)
        eps = 1e-12
        loss = float(-np.mean(y * np.log(a2 + eps) + (1 - y) * np.log(1 - a2 + eps)))
        return loss, (z1, a1, z2, a2)

    def backward(cache):
        z1, a1, z2, a2 = cache
        m = X.shape[0]
        # dL/dz2 for binary cross-entropy composed with sigmoid simplifies to (a2 - y)/m.
        dz2 = (a2 - y) / m
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        da1 = dz2 @ W2.T
        dz1 = da1 * (1 - np.tanh(z1) ** 2)      # d/dz tanh(z) = 1 - tanh²(z)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
        return dW1, db1, dW2, db2

    base_loss, cache = forward(W1, b1, W2, b2)
    dW1, db1, dW2, db2 = backward(cache)

    # Central finite differences: (f(θ+ε) − f(θ−ε)) / 2ε.
    epsilon = 1e-6
    checks = []
    for name, param, analytic in (("W1", W1, dW1), ("W2", W2, dW2)):
        for _ in range(6):
            i = rng.integers(param.shape[0])
            j = rng.integers(param.shape[1])
            original = param[i, j]

            param[i, j] = original + epsilon
            plus, _ = forward(W1, b1, W2, b2)
            param[i, j] = original - epsilon
            minus, _ = forward(W1, b1, W2, b2)
            param[i, j] = original

            numerical = (plus - minus) / (2 * epsilon)
            analytic_value = float(analytic[i, j])
            denominator = max(1e-12, abs(numerical) + abs(analytic_value))
            checks.append(
                {
                    "parameter": f"{name}[{i},{j}]",
                    "analytic": round(analytic_value, 10),
                    "numerical": round(float(numerical), 10),
                    "relative_error": round(
                        abs(numerical - analytic_value) / denominator, 12
                    ),
                }
            )

    max_error = max(c["relative_error"] for c in checks)

    # Train it so there is a loss curve to show.
    curve = []
    for step in range(400):
        loss_value, cache = forward(W1, b1, W2, b2)
        if step % 10 == 0:
            curve.append({"step": step, "loss": round(loss_value, 5)})
        gW1, gb1, gW2, gb2 = backward(cache)
        W1 -= 0.5 * gW1
        b1 -= 0.5 * gb1
        W2 -= 0.5 * gW2
        b2 -= 0.5 * gb2

    return {
        "title": "Backpropagation is the chain rule",
        "dataset": "Titanic (64 real passengers, 3 standardised features)",
        "architecture": "3 → 5 (tanh) → 1 (sigmoid), binary cross-entropy",
        "gradient_checks": checks,
        "max_relative_error": max_error,
        "check_passed": bool(max_error < 1e-6),
        "training_curve": curve,
        "initial_loss": round(base_loss, 5),
        "final_loss": curve[-1]["loss"],
        "chain_rule_steps": [
            {"step": "∂L/∂a₂", "expression": "−(y/a₂ − (1−y)/(1−a₂))/m",
             "note": "derivative of binary cross-entropy w.r.t. the output activation"},
            {"step": "∂a₂/∂z₂", "expression": "a₂(1 − a₂)",
             "note": "derivative of the sigmoid"},
            {"step": "∂L/∂z₂", "expression": "(a₂ − y)/m",
             "note": "the two above multiply and simplify — this cancellation is why "
                     "sigmoid pairs with cross-entropy rather than with squared error"},
            {"step": "∂L/∂W₂", "expression": "a₁ᵀ · ∂L/∂z₂",
             "note": "chain rule through the linear layer"},
            {"step": "∂L/∂a₁", "expression": "∂L/∂z₂ · W₂ᵀ",
             "note": "propagate the error backwards one layer"},
            {"step": "∂L/∂z₁", "expression": "∂L/∂a₁ ⊙ (1 − tanh²(z₁))",
             "note": "derivative of tanh, applied elementwise"},
            {"step": "∂L/∂W₁", "expression": "Xᵀ · ∂L/∂z₁",
             "note": "chain rule through the first linear layer"},
        ],
        "lesson": (
            f"The analytic gradients match central finite differences to a maximum "
            f"relative error of {max_error:.2e}. That agreement is the proof the "
            f"derivation is correct, and the gradient check is the first thing to run when "
            f"a network will not train. Note the third step: ∂L/∂a₂ and ∂a₂/∂z₂ multiply to "
            f"the clean (a₂ − y)/m. That cancellation is exactly why sigmoid is paired with "
            f"cross-entropy — with squared error the sigmoid derivative survives, and it "
            f"vanishes when the unit saturates, stalling learning."
        ),
        "quiz": [
            {
                "question": (
                    "A gradient check gives relative error 0.3 between analytic and "
                    "numerical gradients. What does that indicate?"
                ),
                "options": [
                    "Normal floating-point noise",
                    "A bug in the backward pass",
                    "The learning rate is too high",
                    "The network needs more layers",
                ],
                "answer": 1,
                "explanation": (
                    "Correct gradients agree to roughly 1e-7 with central differences. An "
                    "error of 0.3 is a derivation or implementation error, not noise."
                ),
            }
        ],
    }


# ======================================================================================
# Module 5 — Bias, variance and model capacity
# ======================================================================================
def module_bias_variance() -> dict:
    """Decompose test error into bias and variance over bootstrap resamples.

    The textbook picture is a clean U: bias falls with capacity, variance rises, total
    error is minimised somewhere in the middle. On real data the curve is noisier and the
    minimum is often a plateau rather than a point. Measuring the decomposition rather than
    drawing it is the difference between illustrating the idea and demonstrating it.
    """
    df = pd.read_csv(data.fetch("daily_min_temps"))
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df.Date)
    df["doy"] = df.Date.dt.dayofyear

    rng = np.random.default_rng(SEED)
    sample = df.sample(n=600, random_state=SEED)
    x_all = sample.doy.to_numpy(dtype=float)
    y_all = sample.Temp.to_numpy(dtype=float)

    order = rng.permutation(len(x_all))
    split_at = int(len(order) * 0.7)
    train_i, test_i = order[:split_at], order[split_at:]
    x_train, y_train = x_all[train_i], y_all[train_i]
    x_test, y_test = x_all[test_i], y_all[test_i]

    n_bootstrap = 40
    degrees = list(range(1, 16))
    rows = []
    for degree in degrees:
        predictions = np.zeros((n_bootstrap, len(x_test)))
        train_errors = []
        for b in range(n_bootstrap):
            boot = rng.integers(0, len(x_train), len(x_train))
            coefficients = np.polyfit(x_train[boot], y_train[boot], degree)
            predictions[b] = np.polyval(coefficients, x_test)
            train_errors.append(
                float(np.mean((np.polyval(coefficients, x_train[boot]) - y_train[boot]) ** 2))
            )

        mean_prediction = predictions.mean(axis=0)
        bias_squared = float(np.mean((mean_prediction - y_test) ** 2))
        variance = float(np.mean(predictions.var(axis=0)))
        total = float(np.mean((predictions - y_test[None, :]) ** 2))
        rows.append(
            {
                "degree": degree,
                "bias_squared": round(bias_squared, 4),
                "variance": round(variance, 4),
                "total_test_mse": round(total, 4),
                "train_mse": round(float(np.mean(train_errors)), 4),
                "generalisation_gap": round(total - float(np.mean(train_errors)), 4),
            }
        )

    best = min(rows, key=lambda r: r["total_test_mse"])
    highest = rows[-1]
    first = rows[0]

    # State what the curve did rather than what the textbook says it should do.
    train_monotone = all(
        rows[i]["train_mse"] >= rows[i + 1]["train_mse"] - 0.05 for i in range(len(rows) - 1)
    )
    variance_share = highest["variance"] / max(1e-12, highest["total_test_mse"])
    bias_dominates = variance_share < 0.2
    return {
        "title": "Bias, variance and model capacity",
        "dataset": "Melbourne daily temperature (600 real observations, day-of-year → temp)",
        "n_bootstrap": n_bootstrap,
        "curve": rows,
        "optimal_degree": best["degree"],
        "train_error_monotone": bool(train_monotone),
        "variance_share_at_max_degree": round(float(variance_share), 4),
        "bias_dominates": bool(bias_dominates),
        "lesson": (
            f"Test error is minimised at degree {best['degree']} "
            f"(MSE {best['total_test_mse']}), rising to {highest['total_test_mse']} by "
            f"degree {highest['degree']}. Variance grows monotonically as expected — from "
            f"{first['variance']} at degree {first['degree']} to {highest['variance']} at "
            f"degree {highest['degree']}. "
            + (
                f"But this real curve is NOT the textbook U. Variance accounts for only "
                f"{variance_share:.1%} of test error even at the highest degree; bias² "
                f"dominates throughout. Day-of-year simply does not determine daily "
                f"temperature — the irreducible noise is large, so no amount of capacity "
                f"helps and the curve flattens rather than turning sharply upward. A "
                f"simulated example would show the clean U and hide this, which is the "
                f"more common real situation."
                if bias_dominates
                else "Variance overtakes the bias reduction after the optimum, producing "
                "the classic U-shape."
            )
            + (
                " Training error also falls monotonically with capacity, which is why it "
                "carries no information about generalisation."
                if train_monotone
                else " Note that training error here does NOT fall monotonically: each "
                "degree is averaged over bootstrap resamples, so resampling noise exceeds "
                "the small gain from extra capacity once bias has plateaued. The textbook "
                "monotone training curve assumes a single fixed training set."
            )
        ),
        "quiz": [
            {
                "question": (
                    "Training error keeps decreasing as polynomial degree rises, but test "
                    "error starts increasing after degree 6. What is happening?"
                ),
                "options": [
                    "The optimiser is failing to converge",
                    "Variance is growing faster than bias is shrinking — overfitting",
                    "The test set is too small",
                    "The data is non-stationary",
                ],
                "answer": 1,
                "explanation": (
                    "Extra capacity lets the model track noise specific to its training "
                    "sample. Bias² keeps falling but variance rises faster, so total error "
                    "turns upward. Training error cannot reveal this because it falls "
                    "monotonically with capacity."
                ),
            }
        ],
    }


# ======================================================================================
# Module 6 — Sampling distributions and the CLT
# ======================================================================================
def module_sampling() -> dict:
    """Demonstrate the central limit theorem on a genuinely skewed real distribution.

    Transaction amounts are heavily right-skewed with a long tail, so the convergence of
    the sampling distribution of the mean towards normality is something to be observed
    rather than assumed. The skewness of the sample mean at each n is reported, which makes
    the rate of convergence visible instead of merely asserted.
    """
    from scipy import stats

    df = data.load_csv("credit_card_fraud")
    amounts = df.Amount.to_numpy()
    rng = np.random.default_rng(SEED)

    population = {
        "n": int(len(amounts)),
        "mean": round(float(amounts.mean()), 4),
        "std": round(float(amounts.std()), 4),
        "skewness": round(float(stats.skew(amounts)), 4),
        "kurtosis": round(float(stats.kurtosis(amounts)), 4),
        "median": round(float(np.median(amounts)), 4),
        "histogram": artifacts.histogram(amounts[amounts < 500], bins=40),
    }

    distributions = []
    for n in (1, 2, 5, 10, 30, 100, 500):
        means = np.array([rng.choice(amounts, size=n, replace=False).mean()
                          for _ in range(1500)])
        # Standard error shrinks as sigma/sqrt(n); comparing predicted to observed makes
        # the 1/sqrt(n) law checkable rather than a claim.
        distributions.append(
            {
                "sample_size": n,
                "histogram": artifacts.histogram(means, bins=35),
                "observed_se": round(float(means.std()), 4),
                "predicted_se": round(float(amounts.std() / np.sqrt(n)), 4),
                "skewness": round(float(stats.skew(means)), 4),
                "shapiro_p": round(
                    float(stats.shapiro(means[:500]).pvalue), 6
                ),
            }
        )

    return {
        "title": "Sampling distributions and the central limit theorem",
        "dataset": "ULB transaction amounts (284,807 real values, heavily right-skewed)",
        "population": population,
        "sampling_distributions": distributions,
        "se_law_check": [
            {
                "n": d["sample_size"],
                "observed_over_predicted": round(
                    d["observed_se"] / max(1e-12, d["predicted_se"]), 3
                ),
            }
            for d in distributions
        ],
        "lesson": (
            f"The population of transaction amounts has skewness "
            f"{population['skewness']:.2f} — nothing like a normal distribution. The "
            f"sampling distribution of the mean nevertheless approaches normality as n "
            f"grows: skewness falls from {distributions[3]['skewness']:.2f} at n=10 to "
            f"{distributions[-1]['skewness']:.2f} at n=500, and the observed standard "
            f"error tracks σ/√n to within a few percent from n=10 onward. That √n is why "
            f"quadrupling the sample only halves the error. "
            f"Two honest caveats visible in the table. First, the n=1 and n=2 rows are "
            f"erratic — skewness {distributions[0]['skewness']:.2f} then "
            f"{distributions[1]['skewness']:.2f} — because 1,500 draws from a "
            f"distribution this heavy-tailed under-sample the extreme values, so those "
            f"estimates are dominated by whether a few outliers happened to be drawn. "
            f"Second, convergence is far slower than the familiar rule of thumb suggests: "
            f"at n=30 the sample mean is still clearly skewed "
            f"({distributions[4]['skewness']:.2f}). 'n=30 is enough' is calibrated on "
            f"mildly non-normal populations, not on one with skewness 17."
        ),
        "quiz": [
            {
                "question": (
                    "The population is strongly right-skewed. At what sample size does the "
                    "sampling distribution of the mean become approximately normal here?"
                ),
                "options": [
                    "n = 30, always — it is a fixed rule",
                    "It depends on the population's skewness; here it takes well over 30",
                    "Never, because the population is not normal",
                    "n = 2 is sufficient",
                ],
                "answer": 1,
                "explanation": (
                    "'n = 30' is a rule of thumb calibrated on mildly non-normal "
                    "populations. The more skewed the population, the larger n must be. "
                    "Here the sample mean is still measurably skewed at n = 30."
                ),
            }
        ],
    }


# ======================================================================================
# Orchestration
# ======================================================================================
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    crisp = CrispDm(
        project=PROJECT,
        business_question=(
            "Can the core concepts of applied statistics be taught from real data, "
            "including the cases where real data refuses to behave like the textbook?"
        ),
    )
    crisp.record(
        Phase(
            name="business_understanding",
            summary=(
                "Teaching material almost always uses generated data because generated "
                "data satisfies its assumptions. Students then meet real data where the "
                "assumptions fail and have no framework for it. Every module here is "
                "computed from a real dataset used elsewhere in this portfolio, and where "
                "an assumption breaks, the breakage is the lesson."
            ),
            decisions=[
                Decision(
                    question="Generated illustrations or real data?",
                    choice="Real data throughout, including its inconveniences.",
                    rationale=(
                        "A clean simulated bias-variance U-curve teaches the shape but not "
                        "the judgement. A real curve with a plateau, and a Naive Bayes "
                        "independence assumption that is measurably violated, teach what "
                        "the concepts look like when they are actually used."
                    ),
                    alternatives_rejected=[
                        "Synthetic Gaussian illustrations — clean, and they omit exactly "
                        "the difficulty the student needs to see.",
                    ],
                ),
            ],
            evidence={"n_modules": 6},
        )
    )

    with artifacts.run(PROJECT, seed=SEED) as ctx:
        modules = {}
        for name, fn in (
            ("bayes", module_bayes),
            ("threshold", module_threshold),
            ("gradient_descent", module_gradient_descent),
            ("backprop", module_backprop),
            ("bias_variance", module_bias_variance),
            ("sampling", module_sampling),
        ):
            print(f"\n[{name}] computing …")
            modules[name] = fn()
            print(f"      ✓ {modules[name]['title']}")

        total_quiz = sum(len(m.get("quiz", [])) for m in modules.values())

        crisp.record(
            Phase(
                name="data_understanding",
                summary=(
                    "Five real datasets across six modules: Titanic for Bayes and "
                    "backpropagation, ULB fraud for thresholds and sampling, Melbourne "
                    "temperature for optimisation and bias-variance."
                ),
                evidence={name: m["dataset"] for name, m in modules.items()},
            )
        )
        crisp.record(
            Phase(
                name="data_preparation",
                summary=(
                    "Each module prepares only what its concept requires: median "
                    "imputation and quantile banding for Titanic, chronological splitting "
                    "for fraud, standardisation for the loss surface so learning rates are "
                    "interpretable."
                ),
                evidence={"modules": list(modules)},
            )
        )
        crisp.record(
            Phase(
                name="modeling",
                summary=(
                    "Each module fits the smallest model that demonstrates its concept: "
                    "Naive Bayes against logistic regression, a class-weighted logistic "
                    "fraud scorer, closed-form and gradient-descent linear fits, a "
                    "hand-differentiated two-layer network, and bootstrapped polynomials."
                ),
                evidence={
                    "gradient_check_max_error": modules["backprop"]["max_relative_error"],
                    "gradient_check_passed": modules["backprop"]["check_passed"],
                    "optimal_polynomial_degree": modules["bias_variance"]["optimal_degree"],
                },
            )
        )
        crisp.record(
            Phase(
                name="evaluation",
                summary=(
                    f"Six modules, {total_quiz} assessment questions, every figure computed "
                    f"from real data. The backpropagation module verifies its own "
                    f"derivation to a maximum relative error of "
                    f"{modules['backprop']['max_relative_error']:.2e}."
                ),
                evidence={
                    "n_modules": len(modules),
                    "n_quiz_questions": total_quiz,
                    "self_verifying": ["backprop gradient check"],
                },
                risks=[
                    "Every module uses one dataset per concept. A concept demonstrated on "
                    "one dataset is illustrated, not established.",
                    "The quiz has a single correct answer per question, which suits "
                    "concept checks but cannot assess the judgement the modules argue is "
                    "the real skill.",
                ],
            )
        )
        crisp.record(
            Phase(
                name="deployment",
                summary=(
                    "Each module exports the arrays behind a live widget — loss surface "
                    "grids, ROC and PR traces, bootstrap curves, sampling histograms — so "
                    "the published page recomputes nothing and stays interactive offline."
                ),
                evidence={"modules": list(modules)},
            )
        )

        for name, payload in modules.items():
            artifacts.write(OUT / f"{name}.json", payload, context=ctx)
        artifacts.write(
            OUT / "index.json",
            {
                "modules": [
                    {"key": k, "title": v["title"], "dataset": v["dataset"],
                     "n_quiz": len(v.get("quiz", []))}
                    for k, v in modules.items()
                ],
                "n_quiz_questions": total_quiz,
            },
            context=ctx,
        )
        artifacts.write(OUT / "crispdm.json", crisp.to_dict(), context=ctx)
        artifacts.write(
            OUT / "provenance.json",
            {
                "datasets": [
                    data.provenance_record(k)
                    for k in ("titanic", "credit_card_fraud", "daily_min_temps")
                ]
            },
            context=ctx,
        )

    print(f"\n✓ {PROJECT} complete — {len(modules)} modules, {total_quiz} quiz questions")


if __name__ == "__main__":
    main()
