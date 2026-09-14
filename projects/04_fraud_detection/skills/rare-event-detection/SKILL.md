---
name: rare-event-detection
description: Build and evaluate a detector when the positive class is under 1%
---

# Build and evaluate a detector when the positive class is under 1%

*Derived from [`projects/04_fraud_detection`](../../), which applied this procedure to: Credit Card Fraud Detection (ULB, Sep 2013).*

This is a **transferable procedure**, not instructions for rerunning the script in this
repository. It states what to do on a new dataset, where the judgement calls are, and which
failure modes this project actually hit.

## When to use this

Rare-event detection: fraud, defects, intrusions, adverse events. Anything with a positive
rate below a few percent.

## Procedure

### 1. Compute the null-model accuracy first, and keep it visible

```python
prevalence = y.mean()
print(f"Always-negative accuracy: {1 - prevalence:.4%}")
```

Report it beside every accuracy figure you ever quote. This makes the metric impossible to
misuse.

### 2. Lead with PR-AUC, and show ROC-AUC beside it

ROC's false-positive rate divides by every negative in the dataset, so thousands of false
alarms barely move it. Precision divides by the model's own alert volume — what the operator's
queue contains.

Expect them to diverge sharply. That divergence is worth showing explicitly.

### 3. Split by time if events are bursty

Fraud, intrusions and defects cluster. A random split scatters one episode across both
partitions, so the model is scored on records whose siblings it trained on. Every metric rises
and none of them warns you.

### 4. Do not synthesise minority samples

SMOTE interpolates between minority neighbours. In high-dimensional space with few positives
the neighbours are far apart, so synthetic points land where no real positive has been
observed. The model learns a boundary around fabricated data.

Use class weighting, which rebalances without inventing observations.

### 5. Ablate the reweighting — do not follow the advice

The standard recommendation (`scale_pos_weight = n_neg/n_pos`) is derived from linear models
and does not transfer to high-capacity ensembles. With few positives, an extreme weight makes
every split chase the same handful of rows.

```python
for variant in [{}, {"scale_pos_weight": 10},
                {"scale_pos_weight": ratio}, {"is_unbalance": True}]:
    ...  # measure, do not assume
```

**Publish all variants**, not the winner.

### 6. Choose the threshold from costs, not from 0.5

Trace expected cost across thresholds under your FN:FP ratio. State the ratio as an
assumption and publish the whole curve so a reader with different costs can pick their own
point.

### 7. Report the alert-budget table

Operationally, the question is not "what threshold" but "how many alerts can we review". The
table showing recall and precision at 50, 100, 250, 500 and 1,000 alerts is more useful to a
decision-maker than any single metric.

## Decision points requiring judgement

**The no-skill floor is the prevalence of the set being scored**, not of the full dataset. A
chronological split does not preserve the base rate.

**Threshold selected on the test set is mildly optimistic.** Use a separate validation period
in production.

---

## Reproduce the reference implementation

```bash
export PYTHONPATH=lib
python3 projects/04_fraud_detection/pipeline/build.py
python3 tools/audit.py
```

Results, with their run provenance, land in
[`projects/04_fraud_detection/artifacts/`](../../artifacts/).
