---
name: teaching-from-real-data
description: Build statistical teaching material where assumption violations are the content
---

# Build statistical teaching material where assumption violations are the content

*Derived from [`projects/08_crispdm_academy`](../../), which applied this procedure to: Titanic Passenger Manifest, Credit Card Fraud Detection (ULB, Sep 2013), Daily Minimum Temperatures, Melbourne (1981-1990).*

This is a **transferable procedure**, not instructions for rerunning the script in this
repository. It states what to do on a new dataset, where the judgement calls are, and which
failure modes this project actually hit.

## When to use this

Building teaching material, or checking whether a concept you learned from a textbook figure
behaves that way in practice.

## Procedure

### 1. Use real data, and let it misbehave

Generated data satisfies its assumptions. That is the point of generating it, and it is why
students who learn from it are unprepared for the first real dataset.

### 2. Detect assumption violations in code, not in prose

Do not write "the bias-variance curve forms a U". Compute whether it does:

```python
variance_share = highest["variance"] / highest["total_test_mse"]
bias_dominates = variance_share < 0.2

train_monotone = all(rows[i]["train_mse"] >= rows[i + 1]["train_mse"] - tol
                     for i in range(len(rows) - 1))
```

Then branch the explanation on the measured result. This is the only way the lesson stays
correct when the data changes.

### 3. Quantify the violation rather than mentioning it

Naive Bayes assumes conditional independence. Measure Cramér's V between feature pairs *within
each class* and report how many pairs violate it, then show the consequence: ranking usually
survives, calibration usually does not.

### 4. Include the failure cases teaching figures omit

- A learning rate that **diverges**, not just one that converges slowly
- A sampling distribution still visibly skewed at n = 30
- A bias-variance curve with no visible U

These are more memorable and more useful than the clean versions.

### 5. Make gradient checking a first-class lesson

```python
numerical = (f(theta + eps) - f(theta - eps)) / (2 * eps)
relative_error = abs(numerical - analytic) / (abs(numerical) + abs(analytic))
assert relative_error < 1e-6
```

Correct gradients agree to ~1e-7. An error near 0.3 is a bug, not noise. This is the single
most useful debugging tool for a network that will not train.

### 6. Show the same model under two metrics

The clearest way to teach why a metric matters is to show one model scoring well on one and
badly on another. ROC-AUC 0.94 with PR-AUC 0.04 on the same predictions makes the point in a
way no explanation does.

### 7. Flag your own artefacts

If a low-n estimate is erratic because of sampling noise, say so rather than smoothing the
curve. The artefact is itself a lesson about estimation.

## Decision points requiring judgement

**One dataset per concept illustrates; it does not establish.** Say which you are doing.

**Single-answer quizzes check recall, not judgement** — which is the actual skill the material
argues for.

---

## Reproduce the reference implementation

```bash
export PYTHONPATH=lib
python3 projects/08_crispdm_academy/pipeline/build.py
python3 tools/audit.py
```

Results, with their run provenance, land in
[`projects/08_crispdm_academy/artifacts/`](../../artifacts/).
