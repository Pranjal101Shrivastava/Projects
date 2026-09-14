---
name: target-leakage-detection
description: Find the feature that cannot exist at scoring time, before it reaches a leaderboard
---

# Find the feature that cannot exist at scoring time, before it reaches a leaderboard

*Derived from [`projects/06_automl_tournament`](../../), which applied this procedure to: Bank Marketing — Portuguese Term Deposit Campaigns (2008–2010).*

This is a **transferable procedure**, not instructions for rerunning the script in this
repository. It states what to do on a new dataset, where the judgement calls are, and which
failure modes this project actually hit.

## When to use this

Before trusting any leaderboard. This is a review procedure, not a modelling one.

## Procedure

### 1. Ask the causal-timing question of every column

> **Would this value exist at the moment the prediction must be made?**

This requires no statistics and catches leaks nothing else will. For each feature, identify
when it gets populated relative to the prediction event.

Classic offenders:

| Feature | Why it leaks |
|---|---|
| Call / session duration | Produced by the interaction being predicted |
| Final status fields | Written after the outcome |
| Aggregates over "all time" | Include the future |
| Row identifiers correlated with time | Encode ordering |
| Anything from a downstream system | Populated after the decision |

### 2. Screen univariate predictive power

```python
for column in features:
    print(column, roc_auc_score(y, df[column]))
```

Anomalously high single-feature scores warrant investigation. **Necessary but not sufficient**
— genuinely strong features exist, and leaks can be subtle.

### 3. Look for impossible cells

The decisive evidence is usually a degenerate case. If duration = 0 always means "no", the
feature is downstream of the outcome. Cross-tabulate extremes against the target.

### 4. Quantify it: run the tournament twice

Identical protocol, identical folds, identical seeds. One difference: the suspect column.

The gap is the price of the leak, and it is the only way to make the argument concretely to
someone who wants to keep the feature.

### 5. Understand what will NOT catch it

- Cross-validation — clean, because the model is correctly modelling the data it was given
- Generalisation gap — healthy
- Calibration curves — fine
- Confusion matrices — normal
- Learning curves — unremarkable

There is no statistical test. A leaking feature and a strong feature have the same
distributional signature.

### 6. When building the tournament itself

**Search families, not just hyperparameters.** A sweep over boosting depths is a tuner.
Include a linear model and a deliberately weak reference so you can see how much of the score
came from capacity.

**Preprocessing inside the pipeline**, so each fold refits it.

**Stack on out-of-fold predictions only.** Fitting the meta-learner on in-fold predictions
lets it see base-model outputs for rows those models memorised — leakage one level up.

## Decision points requiring judgement

Some leaks are legitimate features in a different framing. Prior-campaign outcome is knowable
before a call and is fine; call duration is not. The question is always *when*, never *how
predictive*.

---

## Reproduce the reference implementation

```bash
export PYTHONPATH=lib
python3 projects/06_automl_tournament/pipeline/build.py
python3 tools/audit.py
```

Results, with their run provenance, land in
[`projects/06_automl_tournament/artifacts/`](../../artifacts/).
