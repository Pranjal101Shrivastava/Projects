# 06 · AutoML, and the Cost of One Leaking Column

An identical model tournament run **twice** — with and without a column that cannot exist at
scoring time — to price a documented target leak.

| Dataset | Kind | Size | Origin | Licence |
|---|:---:|---|---|---|
| [Bank Marketing — Portuguese Term Deposit Campaigns (2008–2010)](https://raw.githubusercontent.com/selva86/datasets/master/bank-full.csv) | 🟢 **REAL** | 41,188 contacts, 4,640 subscriptions (11.27%) | Direct marketing call records from a Portuguese retail bank, collected May 2008 – November 2010 and published by Moro, Cortez & Rita (2014). Each row is one real client contacted by phone; the target is whether they subscribed to a term deposit. Includes contemporaneous macroeconomic indicators (euribor3m, employment variation rate, consumer confidence). | Creative Commons Attribution 4.0 (UCI Machine Learning Repository). |

## The trap

UCI's Bank Marketing dataset ships with a `duration` column recording how long the sales call
lasted. UCI's own documentation states it *"should be discarded if the intention is to have a
realistic predictive model"*.

It appears in most published notebooks on this dataset anyway, because including it makes
every model look dramatically better.

## The measurement

Same four model families. Same 12 randomised hyperparameter draws each. Same 4-fold
stratified CV. Same stacked ensemble. The **only** difference is one column.

| Model | Leak-free | With leak | Inflation |
|---|---:|---:|---:|
| `stacked_ensemble` | 0.4961 | 0.6939 | **+39.9%** |
| `random_forest` | 0.4960 | 0.6917 | **+39.4%** |
| `hist_gradient_boosting` | 0.4925 | 0.6875 | **+39.6%** |
| `logistic` | 0.4694 | 0.6235 | **+32.8%** |
| `gaussian_nb` | 0.3607 | 0.4003 | **+11.0%** |

**Best PR-AUC moves from 0.4961 to 0.6939 —
39.9% relative inflation from a single column.**

Including one column that cannot exist at scoring time raises the best PR-AUC from 0.4961 to 0.6939 — a 39.9% relative gain that would evaporate entirely in production. Nothing in the cross-validation, the confusion matrix or the calibration curve flags it. Only reasoning about when each value becomes known does.

## Why it is a leak, not a strong feature

| Evidence | Value |
|---|---:|
| `duration` alone, ROC-AUC | **0.8184** |
| Correlation with target | 0.405 |
| Mean call length — subscribed | 553s |
| Mean call length — declined | 221s |
| Zero-duration calls | 4 |
| …of which subscribed | **0** |

A single column achieving ROC-AUC 0.818 on its own is a red flag, not a feature. Subscribers average 553s on the call against 221s for those who decline — but that is a consequence of the outcome, not a predictor of it. The value cannot be known before the call is placed, which is exactly when a prioritisation model must score.

The zero-duration row is the clean confirmation: all 4
zero-second calls are non-subscriptions, exactly as the causal argument predicts. You cannot
subscribe to a term deposit during a call that never happened.

## How to catch a leak

**Test 1 — implausible univariate power.** Does any single column predict the target
implausibly well alone? Necessary, but not sufficient: a genuinely strong feature can also
score highly.

**Test 2 — causal timing.** *Would this value exist at the moment the prediction must be
made?* This is the decisive one and it requires no statistics. A call-prioritisation model
scores **before dialling**, when call length does not yet exist.

No cross-validation scheme, confusion matrix or calibration curve detects this. Only
reasoning about the order of events does.

## Leaderboards

### Leak-free — the deployable result

| Model | Test PR-AUC | CV PR-AUC | ROC-AUC | Brier | Search |
|---|---:|---:|---:|---:|---:|
| `stacked_ensemble` | 0.4961 | — | 0.8173 | 0.0754 | 36s |
| `random_forest` | 0.4960 | 0.4664 ± 0.011 | 0.8177 | 0.0744 | 245s |
| `hist_gradient_boosting` | 0.4925 | 0.4665 ± 0.011 | 0.8152 | 0.0747 | 29s |
| `logistic` | 0.4694 | 0.4450 ± 0.018 | 0.8053 | 0.0768 | 10s |
| `gaussian_nb` | 0.3607 | 0.3486 ± 0.007 | 0.7797 | 0.1441 | 5s |

### With leak — demonstration only

| Model | Test PR-AUC | CV PR-AUC | ROC-AUC | Brier | Search |
|---|---:|---:|---:|---:|---:|
| `stacked_ensemble` | 0.6939 | — | 0.9538 | 0.0549 | 36s |
| `random_forest` | 0.6917 | 0.6579 ± 0.008 | 0.9521 | 0.0537 | 300s |
| `hist_gradient_boosting` | 0.6875 | 0.6636 ± 0.010 | 0.9535 | 0.0533 | 51s |
| `logistic` | 0.6235 | 0.5905 ± 0.018 | 0.9423 | 0.0606 | 11s |
| `gaussian_nb` | 0.4003 | 0.3851 ± 0.005 | 0.8428 | 0.1346 | 5s |

## Searching families, not just hyperparameters

A search over boosting depths is a hyperparameter tuner, not AutoML. Four families with
genuinely different inductive biases were searched:

- **`stacked_ensemble`** — logistic meta-learner over out-of-fold predictions of random_forest, hist_gradient_boosting, logistic
- **`random_forest`** — bagged axis-aligned partitions, variance reduction
- **`hist_gradient_boosting`** — sequential residual fitting, bias reduction
- **`logistic`** — linear decision boundary in the encoded space
- **`gaussian_nb`** — conditional independence assumption — deliberately naive reference

Including a deliberately naive Bayes reference shows how much of the final score came from
model capacity: it reaches
0.361
against the winner's 0.496.

## Stacking without leaking one level up

The meta-learner is trained on **cross-validated out-of-fold predictions only**. Fitting it on
in-fold predictions would let it observe base-model outputs for rows those models had
effectively memorised — leakage one level up, and a common way stacked ensembles get silently
overfitted.

## Screens

![06 automl](../../docs/screenshots/06_automl.png)


## CRISP-DM record

> **Business question.** Which model should the bank use to prioritise term-deposit calls — and how much of any reported performance is real rather than leaked?

### Business Understanding

The operational goal is ranking prospects so a fixed calling budget reaches the most likely subscribers. The methodological goal, which matters more here, is establishing what the model's performance actually is once a documented target leak is removed.

**Should 'duration' be used as a feature?**

- **Chose:** No — and the tournament is run both ways to price the mistake.
- **Why:** Call duration is unknown until the call is over, by which point the outcome is effectively determined: a zero-second call is a rejection. Using it produces a model that cannot be deployed, because at scoring time — before dialling — the value does not exist. UCI's documentation says so explicitly, and most published work on this dataset uses it anyway.
- *Rejected:* Include duration for 'benchmark purposes' — the resulting number is not a forecast of anything obtainable in production.

### Data Understanding

41,188 contacts, 11.27% positive. The 'duration' column alone achieves ROC-AUC 0.818 against the target — the signature of a leak rather than of a strong feature.

**How is a leaking feature identified in the first place?**

- **Chose:** Univariate predictive power plus a causal-timing check.
- **Why:** Two questions catch most leaks. First, does any single column predict the target implausibly well on its own? Second, and decisively: would this value exist at the moment the prediction has to be made? 'duration' fails the second outright — it is generated by the very event being predicted.

### Data Preparation

Median imputation and standardisation for numerics, one-hot encoding with rare-level folding for categoricals — all inside a ColumnTransformer within the pipeline, so each CV fold refits the transforms on its own training rows. The only difference between the two runs is the presence of one column.

### Modeling

Four model families with genuinely different inductive biases, 12 randomised hyperparameter draws each under 4-fold stratified CV, plus a stacked ensemble over the top three trained on out-of-fold predictions.

**Why search families rather than tune one harder?**

- **Chose:** Four families spanning different inductive biases.
- **Why:** A search over boosting depths is a hyperparameter tuner, not AutoML. Including a linear model and a deliberately naive Bayes reference shows how much of the final score comes from model capacity and how much was available from any reasonable baseline.

**How is the stacking meta-learner trained?**

- **Chose:** On cross-validated out-of-fold predictions only.
- **Why:** Fitting the meta-learner on in-fold predictions lets it see base-model outputs for rows those models effectively memorised. The ensemble then learns to trust an accuracy that does not exist out of sample — leakage one level up, and a common way stacking silently overfits.

### Evaluation

Including one column that cannot exist at scoring time raises the best PR-AUC from 0.4961 to 0.6939 — a 39.9% relative gain that would evaporate entirely in production. Nothing in the cross-validation, the confusion matrix or the calibration curve flags it. Only reasoning about when each value becomes known does.

**Limitations**

- The leak-free leaderboard is the deployable one. Any comparison against published results on this dataset that include 'duration' is not like-for-like.
- Macroeconomic columns are retained in the leak-free run. They are knowable at scoring time, but they make the model partly a function of the economic cycle, so performance will drift as conditions change.
- Randomised search with 12 draws per family is a light budget. The ranking between close families should be read as provisional.

### Deployment

Both leaderboards ship so the comparison is inspectable, with the leak-free run marked as the deployable result and the leaked run labelled as a demonstration.


## Run it

```bash
git clone https://github.com/Pranjal101Shrivastava/Projects
cd Projects
pip install -r requirements.txt
export PYTHONPATH=lib

python3 projects/06_automl_tournament/pipeline/build.py   # rebuilds every artifact below
python3 tools/audit.py                          # static leakage audit
```

Datasets download on first run into `.data/` and are verified against their recorded
SHA-256 on every run thereafter. The pipeline is seeded, so a rerun at the same commit
reproduces the same numbers.

---

**Live:** [https://pranjal101shrivastava.github.io/Projects/#/p/automl](https://pranjal101shrivastava.github.io/Projects/#/p/automl) ·
**Method:** [`pipeline/build.py`](./pipeline/build.py) ·
**Audit:** [`audit.md`](./audit.md) ·
**Artifacts:** [`artifacts/`](./artifacts/)

*Generated at commit `fc77533` · seed 42 · 729.89s · Python 3.11.15 · numpy 2.4.6 · pandas 3.0.5 · sklearn 1.9.1 · lightgbm 4.7.0 · torch 2.14.0+cu130*
