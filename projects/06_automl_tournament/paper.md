# Pricing a Target Leak: An Identical Tournament Run Twice

## 1. Motivation

Target leakage is widely discussed and rarely quantified. This study measures it directly, by
executing the same model selection procedure twice under identical conditions, differing only
in whether one documented leaking column is available.

## 2. The leaking feature

The Bank Marketing dataset records `duration`, the length in seconds of the marketing call.
The UCI documentation states the attribute "highly affects the output target (e.g. if
duration=0 then y='no')" and "should be discarded if the intention is to have a realistic
predictive model".

Its causal status is unambiguous: the value is produced *by* the call whose outcome is being
predicted. At the moment a prioritisation model must score — before dialling — it does not
exist.

### 2.1 Diagnostic evidence

| Statistic | Value |
|---|---:|
| Univariate ROC-AUC | 0.8184 |
| Point-biserial correlation | 0.4053 |
| Mean duration, subscribed | 553.2s |
| Mean duration, declined | 220.8s |
| Zero-duration records | 4 |
| Zero-duration subscriptions | 0 |

A single feature attaining ROC-AUC 0.818 is a diagnostic signal,
though not conclusive on its own. The zero-duration cell is conclusive: a call of zero seconds
cannot produce a subscription, and none does.

## 3. Experimental design

Both runs used: four model families (regularised linear, bagged trees, gradient boosting, and
a conditional-independence reference), 12 randomised hyperparameter draws per family, 4-fold stratified CV, scored by average precision. Preprocessing is inside the pipeline so it is refitted per fold., followed by a stacking
ensemble over the top three base models with a logistic meta-learner trained on
cross-validated out-of-fold predictions.

Preprocessing — median imputation, standardisation, one-hot encoding with rare-level folding
— was placed inside the estimator pipeline so that every transform is refitted on each fold's
training rows. Fitting transforms outside the pipeline would contaminate the comparison with a
*second* leak.

## 4. Results

| Model | Leak-free PR-AUC | Leaked PR-AUC | Inflation |
|---|---:|---:|---:|
| `stacked_ensemble` | 0.4961 | 0.6939 | **+39.9%** |
| `random_forest` | 0.4960 | 0.6917 | **+39.4%** |
| `hist_gradient_boosting` | 0.4925 | 0.6875 | **+39.6%** |
| `logistic` | 0.4694 | 0.6235 | **+32.8%** |
| `gaussian_nb` | 0.3607 | 0.4003 | **+11.0%** |

Including one column that cannot exist at scoring time raises the best PR-AUC from 0.4961 to 0.6939 — a 39.9% relative gain that would evaporate entirely in production. Nothing in the cross-validation, the confusion matrix or the calibration curve flags it. Only reasoning about when each value becomes known does.

Two features of this table are worth attention.

**Inflation is universal but not uniform.** Every family gains, but the naive-Bayes reference
gains least. Higher-capacity models extract more from the leak, so leakage widens the apparent
gap between simple and complex models — which is precisely the comparison a practitioner uses
to justify complexity.

**Cross-validation offers no warning.** CV scores in the leaked run are internally consistent
with its test scores. The generalisation gap looks healthy. Every diagnostic a practitioner
would normally consult reports that the model is sound.

## 5. Detection

Two tests, applied in order.

**Univariate power.** Screen each feature's standalone predictive power. Anomalously high
values warrant investigation. Necessary but insufficient — legitimately strong features exist.

**Causal timing.** For each feature, ask whether its value exists at the moment of prediction.
This test is decisive and requires no statistics. It is also the only one that catches this
case, because no distributional property of `duration` marks it as illegitimate; only its
position in the causal order does.

## 6. Limitations

- The leak-free leaderboard is the deployable one. Any comparison against published results on this dataset that include 'duration' is not like-for-like.
- Macroeconomic columns are retained in the leak-free run. They are knowable at scoring time, but they make the model partly a function of the economic cycle, so performance will drift as conditions change.
- Randomised search with 12 draws per family is a light budget. The ranking between close families should be read as provisional.

*Generated at commit `fc77533` · seed 42 · 729.89s · Python 3.11.15 · numpy 2.4.6 · pandas 3.0.5 · sklearn 1.9.1 · lightgbm 4.7.0 · torch 2.14.0+cu130*
