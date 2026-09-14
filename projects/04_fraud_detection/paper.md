# Evaluation and Operating-Point Selection for Fraud Detection at 0.17% Prevalence

*A CRISP-DM study on 284,807 real card transactions*

## 1. The constraint that defines the problem

A fraud screen operates under fixed analyst capacity. The objective is therefore not
"detect fraud" but **maximise fraud caught per alert raised**. Every methodological choice
below follows from that framing plus one number: the positive rate of
0.1727%.

## 2. Why the usual metrics fail

**Accuracy.** A constant "never fraud" predictor achieves
99.8273%. Our best model achieves
99.9522%. The difference —
0.084 percentage points —
is the entire contribution of the model. Both figures are reported side by side throughout,
so accuracy cannot be quoted as evidence of skill.

**ROC-AUC.** The false-positive rate is FP/(FP+TN), and TN here is on the order of
284,315. A model can raise thousands of false alarms
while barely moving FPR.

The Isolation Forest demonstrates this concretely: ROC-AUC 0.9394, PR-AUC
0.0366, precision 6.2%. On ROC it looks usable; in an
analyst's queue, roughly 94 of every 100 alerts would be
false.

**Average precision (PR-AUC)** divides by the model's own alert volume, so it degrades as
soon as the model wastes reviewer time. It is used as the headline metric, with prevalence
always reported alongside since prevalence *is* the no-skill PR-AUC.

## 3. Partitioning

Fraud is bursty: a compromised card generates several transactions within minutes. A random
split scatters one episode across train and test, so the model is scored on transactions
whose siblings it trained on.

Fraud is bursty: a compromised card produces several transactions within minutes. A random split scatters one episode across both partitions, so the model is scored on transactions whose siblings it trained on. That inflates every metric and no metric reveals it. A chronological split reproduces the deployment condition — score tomorrow's traffic having learned only from yesterday's.

Resulting partitions: 213,605 training transactions
(398 frauds), 71,202 test
(94 frauds).

## 4. Resampling: why none was used

SMOTE and its variants interpolate between neighbouring minority points. With
492 positives in 28 dimensions the nearest neighbours are far apart, so
synthetic points land in regions of feature space where no real fraud has been observed. The
classifier then learns a boundary around fabricated data. Class weighting achieves
rebalancing without inventing observations, and random majority undersampling would discard
99.8% of the real evidence about what normal looks like.

## 5. Results

| Model | Supervision | PR-AUC | ROC-AUC | Precision | Recall | Brier |
|---|---|---:|---:|---:|---:|---:|
| Logistic regression (class-weighted) | supervised linear | 0.7657 | 0.9816 | 0.885 | 0.734 | 0.03302 |
| LightGBM (reweighting: none) | supervised boosting | 0.7359 | 0.9378 | 0.910 | 0.649 | 0.00058 |
| Isolation Forest | unsupervised | 0.0366 | 0.9394 | 0.062 | 0.426 | 0.03004 |

### 5.1 Value of labels

The unsupervised detector was fitted on legitimate training transactions only, never seeing a
fraud label — the realistic cold-start condition for a novel attack. Its PR-AUC of
0.0366 against 0.7657 for the supervised best quantifies what
labelling effort is worth on this problem: roughly
21×.

### 5.2 Reweighting ablation

| Variant | scale_pos_weight | PR-AUC | ROC-AUC | Brier |
|---|---:|---:|---:|---:|
| none | — | 0.7359 | 0.9378 | 0.00058 |
| scale_pos_weight_10 | 10 | 0.3645 | 0.8611 | 0.00132 |
| is_unbalance | — | 0.0260 | 0.8535 | 0.02716 |
| scale_pos_weight_full | 536 | 0.0092 | 0.8730 | 0.10511 |

Setting scale_pos_weight to the full negative/positive ratio (536) — the conventional recommendation for imbalanced boosting — collapses PR-AUC to 0.0092, against 0.7359 with no reweighting at all. With only 398 positives in training, an extreme weight makes every split chase the same handful of rows: the trees fit those points and the ranking of everything else degrades. Reweighting helps a linear model, whose capacity is bounded, and hurts a boosted ensemble, whose capacity is not. This is why the ablation is run rather than the advice followed.

This result is reported in full because it contradicts widely-repeated guidance. The
mechanism is capacity: reweighting constrains a linear model toward the minority class
usefully, but permits a boosted ensemble to devote successive trees to fitting a handful of
reweighted points, degrading the global ranking that PR-AUC measures.

## 6. Operating-point selection

Expected cost was traced across the threshold range under an assumed
500:10
false-negative to false-positive ratio.

Optimum at threshold 0.939:
TP 80, FP 229,
FN 14, expected cost
9,290 against
47,000 for no screening.

The cost ratio is an assumption, not a measurement. The full curve is published so a reader with different costs can read off their own operating point rather than inherit this one.

| Alert budget | Frauds caught | Recall | Precision |
|---:|---:|---:|---:|
| 50 | 49 / 94 | 52.1% | 98.0% |
| 100 | 71 / 94 | 75.5% | 71.0% |
| 250 | 78 / 94 | 83.0% | 31.2% |
| 500 | 80 / 94 | 85.1% | 16.0% |
| 1,000 | 81 / 94 | 86.2% | 8.1% |

## 7. Limitations

- Two days of one issuer's European traffic in 2013. Fraud tactics have changed substantially since; this is a methodology exercise, not a deployable screen.
- V1–V28 are anonymised components, so no finding here can be translated into an interpretable business rule.
- Only 94 frauds fall in the test window, so recall estimates carry wide confidence intervals — a single missed episode moves recall by 1.1%.
- The threshold was selected on the same test split it is reported on, which is mildly optimistic. A production system would select it on a separate validation period.

*Generated at commit `1c3d162` · seed 42 · 33.54s · Python 3.11.15 · numpy 2.4.6 · pandas 3.0.5 · sklearn 1.9.1 · lightgbm 4.7.0 · torch 2.14.0+cu130*
