# Selecting a Segmentation for Reproducibility Rather Than Geometry

*A CRISP-DM study on 41,188 real bank marketing contacts*

## 1. Motivation

A term-deposit campaign operates under a fixed calling budget. Segmentation is only worth
performing if it changes who gets called, which requires two properties simultaneously: the
segments must reproduce on a new sample of the same population, and they must differ in
outcome. Stability without outcome separation yields tidy segments nobody can act on;
outcome separation without stability yields a story that will not survive the next quarter.

## 2. Feature space

The subscription outcome was withheld entirely from clustering and reserved for validation.
Call duration was also excluded: it records how long the call lasted, which is known only
after the outcome is effectively settled, so clustering on it would embed the answer in the
segments.

Missing categorical values, coded as the literal string `unknown`, were retained as an
explicit level rather than mode-imputed. A customer whose employment or education the bank
failed to record differs systematically from one it recorded; imputation would erase that
signal and manufacture certainty the data does not contain.

Two candidate feature sets were carried forward — customer attributes alone, and customer
attributes plus macroeconomic context — because euribor3m and related series vary with the
*date of the call* rather than with the person, risking segments that are really time
periods wearing customer labels. The choice between them was made on stability rather than
asserted in advance; `customer_only` was selected.

## 3. Model selection

Two criteria were applied in sequence.

**Internal validity.** Silhouette rewards separation, Calinski-Harabasz rewards compactness
relative to spread, Davies-Bouldin penalises overlap. Their agreement or disagreement is
informative in itself.

**Bootstrap stability.** For each k, a reference partition was computed, then
30 resamples with replacement were independently clustered and compared to
the reference on their shared rows by adjusted Rand index. Mean ARI ≥ 0.75
was required.

| Criterion | Preferred k |
|---|---|
| Silhouette | 2 |
| Calinski-Harabasz | 2 |
| Davies-Bouldin | 2 |
| Stability (ARI ≥ 0.75) | 6, 7 |

The criteria disagree, and the disagreement is the substantive result. Every index prefers
k = 2; no such partition reproduces. We adopt k = 6.

## 4. External validation

| Seg | Customers | Share | Conversion | Lift | Mean calls | Prior contact |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,515 | 3.7% | 63.83% | 5.67× | 1.8 | 100% |
| 3 | 4,019 | 9.8% | 12.44% | 1.10× | 2.0 | 0% |
| 4 | 13,314 | 32.3% | 12.34% | 1.09× | 2.1 | 0% |
| 2 | 9,878 | 24.0% | 9.68% | 0.86× | 2.2 | 0% |
| 0 | 10,870 | 26.4% | 4.67% | 0.41× | 2.2 | 0% |
| 5 | 1,592 | 3.9% | 4.15% | 0.37× | 12.9 | 0% |

χ² = 4786.4 on 5 degrees of freedom,
p < 1e-300.

Because the outcome was never available to the clustering algorithm, this separation is
external evidence rather than a restatement of the optimisation objective.

The operationally significant segment is 5: 1,592 customers
receiving 12.9 calls on average and converting at
4.15%. Contact effort is being spent on a saturated cohort.

## 5. How much structure is real?

| Comparison | Adjusted Rand index |
|---|---:|
| K-means vs Gaussian mixture | 0.4924 |
| K-means vs Ward linkage | 0.5046 |

Agreement between algorithms with different geometric assumptions is evidence that the structure is in the data rather than in the method. Disagreement means at least one algorithm is imposing its own shape.

At approximately 0.5, roughly half the partition structure derives from K-means's isotropic
assumption rather than from the data. This is reported because a segmentation presented
without it invites the reader to treat cluster boundaries as discovered natural kinds.

## 6. Limitations

- Macroeconomic columns (euribor3m, emp.var.rate, nr.employed) vary with calendar time rather than with the customer. Including them risks segmenting by *when* someone was called rather than by who they are — examined in the feature-set ablation below.
- The three validity indices unanimously prefer k=2, while the bootstrap admits only k∈[6, 7]. Geometry and reproducibility disagree, and the reproducible answer was taken. A k=2 split would be the cleanest geometrically and would also be nearly useless commercially, which is a reminder that internal indices optimise a mathematical objective rather than a business one.
- K-means agrees with GMM at ARI 0.4924 and with Ward linkage at 0.5046 — moderate, not strong. Roughly half the partition structure is therefore imposed by K-means's spherical assumption rather than present in the data. Segment boundaries should be treated as one defensible partition among several, not as discovered natural kinds.
- The highest-converting segment is defined largely by having been contacted in a previous campaign (100% prior-contact share). That is a legitimate predictor — it is known before the call is placed — but it means the segment mostly re-identifies already-engaged customers rather than revealing a latent group.
- Segments are descriptive, not causal. A segment converting at 3× the average is not evidence that moving a customer into it would raise their probability of subscribing.
- One institution, one product, 2008–2010, during a financial crisis. Segment structure is unlikely to transfer.

*Generated at commit `4aa8c30` · seed 42 · 347.5s · Python 3.11.15 · numpy 2.4.6 · pandas 3.0.5 · sklearn 1.9.1 · lightgbm 4.7.0*
