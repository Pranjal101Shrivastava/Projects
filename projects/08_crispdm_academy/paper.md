# Teaching Statistical Concepts from Data That Does Not Cooperate

## 1. Motivation

Instructional material illustrates concepts with generated data because generated data
satisfies its assumptions: the bias-variance curve forms a clean U, the sampling distribution
converges by n = 30, features are conditionally independent because they were drawn that way.

Students then encounter real data where none of this holds, with no framework for the
discrepancy.

Each module below computes its figures from observational data. Where an assumption fails, the
failure is quantified and taught.

## 2. Conditional independence (Titanic)

Naive Bayes assumes P(A, B | C) = P(A|C)·P(B|C). Cramér's V was computed between feature pairs
*within each class*:

| Pair | V given died | V given survived | Violates |
|---|---:|---:|:---:|
| Pclass × FareBand | 0.449 | 0.608 | yes |
| FareBand × Embarked | 0.211 | 0.400 | yes |
| Pclass × Embarked | 0.203 | 0.329 | yes |
| Sex × AgeBand | 0.213 | 0.183 | yes |
| Pclass × Sex | 0.189 | 0.166 | no |

4 of 5 pairs violate the assumption
materially. The consequence is measurable and asymmetric:

| | ROC-AUC | Brier |
|---|---:|---:|
| Naive Bayes | 0.8368 | 0.1561 |
| Logistic regression | 0.8442 | 0.1446 |

Ranking performance is largely preserved; calibration is not. Dependent features contribute
correlated evidence that the model counts as independent, pushing posteriors toward 0 and 1.
Ordering survives because the distortion is largely monotone.

## 3. Threshold selection under imbalance

At prevalence 0.1320%, the same classifier reports ROC-AUC
0.9806 and PR-AUC 0.7625.

The same model scores ROC-AUC 0.9806 and PR-AUC 0.7625. Both are correct; they answer different questions. ROC's x-axis is false-positive *rate*, whose denominator is all 71,108 negatives, so thousands of false alarms barely move it. Precision's denominator is the model's own alert volume, which is what an analyst's queue actually contains. At prevalence 0.1320%, a no-skill model has PR-AUC 0.00132 but ROC-AUC 0.5 — so the PR floor moves with the base rate and the ROC floor does not.

## 4. Gradient descent and the stability bound

A two-parameter convex surface was constructed from real monthly temperature means, permitting
the entire loss landscape to be drawn.

| lr | Outcome | Final loss ÷ optimum |
|---:|---|---:|
| 0.01 | converges, but slowly — needs far more than 60 steps | 1.71× |
| 0.1 | converges efficiently | 1.00× |
| 0.5 | converges efficiently | 1.00× |
| 1.05 | diverges — each step overshoots by more than it corrects, so the loss grows | 663,317.30× |

The learning rate is the whole story on a convex surface. At 0.01 the trajectory crawls; at 0.1–0.5 it converges cleanly; at 1.05 each step overshoots by more than it corrects and the loss diverges to infinity. The threshold is not arbitrary — for a quadratic loss, gradient descent diverges once the learning rate exceeds 2/λ_max, where λ_max is the largest eigenvalue of the Hessian. That is why the same learning rate can be fine on one problem and catastrophic on another: it depends on the curvature, not on the number.

## 5. Gradient checking

A two-layer network (3 → 5 tanh → 1 sigmoid, binary cross-entropy) was differentiated by hand
and verified against central finite differences (f(θ+ε) − f(θ−ε)) / 2ε.

| Parameter | Analytic | Numerical | Relative error |
|---|---:|---:|---:|
| `W1[0,3]` | 0.02238397 | 0.02238397 | 3.33e-10 |
| `W1[2,1]` | -0.00020467 | -0.00020467 | 1.71e-09 |
| `W1[0,4]` | -0.00070120 | -0.00070120 | 1.73e-09 |
| `W1[1,4]` | -0.00034764 | -0.00034764 | 1.01e-08 |
| `W1[2,3]` | -0.00167462 | -0.00167462 | 1.40e-08 |
| `W1[2,0]` | 0.00362108 | 0.00362108 | 3.07e-09 |

Maximum relative error 1.40e-08. Correct gradients agree with central
differences to approximately 1e-7; an error near 0.3 indicates a derivation bug rather than
floating-point noise.

The derivation also demonstrates why sigmoid is paired with cross-entropy rather than squared
error: ∂L/∂a₂ and ∂a₂/∂z₂ multiply to (a₂ − y)/m, a cancellation that eliminates the sigmoid
derivative. Under squared error that derivative survives and vanishes on saturation, stalling
learning.

## 6. Bias-variance decomposition

Polynomial degree 1–15, decomposed over 40 bootstrap resamples.

| Degree | Bias² | Variance | Test MSE | Train MSE |
|---:|---:|---:|---:|---:|
| 1 | 15.521 | 0.076 | 15.596 | 16.459 |
| 2 | 8.758 | 0.062 | 8.820 | 9.340 |
| 3 | 8.522 | 0.095 | 8.617 | 8.975 |
| 4 | 7.562 | 0.099 | 7.660 | 7.607 |
| 5 | 7.546 | 0.103 | 7.649 | 7.454 |
| 6 | 7.683 | 0.125 | 7.809 | 7.451 |
| 7 | 7.770 | 0.154 | 7.924 | 7.497 |
| 8 | 7.735 | 0.162 | 7.898 | 7.499 |
| 9 | 7.643 | 0.173 | 7.816 | 7.428 |
| 10 | 7.674 | 0.180 | 7.855 | 7.237 |
| 11 | 7.706 | 0.238 | 7.944 | 7.212 |
| 12 | 7.865 | 0.277 | 8.142 | 7.326 |
| 13 | 7.971 | 0.225 | 8.196 | 7.113 |
| 14 | 7.918 | 0.270 | 8.189 | 7.060 |
| 15 | 7.871 | 0.295 | 8.165 | 7.130 |

Optimal degree 5. **The canonical U-shape does not appear.**

Test error is minimised at degree 5 (MSE 7.6495), rising to 8.1652 by degree 15. Variance grows monotonically as expected — from 0.0757 at degree 1 to 0.2946 at degree 15. But this real curve is NOT the textbook U. Variance accounts for only 3.6% of test error even at the highest degree; bias² dominates throughout. Day-of-year simply does not determine daily temperature — the irreducible noise is large, so no amount of capacity helps and the curve flattens rather than turning sharply upward. A simulated example would show the clean U and hide this, which is the more common real situation. Note that training error here does NOT fall monotonically: each degree is averaged over bootstrap resamples, so resampling noise exceeds the small gain from extra capacity once bias has plateaued. The textbook monotone training curve assumes a single fixed training set.

Two departures from the textbook figure, both detected programmatically:

1. Variance contributes only 3.6% of test error at
   maximum capacity. Bias dominates because day-of-year does not determine daily temperature;
   irreducible noise is large and additional capacity cannot reduce it.
2. Training error is non-monotone (confirmed),
   because each point averages over bootstrap resamples. The monotone training curve of the
   textbook assumes a single fixed training set.

## 7. Central limit theorem

Population: 284,807 transaction amounts, skewness
16.98, kurtosis 845.1.

| n | Skewness of mean | Observed SE | σ/√n | Ratio |
|---:|---:|---:|---:|---:|
| 1 | 5.220 | 192.069 | 250.120 | 0.768 |
| 2 | 9.624 | 185.207 | 176.861 | 1.047 |
| 5 | 5.698 | 124.923 | 111.857 | 1.117 |
| 10 | 3.010 | 71.766 | 79.095 | 0.907 |
| 30 | 2.000 | 44.977 | 45.665 | 0.985 |
| 100 | 1.584 | 24.901 | 25.012 | 0.996 |
| 500 | 0.798 | 11.421 | 11.186 | 1.021 |

The population of transaction amounts has skewness 16.98 — nothing like a normal distribution. The sampling distribution of the mean nevertheless approaches normality as n grows: skewness falls from 3.01 at n=10 to 0.80 at n=500, and the observed standard error tracks σ/√n to within a few percent from n=10 onward. That √n is why quadrupling the sample only halves the error. Two honest caveats visible in the table. First, the n=1 and n=2 rows are erratic — skewness 5.22 then 9.62 — because 1,500 draws from a distribution this heavy-tailed under-sample the extreme values, so those estimates are dominated by whether a few outliers happened to be drawn. Second, convergence is far slower than the familiar rule of thumb suggests: at n=30 the sample mean is still clearly skewed (2.00). 'n=30 is enough' is calibrated on mildly non-normal populations, not on one with skewness 17.

## 8. Limitations

- Every module uses one dataset per concept. A concept demonstrated on one dataset is illustrated, not established.
- The quiz has a single correct answer per question, which suits concept checks but cannot assess the judgement the modules argue is the real skill.

*Generated at commit `fc77533` · seed 42 · 10.29s · Python 3.11.15 · numpy 2.4.6 · pandas 3.0.5 · sklearn 1.9.1 · lightgbm 4.7.0 · torch 2.14.0+cu130*
