# The bias-variance curve in your textbook is not what real data does

*Six statistics lessons rebuilt on real data, and what changed*

Open any machine learning textbook to the bias-variance section. You will find a figure: bias
falling, variance rising, total error tracing a clean U with an obvious minimum.

I rebuilt that figure from real Melbourne temperature data. Here is what came out.

| Degree | Bias² | Variance | Test MSE |
|---:|---:|---:|---:|
| 1 | 15.52 | 0.08 | 15.60 |
| 4 | 7.56 | 0.10 | 7.66 |
| 7 | 7.77 | 0.15 | 7.92 |
| 10 | 7.67 | 0.18 | 7.85 |
| 13 | 7.97 | 0.22 | 8.20 |

Variance does rise, exactly as promised. But look at the magnitudes. Even at degree 15,
variance accounts for **3.6% of total test error**. Bias
dominates the entire range.

There is no dramatic U. There is a shallow dip at degree 5 and then a
gentle drift upward.

## Why the real curve is flat

The textbook U assumes you are fitting a signal that more capacity can capture. Here the
relationship is day-of-year → daily temperature, and day-of-year *does not determine* daily
temperature. Weather is noisy. A perfect model of the seasonal cycle still leaves enormous
irreducible error.

When irreducible noise dominates, extra capacity has almost nothing to grip. Bias plateaus,
variance grows slowly from a tiny base, and the curve flattens.

That is the common case in applied work, and it is the case the textbook figure does not show.

## Training error did not fall monotonically either

Every treatment states that training error decreases monotonically with capacity. Mine does
not, and the reason is instructive.

I averaged each degree over **40 bootstrap resamples** rather than fitting
once. Once bias plateaus, the resample-to-resample variation exceeds the marginal gain from an
extra polynomial term. The averaged training curve wobbles.

The monotone guarantee holds for *one fixed training set*. Average over resamples — which is
what you must do to decompose bias and variance at all — and it stops holding.

## "n = 30" is not a rule

Everyone learns that the central limit theorem kicks in around n = 30.

I tested that on real transaction amounts, a distribution with skewness
**17.0** — an extremely heavy right tail.

| Sample size | Skewness of the sample mean |
|---:|---:|
| 1 | 5.22 |
| 2 | 9.62 |
| 5 | 5.70 |
| 10 | 3.01 |
| 30 | 2.00 |
| 100 | 1.58 |
| 500 | 0.80 |

At n = 30 the sample mean is still skewed **2.00**. That is not approximately
normal. You would need several hundred observations before a normal approximation is safe here.

"n = 30" was calibrated on populations that are mildly non-normal. Applied to a genuinely
skewed one it is simply wrong, and nobody mentions the caveat.

## Naive Bayes with its assumption comprehensively broken

Naive Bayes assumes features are conditionally independent given the class. On Titanic I
measured how badly that fails: **4 of
5 pairs** violate it materially. Passenger class and fare band
reach Cramér's V 0.61
— they are nearly deterministic in each other, which is obvious in hindsight since first-class
tickets cost more.

So the model should fail. It does not:

| | ROC-AUC | Brier score |
|---|---:|---:|
| Naive Bayes | 0.837 | 0.1561 |
| Logistic regression | 0.844 | 0.1446 |

Ranking barely suffers. Calibration measurably does.

The reason is that dependent features contribute the same evidence twice, and Naive Bayes
counts it twice, pushing probabilities toward 0 and 1. But the distortion is mostly *monotone*
— it inflates confidence without reordering cases. So AUC, which only cares about order,
survives. Brier, which cares about the actual numbers, does not.

Which is why Naive Bayes is a fine ranker and a bad probability estimate. That sentence is in
every textbook. Measuring it is more convincing than reading it.

## And the learning rate that explodes

I traced gradient descent across a real loss surface at four learning rates:

| lr | What happened |
|---:|---|
| 0.01 | converges, but slowly — needs far more than 60 steps |
| 0.1 | converges efficiently |
| 0.5 | converges efficiently |
| 1.05 | diverges — each step overshoots by more than it corrects, so the loss grows |

At the largest rate the loss ends
**663,317× above the minimum**.

Teaching figures almost never show divergence. They show a nicely converging trajectory and
mention in the caption that too large a rate is bad. Watching the loss go to infinity on a
surface you can see is considerably more memorable.

---

**Live demo:** [https://pranjal101shrivastava.github.io/Projects/#/p/academy](https://pranjal101shrivastava.github.io/Projects/#/p/academy)
**Code:** [github.com/Pranjal101Shrivastava/Projects](https://github.com/Pranjal101Shrivastava/Projects)
