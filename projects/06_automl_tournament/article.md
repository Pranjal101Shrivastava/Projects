# One column was worth 40% of my model's performance. It shouldn't have been there.

*A tournament run twice, to price a leak*

The UCI Bank Marketing dataset is a standard benchmark. 41,188 real marketing calls from a
Portuguese bank, predicting who subscribes to a term deposit.

It also ships with a trap, and UCI tells you about it:

> *"this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet,
> the duration is not known before a call is performed... this input should be discarded if
> the intention is to have a realistic predictive model."*

That is the documentation for a column called `duration` — how long the sales call lasted.

Search for notebooks on this dataset. Most of them use it.

## So I measured what it is worth

I built a proper AutoML tournament: four model families with different inductive biases, 12
randomised hyperparameter draws each, 4-fold stratified cross-validation scored by average
precision, plus a stacked ensemble.

Then I ran the entire thing **twice**. Identical code, identical folds, identical seeds. The
only difference: one column present or absent.

| Model | Without `duration` | With `duration` |
|---|---:|---:|
| stacked_ensemble | 0.4961 | **0.6939** |
| random_forest | 0.4960 | **0.6917** |
| hist_gradient_boosting | 0.4925 | **0.6875** |
| logistic | 0.4694 | **0.6235** |
| gaussian_nb | 0.3607 | **0.4003** |

**39.9% inflation on the headline number.** Every model. From one
column.

## Why it is a leak and not just a good feature

The statistical signal is suspicious:

- `duration` alone reaches **ROC-AUC 0.818**
- Subscribers average **553 seconds** on the call
- Decliners average **221 seconds**

But suspicion is not proof. Some features really are that good.

Here is the proof. There are **4 calls with duration = 0** in the
dataset. Number of those that resulted in a subscription: **0**.

Of course. You cannot subscribe during a call that did not happen.

That is not a feature predicting an outcome. That is an outcome leaving a fingerprint on a
feature.

## The part that should worry you

I want to be precise about what did *not* catch this.

- Cross-validation: clean. CV and test scores agreed in both runs.
- The generalisation gap: healthy.
- Calibration: fine.
- Confusion matrix: normal.
- Learning curves: unremarkable.

Every diagnostic a careful practitioner consults reported that the leaked model was sound.
Because it *was* sound — it modelled the data it was given, correctly. The data was wrong.

There is no statistical test for this. A leaking feature and a strong feature have the same
distributional signature. What distinguishes them is *when the value comes into existence*,
and that is a question about the world.

## The two questions

**Does any single feature predict the target implausibly well on its own?** A screening
heuristic. Worth running. Not conclusive.

**Would this value exist at the moment I have to make the prediction?**

That is the one. Sit with each column and ask when it gets populated. For a call-prioritisation
model, scoring happens *before dialling*. Call duration does not exist yet. Neither does the
outcome, the agent's notes, or anything else the call produces.

It takes ten minutes and requires no mathematics, and it is the only thing that catches a leak
worth 40% of your reported performance.

## What I shipped

Both leaderboards. The leak-free one is labelled deployable; the leaked one is labelled a
demonstration. Publishing only the clean numbers would have hidden the most useful thing I
learned.

If you benchmark against published results on this dataset, check whether they used
`duration`. Most did. The comparison is not like-for-like.

---

**Live demo:** [https://pranjal101shrivastava.github.io/Projects/#/p/automl](https://pranjal101shrivastava.github.io/Projects/#/p/automl)
**Code:** [github.com/Pranjal101Shrivastava/Projects](https://github.com/Pranjal101Shrivastava/Projects)
