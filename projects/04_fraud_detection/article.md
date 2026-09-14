# A model with 99.95% accuracy that is worth almost nothing

*And the ablation where following best practice made things 82× worse*

The dataset is 284,807 real credit card transactions. 492
of them are fraud. That is **0.1727%**.

My best model achieves **99.952% accuracy**.

Here is a model that achieves 99.868%:

```python
def predict(transaction):
    return "legitimate"
```

That is the whole model. It never fires. It catches nothing. And it is within
0.084 percentage points of
mine.

Any write-up that leads with accuracy on this dataset is either careless or hiding something.

## ROC-AUC has the same problem, less obviously

Everyone knows about the accuracy trap. Fewer people notice that ROC-AUC has a milder version
of it.

I trained an Isolation Forest — an unsupervised detector that only ever sees legitimate
transactions and flags departures from them.

**ROC-AUC: 0.939.** That is a number you would put on a slide.

**Precision: 6.2%.**

For every fraud it catches, it raises about 15 false
alarms. An analyst working that queue spends their entire day on legitimate transactions.

The reason ROC hides this is its denominator. False-positive rate is FP/(FP+TN), and there
are 284,315 negatives here. Thousands of false alarms
barely register. Precision divides by the model's *own alert volume* — the thing that is
actually in the queue.

So I lead with **PR-AUC: 0.7657**, and always print the no-skill floor
(0.00173) next to it.

## The ablation I did not expect

I set out to do the standard thing for imbalanced boosting: set `scale_pos_weight` to the
ratio of negatives to positives. Every tutorial says so. LightGBM's own docs say so.

My model came out at PR-AUC **0.0089**.

That is catastrophically bad — worse than the unsupervised detector. So instead of quietly
tuning until it looked better, I ran all four options:

| What I did | PR-AUC |
|---|---:|
| none | **0.7359** |
| scale pos weight 10 | **0.3645** |
| is unbalance | **0.0260** |
| scale pos weight full | **0.0092** |

**Doing nothing beat the recommended practice by 82×.**

The mechanism, once you see it, is obvious. There are 398 fraud cases in the training set.
Weighting them 536× tells the model that those 398 rows matter more than the other 213,000
combined. A linear model, whose capacity is bounded, responds by shifting its decision
boundary — helpful. A boosted ensemble, whose capacity is not bounded, responds by spending
tree after tree fitting those specific 398 points. It memorises them and the ranking of
everything else falls apart.

Reweighting advice that is correct for logistic regression is not automatically correct for
gradient boosting. Ablations exist because intuitions do not transfer.

## The threshold is not 0.5

`predict()` uses 0.5 by default. There is no reason for that number to be right.

Suppose a missed fraud costs 500 and a wasted
investigation costs 10. Then you can compute the
expected cost at every threshold and pick the minimum — which lands at
**0.939**, catching
80 frauds for
229 false alarms.

But the more useful framing is capacity. An analyst can review so many alerts per day:

| Alerts/day | Frauds caught | Precision |
|---:|---:|---:|
| 50 | 49 / 94 | 98.0% |
| 100 | 71 / 94 | 71.0% |
| 250 | 78 / 94 | 31.2% |
| 500 | 80 / 94 | 16.0% |
| 1,000 | 81 / 94 | 8.1% |

Fifty alerts catches over half the fraud at 98.0%
precision. A thousand alerts catches 86.2% at
8.1% precision.

You are not choosing a threshold. You are choosing how many analysts to hire.

## One more thing: I split by time, not at random

Fraud is bursty. Steal a card, and you use it several times in the next twenty minutes.

Split randomly and those transactions land on both sides of the split. Your model sees three
transactions from an episode in training, then gets scored on the fourth — and recognises the
pattern because it has literally seen its siblings.

Every metric goes up. No metric tells you why.

I split chronologically: train on the first 75% of time, test on the last 25%. That is what
deployment actually looks like — score tomorrow having learned from yesterday. It is also why
my numbers are lower than most published notebooks on this dataset.

---

**Live demo:** [https://pranjal101shrivastava.github.io/Projects/#/p/fraud](https://pranjal101shrivastava.github.io/Projects/#/p/fraud)
**Code:** [github.com/Pranjal101Shrivastava/Projects](https://github.com/Pranjal101Shrivastava/Projects)
