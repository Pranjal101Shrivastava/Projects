# Video 04 · Fraud Detection

**Estimated runtime:** ~6:03 at a measured pace (817 spoken words)
**Live page:** https://pranjal101shrivastava.github.io/Projects/#/p/fraud
**Project directory:** [`projects/04_fraud_detection/`](../../projects/04_fraud_detection/)

> **The one sentence this video has to land:** PR-AUC 0.7657 against a 0.00132 no-skill floor — and the textbook reweighting advice made the boosted model about 80x worse, which is published rather than tuned away.

## Before you record

- Open **https://pranjal101shrivastava.github.io/Projects/#/p/fraud**.
- Have the model table, the ablation chart and the cost curve located before you start.

Everything in plain text below is meant to be **spoken**. Lines marked *On screen* and
*Note to self* are directions — do not read them aloud.

---

## [0:00] Cold open

> **On screen —** the headline stats, PR-AUC beside the no-skill floor

This model is 98.2% accurate by area under the ROC curve.

And I want to start by telling you that number is close to useless here, and show you what
to look at instead.

This is credit card fraud. 284,807 real transactions, of which
0.173% are fraudulent — 492 cases in total.

## [0:24] Why accuracy and ROC both lie here

> **On screen —** the prevalence and always-negative-accuracy figures

Start with accuracy, because it's the most dangerous number in this entire dataset.

If I write a model that returns "not fraud" for every single transaction — one line, no
training, no data — it is 99.8% accurate.

That's not a good model. That's an arithmetic consequence of the class balance. Any metric
where doing nothing scores that well is not measuring skill.

ROC-AUC is subtler, and it's the one people get caught by. ROC plots the true positive rate
against the false positive rate. The false positive rate has the number of *negatives* in
its denominator — and here there are hundreds of thousands of them. So you can add thousands
of false alarms and barely move the curve.

The measure that doesn't have that problem is precision-recall AUC, because precision is
computed against the number of predicted positives. Add false alarms and precision falls
immediately.

So: the no-skill floor for PR-AUC is just the prevalence — 0.00132.

My best model scores 0.7657.

That's a lift of about 580 times over the floor. *That*
is the number worth quoting, and it's only meaningful because the floor is printed next to
it.

## [1:48] Three models, and the one that won is the boring one

> **On screen —** the model comparison table

I ran three approaches that make genuinely different assumptions.

An isolation forest — unsupervised, it doesn't use the labels at all, it just looks for
points that are easy to isolate. PR-AUC 0.0366.

Gradient boosting — LightGBM, fully supervised. PR-AUC 0.7359.

And class-weighted logistic regression — a linear model, the simplest thing here.
PR-AUC 0.7657.

The linear model won.

I want to sit on that for a second, because there's a lesson in it. This dataset is mostly
principal components — it's already been through PCA — and the fraud signal in it is largely
linear in that space. A more flexible model has more ways to overfit
492 positive examples and no extra structure to exploit.

Reaching for the most powerful model available is a reflex. It's often wrong, and the only
way to know is to run the simple one too.

## [2:51] The ablation — where standard advice cost me 80x

> **On screen —** the reweighting ablation chart

Now the part I'm most glad I didn't skip.

Every guide to imbalanced classification with gradient boosting tells you the same thing:
set `scale_pos_weight` to the ratio of negatives to positives. Here that ratio is about
536.

I did that. PR-AUC came out at 0.0092.

That is catastrophically bad — barely above the no-skill floor. And here's the moment that
matters: my instinct was to assume I'd made a mistake, quietly tune it until it looked
reasonable, and never mention it.

Instead I ran the whole ablation. Four configurations, all published.

With no reweighting at all: 0.7359.

With the recommended weight of 536: 0.0092.

That's roughly 80 times worse — by following
the standard advice.

And there's a real reason. With only a few hundred positive rows in training, an extreme
weight makes every split in every tree chase the same handful of points. The trees fit those
rows, and the ranking of everything else degrades.

Notice it helped the linear model and destroyed the boosted one. A linear model's capacity
is bounded, so reweighting shifts its decision boundary. A boosted ensemble's capacity isn't
bounded, so it just memorises the upweighted rows.

The advice isn't wrong in general. It's wrong here. And the only way to know which is to run
the ablation instead of following the guidance.

## [4:28] The threshold is a business decision, not a statistical one

> **On screen —** the cost curve

One more thing, and it's the part that would actually get this deployed.

A model outputs a score. Someone has to pick the cutoff. And the default — point five —
has no justification whatsoever in a problem like this.

The right cutoff depends on relative cost. A missed fraud costs the transaction value plus
the chargeback. A false alarm costs an analyst's time and annoys a customer.

So this curve sweeps the threshold and shows total cost under different assumptions about
that ratio. There's no single optimum, because there's no single business.

What that means practically: I can't hand you a threshold. I can hand you the curve and the
question — how much is a missed fraud worth relative to a false alarm? Answer that, and the
curve gives you the cutoff.

That's the honest handoff.

## [5:28] Close

> **On screen —** the Method tab

So: a metric chosen because the obvious ones are misleading at this prevalence, reported
next to the floor that makes it interpretable. The simplest model winning. An ablation that
contradicts the standard advice, published because it contradicts it. And a threshold left
as the business decision it actually is.

Next video: I run the same model tournament twice — once with one extra column, and once
without it — and price exactly what a single leaked feature is worth.
---

*Generated by [`tools/video_scripts.py`](../../tools/video_scripts.py) from committed
artifacts. Every figure spoken here is the figure on screen.*
