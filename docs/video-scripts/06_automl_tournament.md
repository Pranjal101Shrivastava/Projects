# Video 06 · AutoML & the Leak

**Estimated runtime:** ~4:50 at a measured pace (653 spoken words)
**Live page:** https://pranjal101shrivastava.github.io/Projects/#/p/automl
**Project directory:** [`projects/06_automl_tournament/`](../../projects/06_automl_tournament/)

> **The one sentence this video has to land:** One column that cannot exist at scoring time inflates PR-AUC from 0.4961 to 0.6939 — 39.9% — and no standard diagnostic flags it.

## Before you record

- Open **https://pranjal101shrivastava.github.io/Projects/#/p/automl**.
- Have the clean-vs-leaked comparison and the per-model table visible.
- This is the flagship video — consider recording it first while you're freshest.

Everything in plain text below is meant to be **spoken**. Lines marked *On screen* and
*Note to self* are directions — do not read them aloud.

---

## [0:00] Cold open

> **On screen —** the clean-vs-leaked comparison

If you take one thing from this entire portfolio, I'd like it to be this video.

Same dataset. Same models. Same cross-validation. Same everything — except one column.

Without that column, the best model scores 0.4961.

With it, 0.6939.

That's a 39.9% improvement from a single feature. And that
improvement is completely fake. It would vanish the moment you deployed it.

Here's how to spot it.

## [0:30] The column

> **On screen —** the diagnostic panel for the leaked column

The column is called `duration`. It's the length of the marketing phone call, in
seconds.

And the question that exposes it is the only question that matters for leakage:

**Would this value exist at the moment I have to make the prediction?**

No. It wouldn't.

The prediction I'm making is: should we call this person? If I haven't called them yet, the
call has no length. The value doesn't exist. It comes into existence *because* of the thing
I'm trying to predict.

Look at the numbers. Customers who subscribed averaged 553
seconds on the phone. Customers who declined averaged 221.

That's not the feature predicting the outcome. That's the outcome leaving a fingerprint on
the feature. A long call happened because it was going well.

## [1:25] Why every standard diagnostic misses it

> **On screen —** the per-model table

Now here's the part that should worry you.

That column on its own — one feature, nothing else — gets a ROC-AUC of
0.818.

A single column reaching 0.82 is a red flag, not a feature. But
nothing *automatically* flags it.

Cross-validation doesn't catch it, because the column is present in training and test alike.
The confusion matrix looks great. The calibration curve looks great. Learning curves look
great. Feature importance shows it at the top, which reads like a success.

And the inflation isn't limited to one model. Look at the table —
5 different model families, and every one of them gains:
stacked ensemble from 0.4961 to
0.6939, and so on down the list.

The leak doesn't care what algorithm you use. It's a property of the *data*, and every
diagnostic I've listed measures how well the model fits the data.

The only thing that catches it is reasoning about *when each value becomes known*. That's
not a statistical test. It's a question you ask about every column, and it takes about ten
minutes.

## [2:44] Why I ran the tournament twice

> **On screen —** both tournaments side by side

Here's why this is set up as two full tournaments rather than one warning in a paragraph.

I could have written "don't include leaked features." Everyone has read that sentence.
Nobody changes their behaviour because of it.

What changes behaviour is a number. Running the identical pipeline twice — once clean, once
leaked — *prices* the leak. 39.9% of PR-AUC. That's what it's
worth, on this data, with these models.

And note which way round it is. The clean number, 0.4961, is the
honest one. It's also the less impressive one. If I were optimising for a portfolio that
looks good, I'd keep the column and never mention it — and there is genuinely no way for a
reader to tell from the outputs alone.

That's the uncomfortable part. The leaked version isn't visibly broken. It just doesn't
work in production.

## [3:46] The stacking result nobody mentions

> **On screen —** the model leaderboard

One smaller finding while we're here.

The tournament includes a stacked ensemble — the thing that wins Kaggle competitions. It
came top: 0.4961.

Second place, a plain random forest: 0.4960.

That's a difference in the fourth decimal place. For a substantial increase in complexity,
training time, and the number of things that can break in deployment.

Stacking won. Stacking was also not worth it. Both of those are true, and leaderboards only
ever report the first one.

## [4:21] Close

> **On screen —** the Method tab

One question — would this value exist when I have to predict? — asked of every column,
catches a class of error that no amount of cross-validation will.

It cost nothing to ask. It was worth forty percent of a metric.

Next video: I build a transformer from tensor operations, train it on a CPU, and evaluate
it against baselines that most language-model demos never show you.
---

*Generated by [`tools/video_scripts.py`](../../tools/video_scripts.py) from committed
artifacts. Every figure spoken here is the figure on screen.*
