# Video 01 · NYC Ride-Hail Demand

**Estimated runtime:** ~6:41 at a measured pace (904 spoken words)
**Live page:** https://pranjal101shrivastava.github.io/Projects/#/p/nyc-mobility
**Project directory:** [`projects/01_nyc_mobility/`](../../projects/01_nyc_mobility/)

> **The one sentence this video has to land:** A 39.1% improvement over a zero-parameter baseline is the real result — and the prediction intervals failed at one of three levels, which is published rather than hidden.

## Before you record

- Open **https://pranjal101shrivastava.github.io/Projects/#/p/nyc-mobility** and let the charts finish loading.
- Have the **Findings**, **Method (CRISP-DM)** and **Data provenance** tabs ready to switch between.
- Optionally open `lib/dsx/splits.py` in an editor for the embargo section.

Everything in plain text below is meant to be **spoken**. Lines marked *On screen* and
*Note to self* are directions — do not read them aloud.

---

## [0:00] Cold open

> **On screen —** the project page, scrolled to the headline numbers

Here's a number I could have put at the top of this project and stopped.

Mean absolute error: 15.65. R-squared of 0.957.

On its own, that number is meaningless. And I want to show you why, because it's the
thing that separates a model that works from a model that just looks like it works.

## [0:25] The baseline that makes the number mean something

> **On screen —** the model comparison table, with the seasonal-naive row visible

Here's the comparison that matters.

The simplest possible forecast for this problem is: whatever happened at this hour last
week, that's my prediction. Same hour, same day of the week, seven days ago. Zero
parameters. No training. You can do it in one line.

That baseline gets a mean absolute error of 25.70.

My gradient boosted model gets 15.65.

So the model is better — but now you can see *how much* better. It's a
39.1% improvement over doing almost nothing. That's a
real result, and it's a modest one. If I'd only shown you the R-squared, you'd have
assumed something far more impressive.

Every metric on this site is reported next to its no-skill baseline, for exactly this
reason. A score without a reference point isn't a result. It's a number.

## [1:24] The data

> **On screen —** the Data provenance tab

The data is real. This is 4,534,327 actual taxi and ride-hail dispatch records
from New York City, 2014-04-01 through 2014-09-30, released by
the Taxi and Limousine Commission under a freedom of information request.

It's downloaded on first run and pinned by a SHA-256 hash, so if the file upstream ever
changes underneath this project, it fails loudly instead of quietly shifting every number
downstream.

I'm saying that explicitly because the portfolio this work responds to labels its datasets
as coming from Kaggle, and then generates all of them with a random number generator. That
comparison is documented in the repository.

## [2:10] One judgement call worth showing

> **On screen —** the provenance panel, duplicate rationale expanded

Here's a decision I want to walk through, because the reasoning matters more than the
answer.

There are 82,581 exact duplicate rows in this data — about
1.8% of it. The reflex is to drop duplicates. It's almost
a habit.

I kept them. Here's why.

The timestamps are recorded to the minute, and the coordinates are rounded to about eleven
metres. So two genuinely different cars, dispatched from the same base, on the same block,
in the same minute, are *indistinguishable* in this schema. They look like a duplicate.
They aren't one.

And the giveaway is this: the duplicate rate rises with demand. It's highest at peak
hours. That's the signature of collision under coarse resolution — not of a data fault.

So dropping them would have deleted real demand, and it would have deleted it precisely at
the busiest hours, which is exactly when the forecast matters most.

That reasoning is recorded in the artifact, along with the alternative I rejected. Not
just what I chose — what I chose *against*.

## [3:26] Not shuffling the split

> **On screen —** the split summary, with the embargo line visible

This is a time series, so the split is chronological. I train on the earlier period and
test on the later one — 28 days held out at the end.

But there's a subtler problem, and this is the part most tutorials skip.

My features include a lag of 168 hours — demand at this hour one week
ago. Which means a training row sitting right at the boundary shares lag history with the
first test rows. The partitions aren't actually independent.

So I discard a band of 168 hours at the boundary. That's called an
embargo. 40,596 training rows, 8,064 test rows, and a gap
between them that belongs to neither.

And this isn't something I have to remember to do. The split function in the shared library
refuses to shuffle a temporal split, and it records the embargo in the artifact. The
defence is structural, not a habit.

## [4:32] The result I did not want

> **On screen —** the conformal coverage table, 80% row highlighted

Now the part I nearly didn't publish.

A point forecast isn't much use for staffing. You want a range. So I added conformal
prediction intervals — a method that's supposed to give you a distribution-free coverage
guarantee. Ask for 80% coverage, get
80% coverage.

At the 90% and 95% levels, it held. Close enough to nominal to
call it working.

At 80%, it did not. I asked for 80%. I got
71.2%. That's nearly nine points short.

I could have reported only the two levels that worked. The table would have looked clean.

Instead, here's the diagnosis. Conformal prediction assumes exchangeability — that the
residuals I calibrate on look like the residuals I'll see later. Demand grew substantially
across this window. So the calibration residuals come from a quieter period than the test
residuals. The wide intervals absorb that drift. The tight one can't.

That's a real limitation of the method under trend, it's visible in the data, and it's the
kind of thing you only find if you actually check coverage instead of assuming the
guarantee holds.

## [5:51] Close

> **On screen —** the Method tab, showing recorded decisions with rejected alternatives

So: a model that beats a real baseline by a modest, honest margin. A leakage defence that's
enforced in code rather than remembered. And an uncertainty estimate that failed at one of
three levels, published with the reason it failed.

Every number I just said is read at runtime from a JSON file written by the pipeline. None
of them were typed by hand. If the prose and the artifact ever disagree, the artifact is
right and the prose is a bug.

Code and full write-up are in the repository. Next video, we cluster some customers — and
find out that the algorithms don't agree with each other nearly as much as you'd hope.
---

*Generated by [`tools/video_scripts.py`](../../tools/video_scripts.py) from committed
artifacts. Every figure spoken here is the figure on screen.*
