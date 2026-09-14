# Video 03 · Market Basket Mining

**Estimated runtime:** ~6:55 at a measured pace (934 spoken words)
**Live page:** https://pranjal101shrivastava.github.io/Projects/#/p/market-basket
**Project directory:** [`projects/03_market_basket/`](../../projects/03_market_basket/)

> **The one sentence this video has to land:** Two from-scratch implementations agree byte-for-byte on 13,106 itemsets at a 189x speed difference — and 5,144 mined rules were thrown out for failing FDR control.

## Before you record

- Open **https://pranjal101shrivastava.github.io/Projects/#/p/market-basket**.
- Have `projects/03_market_basket/pipeline/build.py` open at the two algorithm implementations.
- Locate the dissociations table in advance — it's below the top-rules table.

Everything in plain text below is meant to be **spoken**. Lines marked *On screen* and
*Note to self* are directions — do not read them aloud.

---

## [0:00] Cold open

> **On screen —** the algorithm comparison table

Two algorithms. Same data. Same output — byte for byte identical, all
13,106 results.

One took 117.9 seconds. The other took
0.62.

That's 189 times faster for exactly the same answer. In
this video I'll show you where that factor comes from — and then I'll show you the part
almost nobody does, which is throwing away about ten percent of the results on statistical
grounds.

## [0:29] The data and the problem

> **On screen —** the Data provenance tab

9,835 real grocery baskets across 169 product
categories. This is the classic groceries dataset from the R `arules` package — genuine
point-of-sale transactions.

The question is: which products actually pull each other into the basket? Because if you
get that right you can place them together. And if you get it wrong, you rearrange an aisle
around a pattern that was noise, which costs real money and is very hard to detect
afterwards.

## [1:02] Why I wrote both algorithms by hand

> **On screen —** the pipeline source, the Apriori and FP-Growth functions

I implemented both from first principles instead of calling a library, and I want to be
clear about why — it's not for the sake of it.

Two reasons.

First, if you call `mlxtend.apriori`, the difference between the two algorithms is hidden
inside a function, and the runtime comparison means nothing.

Second — and this is the real one — running two independent implementations on the same
data gives you something a single library call never does. **If they disagree, at least one
of them is wrong.**

They didn't disagree. All 13,106 frequent itemsets, identical.

And the pipeline now raises an exception and refuses to write any results if they ever
diverge. That's a much stronger correctness statement than "the code ran without an error."

## [1:57] Where the speedup comes from

> **On screen —** the per-level candidate counts for Apriori

So where does 189x come from?

Apriori works level by level. It generates candidate itemsets of size one, counts them,
uses the survivors to build candidates of size two, counts those, and so on. Each level
means another full pass over the database.

It has a genuinely clever pruning rule — if any subset of a candidate is infrequent, the
candidate can't be frequent, so you can discard it before counting. And that rule works: it
eliminated 72.0% of candidates before counting.

But look at the raw volume. It still generated 267,573
candidates across 4 database scans.

FP-Growth reads the database exactly 2 times. Once to
count how often each item appears, once to build a prefix tree where common beginnings are
shared. Then it mines that tree recursively, entirely in memory.

Candidates generated: **zero**. None. It never proposes an itemset and then checks it. It
reads them off the structure.

121,317 tree nodes, and it's done.

## [3:08] The part that changed the results

> **On screen —** the FDR summary

Now here's the step that actually changed what I published.

Mining generated 52,285 candidate rules. The normal thing to do is sort
by lift, take the top ten, and put them on a slide.

Think about what that is statistically. With 169 items there are
tens of thousands of possible pairs. Test tens of thousands of hypotheses at p below
point-oh-five, and you expect *thousands* of false positives — even if nothing in the data
is associated with anything.

So every single rule got a Fisher exact test, and then a Benjamini-Hochberg correction for
multiple comparisons.

**5,144 rules failed.** They're excluded from every table I publish.
47,141 survived.

I chose Benjamini-Hochberg over Bonferroni deliberately. Bonferroni controls the chance of
*any* false positive, and at this scale it would reject genuine affinities along with the
noise. Benjamini-Hochberg controls the expected *proportion* of false discoveries in the
list — which is the right guarantee when the output is a shortlist a human is going to work
through.

And because "I implemented a statistical correction" is a claim you should be suspicious
of, I checked mine against `statsmodels`. Maximum difference in q-values: about one times
ten to the minus sixteen. Identical rejection sets.

## [4:37] What survived

> **On screen —** the top rules table, then scroll to the dissociations

Here's the strongest surviving affinity: lift of 35.7. And the nice thing is
it's interpretable — you can look at it and see why those products go together.

Now scroll down, because this is my favourite part.

These are *dissociations*. Pairs bought together **less** often than chance predicts.

The problem is that lift squashes all negative association into the range zero to one, so
everything down there looks about the same. Zhang's metric spans minus one to plus one and
separates them properly.

The strongest one scores -0.63 — canned beer actively repelling whole milk.
Those are two different shopping trips, the beer run and the big shop, and a merchandiser
who knows that won't waste an end-cap trying to bridge them.

One caveat I want to state out loud, because it's easy to miss. Look at the q-value column
on these rows: it reads 1.0. That is not a mistake, and it is not evidence
*for* these rules. My significance test is a one-sided Fisher exact test — it asks whether
two items co-occur *more* than chance. A pair that co-occurs *less* than chance will always
score near one on that test, by construction.

So the dissociations are ranked by Zhang's metric, and the FDR machinery says nothing about
them either way. Certifying depletion would need the test pointed the other way. I haven't
done that, so I'm not going to claim these are statistically established — only that the
effect size is large and the interpretation is plausible.

## [6:28] Close

> **On screen —** the project README

Two implementations that check each other. A speedup you can trace to a specific
algorithmic idea. And a multiple-comparisons correction that removed thousands of my own
rules before I published the rest.

Next video: fraud detection, where the positive class is under two-tenths of one percent,
and where the standard textbook advice for imbalanced data made my model eighty times
worse.
---

*Generated by [`tools/video_scripts.py`](../../tools/video_scripts.py) from committed
artifacts. Every figure spoken here is the figure on screen.*
