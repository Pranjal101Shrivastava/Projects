# I wrote Apriori and FP-Growth from scratch so they could check each other

*189× apart in runtime, byte-identical in output*

Most association-rule tutorials call `mlxtend.apriori`, print the top ten rules by lift, and
stop. I wanted two things that approach cannot give you: an understanding of *why*
FP-Growth is faster, and any reason to believe the rules are real.

So I implemented both algorithms by hand.

## The correctness check that fell out for free

Running two independent implementations on the same data gives you something a single
library call never does: if they disagree, at least one is wrong.

They did not disagree. All **13,106 frequent itemsets**, identical.

The pipeline now raises an exception and refuses to write results if they ever diverge.
That is a much stronger statement than "the code ran without error".

## Where the 189× comes from

| | Apriori | FP-Growth |
|---|---:|---:|
| Time | 117.943s | 0.624s |
| Database passes | 4 | 2 |
| Candidates generated | 267,573 | **zero** |

Apriori works breadth-first. At each level it generates candidate itemsets, prunes those
containing an infrequent subset, then scans the entire database to count the survivors. It
generated 267,573 candidates and pruned
72% of them before counting — the pruning works, and it
is still doing enormous work.

FP-Growth reads the database twice, total. Once to count item frequencies, once to build a
prefix tree where common prefixes are shared. Then it mines that tree recursively, in memory,
generating no candidates at all.

## The part that changed the results

Here is the thing almost nobody does.

I generated **52,285 candidate rules**. Then I asked: how many of these
would look this good by chance?

With 169 item categories there are roughly 28,000 possible pairs. Test
28,000 hypotheses at p < 0.05 and you expect ~1,400 false positives *even if nothing is
associated with anything*.

So every rule got a Fisher exact test and a Benjamini-Hochberg corrected q-value.

**5,144 rules failed** and are excluded from every table I publish.

I chose BH over Bonferroni deliberately. Bonferroni would control the probability of *any*
false positive, which at this scale means rejecting real affinities too. BH controls the
expected *proportion* of false discoveries in the list — the right guarantee when the output
is a shortlist a merchandiser will work through.

## What survived

| Rule | Lift |
|---|---:|
| {liquor} → {bottled beer, red/blush wine} | 35.7 |
| {bottled beer, red/blush wine} → {liquor} | 35.7 |
| {Instant food products} → {hamburger meat, soda} | 26.2 |
| {hamburger meat, soda} → {Instant food products} | 26.2 |
| {processed cheese} → {ham, white bread} | 22.9 |
| {ham, white bread} → {processed cheese} | 22.9 |

An alcohol basket. A convenience-meal basket. A sandwich basket — `processed cheese` →
`{ham, white bread}` at lift 22.9
if you want the clearest example of a rule you could act on tomorrow.

## And the rules that run the other way

Lift below 1 means two items appear together *less* often than chance predicts. But lift
squashes all negative association into the range [0, 1), so everything looks similar.

Zhang's metric spans [−1, 1] and separates them:

| Rule | Zhang |
|---|---:|
| {canned beer} → {whole milk} | -0.63 |
| {UHT-milk} → {whole milk} | -0.61 |
| {canned beer} → {root vegetables} | -0.55 |
| {canned beer} → {yogurt} | -0.54 |

Canned beer actively repels whole milk, root vegetables and yogurt. Those are different
shopping trips — the beer run and the weekly shop — and a merchandiser who knows that will
not waste an end-cap trying to bridge them.

---

**Live demo:** [https://pranjal101shrivastava.github.io/Projects/#/p/market-basket](https://pranjal101shrivastava.github.io/Projects/#/p/market-basket)
**Code:** [github.com/Pranjal101Shrivastava/Projects](https://github.com/Pranjal101Shrivastava/Projects)
