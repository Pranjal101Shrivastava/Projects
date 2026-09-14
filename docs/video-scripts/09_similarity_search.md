# Video 09 · Sub-Linear Similarity Search

**Estimated runtime:** ~6:19 at a measured pace (854 spoken words)
**Live page:** https://pranjal101shrivastava.github.io/Projects/#/p/similarity-search
**Project directory:** [`projects/09_similarity_search/`](../../projects/09_similarity_search/)

> **The one sentence this video has to land:** 100.0% recall costs almost all the speedup (3.0x versus 45x at 90.0%) — a speedup without a recall figure is half a sentence.

## Before you record

- Open **https://pranjal101shrivastava.github.io/Projects/#/p/similarity-search**.
- The S-curve chart is interactive — click through the configurations on camera.
- Have the stratified accuracy table ready for the mistake section.

Everything in plain text below is meant to be **spoken**. Lines marked *On screen* and
*Note to self* are directions — do not read them aloud.

---

## [0:00] Cold open

> **On screen —** the recall/speed trade-off table

Here's a result I could publish about this project:

**360 times faster** than exhaustive search.

It's true. It's also worthless, and I can prove it's worthless — because that configuration
found 0.8% of the duplicates it was looking for.

This video is about the number that every locality-sensitive hashing write-up reports, and
the number almost none of them report alongside it.

## [0:27] The problem and the data

> **On screen —** the Data provenance tab

The task is entity resolution. 1,534 distinct company name
strings, pulled from 28,156 real consumer finance complaints filed with the
US Consumer Financial Protection Bureau.

You get pairs like "United Collection Bureau" and "United Collection Bureau, Incorporated."
Same company. Two strings. And if you don't catch that, your report is quietly split in half.

Comparing everything to everything is quadratic. At this size that's
1,175,811 pairs — which is exactly the point.

## [1:00] Computing the exact answer on purpose

> **On screen —** the exact-search figures

Here's the decision that makes this project different.

1,175,811 pairs is large, but it is *not* too large. I ran it.
2.68 seconds. It found 1,659 pairs above a Jaccard similarity
of 0.5.

That's ground truth. Now every approximation can be scored against it.

Most write-ups skip this step, and the reason is understandable — the whole point of LSH is
to avoid the exhaustive computation. But if you never compute it, you can only report your
speedup. You cannot report what it cost you.

And if the data really is too big, subsample it. A few thousand items gives you a
ground-truth recall curve that transfers far better than no curve at all.

## [1:52] How it works, briefly

> **On screen —** the S-curve chart

Two ideas, quickly.

**MinHash.** Break each name into overlapping character
3-grams — so it's a set of short fragments, about
22 per name here. Then hash the set
128 different ways and keep the minimum each time. The fraction of those
minima that two names share is an unbiased estimate of their Jaccard similarity. You've
turned a set comparison into a fixed-length signature.

**Banding.** Split the signature into b bands of r rows. Two items become candidates if any
single band matches exactly.

The probability of becoming a candidate is one minus, one minus s to the r, all to the b.
It's an S-curve, and the steep part sits near one over b, to the power one over r.

Which means choosing bands and rows *is* choosing where that step falls. You're not tuning.
You're placing a threshold, deliberately.

## [2:54] What the approximation actually cost

> **On screen —** the full trade-off table

So here's the table, and this is the centre of the video.

At 8 bands by 16 rows:
360 times faster, 0.8% recall. It missed
1,645 of 1,659 true pairs. That's the headline
number I opened with, and now you can see what it bought.

At 26 by 4: 45.2 times faster,
90.0% recall, 166 missed.

At 64 by 2: **100.0% recall** — every pair — and
the speedup falls to 3.0 times. Because getting them all requires
examining 144,280 candidate pairs, and every one has to be verified.

That's not a tuning failure. That's the guarantee LSH offers. It's probabilistic, not
exhaustive, and if a missed pair is unacceptable in your application, you need exact search.

One detail on how I timed this: the speedups **include** verification time. LSH proposes
candidates, exact comparison disposes of them. Excluding the verification would have let me
report 8.0 times instead of
3.0 for that last row — and that's the kind of omission that
makes a benchmark meaningless.

## [4:11] Where I was wrong twice

> **On screen —** the stratified accuracy table

Now the mistake, because I think it's more instructive than the result.

My first version reported MinHash's accuracy like this: observed standard deviation
0.0149, theoretical value
0.0884. Look how much better than theory my implementation
is.

Stop. An estimator **cannot** beat its own variance. That's not a good result. It's a bug.

And it was two bugs stacked.

First: MinHash's standard deviation is not a constant. It's the square root of s times one
minus s, over k. It's near zero for dissimilar pairs. And
89.8% of randomly drawn company-name pairs are dissimilar.
So my impressive aggregate was mostly measuring pairs that are trivially easy.

Second: the figure I was comparing against — one over root k — is the rule of thumb everyone
quotes, and it is **twice** the true maximum, which is nought point five over root k, or
0.0442 here.

Stratified by true similarity, the picture is boring and correct. In each band, observed
spread tracks theory closely — 0.0000 against
0.0000, 0.0398 against
0.0427.

Mean error across all sampled pairs: +0.00240. Unbiased, as the theory
requires.

The general lesson: if your method appears to beat its own theoretical limit, you have the
theory wrong, not a breakthrough.

## [5:43] Close

> **On screen —** the top matches table and the caveat below it

And one limitation that belongs right next to the results.

Ground truth here is "string similarity above a threshold" — not "genuinely the same
company." Two subsidiaries of one group can score highly. A company that rebranded entirely
scores low. So every recall figure measures agreement with exact string search, not
agreement with reality.

Next video: the COMPAS criminal risk score, where I'll show you that ProPublica and the
vendor were both right, and prove why with algebra on the real numbers.
---

*Generated by [`tools/video_scripts.py`](../../tools/video_scripts.py) from committed
artifacts. Every figure spoken here is the figure on screen.*
