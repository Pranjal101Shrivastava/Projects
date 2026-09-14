# Video 02 · Customer Segmentation

**Estimated runtime:** ~4:40 at a measured pace (630 spoken words)
**Live page:** https://pranjal101shrivastava.github.io/Projects/#/p/segmentation
**Project directory:** [`projects/02_customer_segmentation/`](../../projects/02_customer_segmentation/)

> **The one sentence this video has to land:** The segments separate an outcome the algorithm never saw (4.15% to 63.83% conversion) — but three algorithms only agree at ARI 0.49, so half the structure is the method's assumption.

## Before you record

- Open **https://pranjal101shrivastava.github.io/Projects/#/p/segmentation**.
- Have the conversion chart, the stability sweep and the algorithm-agreement panel located in advance — you'll move between them quickly.

Everything in plain text below is meant to be **spoken**. Lines marked *On screen* and
*Note to self* are directions — do not read them aloud.

---

## [0:00] Cold open

> **On screen —** the segment conversion chart

Clustering is the easiest place in data science to fool yourself.

You run k-means, you get six groups, you give them names like "high value" and "at risk",
you put them on a slide, and nobody can tell you you're wrong — because there's no right
answer to check against.

So in this project I tried to make it possible to be wrong. Three different ways.

## [0:28] The data

> **On screen —** the Data provenance tab

41,188 real contact records from a Portuguese bank's direct marketing
campaigns, from the UCI machine learning repository. Real customers, real phone calls, real
outcomes.

The key thing: each record also carries whether that person actually subscribed to the
product. I deliberately kept that column *out* of the clustering. The algorithm never sees
it.

Which means I have something most clustering projects don't — an answer key the method
wasn't allowed to look at.

## [1:01] Test one: does it predict something it never saw?

> **On screen —** the conversion-by-segment table

So here's the first real test. If these segments are picking up genuine structure, they
should differ on the outcome the algorithm was blind to.

They do, and the spread is large.

The best segment converts at 63.83%. The worst converts at
4.15%. Overall conversion across everyone is
11.27%.

Chi-squared is 4,786, with a p-value far below any threshold you'd care about.

That's the segmentation earning its keep. Not because the clusters look tidy on a scatter
plot — because they separate an outcome nobody told the algorithm about.

## [1:42] Test two: would I get the same answer twice?

> **On screen —** the stability sweep chart

Second test, and this is the one that changed my answer.

I resampled the data 30 times, re-clustered each resample, and measured
how much the groupings agreed with each other using the adjusted Rand index. If a clustering
falls apart when you shake the data slightly, it was never real structure — it was noise
with a label on it.

Now, here's what makes this interesting.

The standard internal validity indices — silhouette, Calinski-Harabasz, Davies-Bouldin —
all voted for 2 clusters. All three of them. Unanimously.

Stability said something different, and I went with stability. I chose
k equals 6.

Because 2 clusters is a geometrically tidy answer that tells you almost nothing
about a business. And the indices are measuring compactness — not reproducibility, and
certainly not usefulness.

That's a judgement call. It's recorded as one, with the alternative I rejected written down
beside it.

## [2:45] Test three — and the result I did not like

> **On screen —** the cross-algorithm agreement figures

Third test. If the structure is really in the data, then algorithms with different
assumptions should find roughly the same thing.

K-means assumes clusters are spherical and about the same size. A Gaussian mixture allows
stretched, tilted, overlapping clusters. Ward linkage builds a hierarchy by minimising
variance increase. Three genuinely different ideas about what a cluster *is*.

I ran all three and compared them.

K-means versus the Gaussian mixture: adjusted Rand index 0.49.
K-means versus Ward linkage: 0.50.

Those are moderate numbers. Not bad — but not the agreement you'd want before telling
someone these are natural groups.

Here's how I read that honestly. Roughly half the structure I'm seeing is in the data.
The other half is the spherical assumption I imposed when I picked k-means.

So the correct claim is not "here are the six customer types." The correct claim is: this
is one defensible partition, it reproduces under resampling, and it separates an outcome it
never saw. That's a useful thing. It is not a discovery of natural kinds, and I'm not going
to describe it as one.

## [4:06] Close

> **On screen —** the Method tab, decisions with rejected alternatives

Three tests: external validation, stability under resampling, and agreement across
algorithms. Two of them passed convincingly. The third came back moderate, and that's
reported as a limitation rather than left out.

The honest version of a clustering project includes the part where the algorithms disagree.

Next video: I implement two frequent-itemset mining algorithms from scratch, find a
189-times speed difference between them, and then throw away about ten percent of my own
results on statistical grounds.
---

*Generated by [`tools/video_scripts.py`](../../tools/video_scripts.py) from committed
artifacts. Every figure spoken here is the figure on screen.*
