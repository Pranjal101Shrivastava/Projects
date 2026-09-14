# Your clustering is probably not reproducible, and you would never know

*What happens when you bootstrap a segmentation 30 times*

Here is the standard customer segmentation workflow. Load the data. Run K-means with k=4,
because four personas fit on a slide. Colour the scatter plot. Name the segments something
like "High-Value Loyalists" and "At-Risk Dormants". Ship it.

Every step of that is defensible except the part nobody checks: **would you get the same
segments from a different sample of the same customers?**

I ran that check on 41,188 real bank marketing contacts, and the answer reorganised the
whole project.

## The indices were unanimous, and unhelpful

First, the textbook approach. Sweep k from 2 to 10, score each with three internal validity
indices, take the winner.

All three agreed: **k = 2**.

Silhouette, Calinski-Harabasz, Davies-Bouldin — unanimous. In most write-ups that
unanimity would be reported as strong evidence and the analysis would proceed with two
segments.

## Then I resampled

For each k, I took 30 bootstrap resamples, clustered each independently,
and measured how much each resample's partition agreed with the original using adjusted Rand
index. ARI of 1.0 means identical; 0 means no better than chance.

Only **k ∈ {6, 7}** cleared 0.75.

The unanimous geometric winner was not among them.

The two solutions optimise different things. Internal indices ask "are these clusters
compact and well-separated?" — a question about geometry. Bootstrap stability asks "would I
find these clusters again?" — a question about whether the finding is real. A two-way split
is geometrically clean and tells a marketing team nothing they did not already know.

I went with reproducibility: **k = 6**.

## The segments earned their keep

The clustering never saw whether anyone subscribed. I held the outcome out entirely, then
checked afterwards.

| Segment | Share | Conversion | Defining trait |
|---|---:|---:|---|
| 1 | 3.7% | 63.83% | 100% previously contacted |
| 3 | 9.8% | 12.44% | 2.0 mean calls, cellular |
| 4 | 32.3% | 12.34% | 2.1 mean calls, cellular |
| 2 | 24.0% | 9.68% | 2.2 mean calls, cellular |
| 0 | 26.4% | 4.67% | 2.2 mean calls, telephone |
| 5 | 3.9% | 4.15% | 12.9 mean calls, telephone |

Conversion ranges from 4.15% to 63.83%.
χ² = 4786. The segments are real and they matter.

## The finding worth acting on is the bad segment

Everyone's eye goes to segment 1 at 63.83%. But it
is 100% previously-contacted customers — the model has
rediscovered "people who already said yes once are likely to say yes again". True, useful,
unsurprising.

Segment 5 is the one that should change behaviour tomorrow.
1,592 customers. **12.9 calls each on average.**
Converting at 4.15% — a third of baseline.

That is a cohort absorbing an enormous amount of calling effort and returning almost nothing.
Nobody had to build a model to stop calling them; somebody had to notice.

## What I am still not claiming

K-means agrees with a Gaussian mixture at ARI **0.4924** and with
hierarchical clustering at **0.5046**.

That is moderate. It means roughly half of what looks like structure is K-means's assumption
that clusters are spherical blobs of similar size. Change the algorithm and half the
boundaries move.

So these are *a* defensible partition, not *the* natural segments of this population. That
distinction rarely survives into a segmentation deck, and it should.

---

**Live demo:** [https://pranjal101shrivastava.github.io/Projects/#/p/segmentation](https://pranjal101shrivastava.github.io/Projects/#/p/segmentation)
**Code:** [github.com/Pranjal101Shrivastava/Projects](https://github.com/Pranjal101Shrivastava/Projects)
