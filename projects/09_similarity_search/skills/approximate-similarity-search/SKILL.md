---
name: approximate-similarity-search
description: Deploy LSH for near-duplicate detection and measure what the approximation costs
---

# Deploy LSH for near-duplicate detection and measure what the approximation costs

*Derived from [`projects/09_similarity_search`](../../), which applied this procedure to: US Consumer Financial Complaints.*

This is a **transferable procedure**, not instructions for rerunning the script in this
repository. It states what to do on a new dataset, where the judgement calls are, and which
failure modes this project actually hit.

## When to use this

You need to find near-duplicates — customer records, company names, documents, product
listings — and the number of pairs has made exhaustive comparison unattractive.

## Procedure

### 1. Compute the exact answer while you still can

This is the step almost every LSH write-up skips, and it is the one that makes the rest
meaningful. Below roughly a million pairs, exhaustive comparison is seconds of work. Do it,
save the result, and score every approximation against it.

If your data is already too large for this, **subsample it until it is not**. A 5,000-item
sample gives you a ground-truth recall curve that transfers to the full set far better than
no curve at all.

### 2. Normalise deliberately, and record what you chose not to do

Lowercasing and punctuation stripping are safe. Removing corporate suffixes (`Inc`, `LLC`,
`GmbH`) is *not* — it makes matching look better by deleting the exact tokens that
distinguish two legally separate entities. Whatever you decide, record it beside the results:
normalisation choices move recall more than parameter tuning does.

### 3. Shingle, then estimate

Character k-grams (k = 3 for short strings like names, larger for documents) turn strings
into sets so Jaccard applies. MinHash with k permutations estimates Jaccard in O(k) per item
instead of O(|set|) per pair.

### 4. Choose bands and rows as a threshold decision, not a tuning exercise

P(candidate) = 1 − (1 − sʳ)ᵇ, with the step near (1/b)^(1/r). Pick where you want the step —
that is the whole design. Sweep several (b, r) pairs and report the curve, not one point.

### 5. Include verification time in every speedup you report

LSH proposes; exact comparison disposes. A speedup computed from banding time alone is
measuring half the pipeline, and the half it omits grows with recall.

### 6. Report accuracy stratified by true similarity

The estimator's standard deviation is √(s(1−s)/k), maximal at s = 0.5 and near zero at the
extremes. Since most random pairs are dissimilar, any aggregate error figure is dominated by
the easy cases and looks far better than the estimator actually is on the pairs you care
about.

## Decision points requiring judgement

**What recall do you actually need?** Full recall typically costs most of the speed
advantage. Whether that matters depends on whether a missed pair is an inconvenience or a
compliance failure — and no benchmark can answer that.

**F1 is probably the wrong selector.** It weights a missed pair and a wasted comparison
equally. Almost no application does.

## Failure modes this project hit

**Comparing observed spread against 1/√k.** That rule of thumb is *twice* the true maximum of
0.5/√k. Combined with an unstratified aggregate, it made the estimator appear to beat its own
theoretical variance — an impossible result, and the signal that the theory line was wrong
rather than the measurement. If your estimator looks better than theory, you have the theory
wrong.

**Ground truth that is not ground truth.** "Jaccard above a threshold" is not "the same
entity". Recall figures measure agreement with exact string search, not with reality, and
saying so is part of reporting them.

---

## Reproduce the reference implementation

```bash
export PYTHONPATH=lib
python3 projects/09_similarity_search/pipeline/build.py
python3 tools/audit.py
```

Results, with their run provenance, land in
[`projects/09_similarity_search/artifacts/`](../../artifacts/).
