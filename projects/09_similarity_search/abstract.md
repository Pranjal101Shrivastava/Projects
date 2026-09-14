# Abstract — Measuring What Locality-Sensitive Hashing Costs

**Objective.** Quantify the accuracy price of approximate near-duplicate detection by
computing the exact answer alongside it, rather than reporting a speedup in isolation.

**Data.** 1,534 distinct company name strings extracted from
28,156 US Consumer Financial Protection Bureau complaint records, yielding
1,175,811 candidate pairs. Names were normalised and decomposed into character
3-shingles (mean 21.91 per name); corporate
suffixes were deliberately retained.

**Method.** MinHash with k = 128 permutations and banded LSH were
implemented from first principles. Exhaustive pairwise Jaccard computation provided ground
truth at a threshold of 0.5. 5 band/row configurations were evaluated,
each with its candidate set verified exactly so that reported speedups include verification
cost. Estimator accuracy was assessed on 3,994 sampled pairs, stratified
by true similarity.

**Results.** Exhaustive search required 2.682s for 1,175,811 pairs and
identified 1,659 pairs above threshold. Full recall was achievable at
3.0× (64 bands × 2
rows, 144,280 candidates); relaxing to
90.0% recall yielded 45.2×.
MinHash estimates were unbiased (mean error +0.00240); within-band observed
standard deviations tracked the theoretical √(s(1−s)/k) to within
0.0172 across all bands.

**Conclusion.** Speedup figures for approximate search are uninterpretable without the recall
they purchased. The aggregate error of a MinHash estimator is also uninterpretable, because
89.8% of random pairs fall in the region where its variance
vanishes; stratification by true similarity is required.

**Keywords.** MinHash, locality-sensitive hashing, entity resolution, Jaccard similarity,
approximate nearest neighbour
