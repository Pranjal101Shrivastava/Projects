# I built LSH from scratch, then computed the exact answer to see what it cost me

*Every LSH tutorial reports a speedup. Almost none report the recall.*

Locality-sensitive hashing is the standard trick for finding near-duplicates without
comparing everything to everything. The pitch writes itself: 1,175,811 pairs
becomes 144,280 candidates, done.

But LSH is an **approximation**. It will miss pairs. A speedup without a recall figure is
half of a sentence.

So I computed the exact answer too.

## The setup

1,534 distinct company names, pulled from 28,156
real consumer-finance complaints. Names like `United Collection Bureau` and
`United Collection Bureau, Inc.` — the same company, two strings, and the kind of thing that
quietly splits a report in half.

Exhaustive comparison: **1,175,811 pairs in 2.682s**, finding
1,659 matches at Jaccard ≥ 0.5. That is my ground truth.

## What LSH actually cost

| Bands × rows | Recall | Missed | Speedup |
|---|---:|---:|---:|
| 8 × 16 | **0.8%** | 1,645 | 359.7× |
| 16 × 8 | **16.8%** | 1,380 | 191.1× |
| 26 × 4 | **90.0%** | 166 | 45.2× |
| 32 × 4 | **93.5%** | 107 | 9.6× |
| 64 × 2 | **100.0%** | 0 | 3.0× |

Read the top row again. **360× faster** — and it found
0.8% of the duplicates. That is the number I could have published if I had
never computed ground truth, and it is worthless.

At the other end, perfect recall costs almost everything: 144,280
candidates to examine, 3.0× left over.

The honest middle is 26 × 4:
45.2× for 90.0% recall. Whether
that is a good deal depends on whether the 166 missed
pairs are an annoyance or a regulatory problem. No benchmark can answer that for you.

## The mistake I made, and kept

I first reported MinHash's accuracy like this: observed standard deviation
0.0149, theoretical 0.0884. Look how
much better than theory my implementation is!

An estimator cannot beat its own variance. That is not a good result, it is a bug — and it was
two bugs.

**First**, MinHash's standard deviation is not a constant. It is √(s(1−s)/k), which is nearly
zero for dissimilar pairs. And 89.8% of randomly drawn pairs
of company names are dissimilar. My "impressive" aggregate was mostly measuring pairs that are
trivially easy.

**Second**, the figure I was comparing against — 1/√k — is the rule of thumb everyone quotes,
and it is **twice** the true maximum of 0.5/√k = 0.0442.

Stratified by true similarity, the picture is boring and correct:

| Similarity | Observed sd | Theory |
|---|---:|---:|
| [0.00, 0.01) | 0.0000 | 0.0000 |
| [0.01, 0.10) | 0.0174 | 0.0180 |
| [0.10, 0.30) | 0.0307 | 0.0336 |
| [0.30, 0.50) | 0.0398 | 0.0427 |

The estimator behaves exactly as advertised. Mean error +0.00240, unbiased.
That is the result — and it took getting it wrong twice to state it properly.

## What it found

| Name A | Name B | Jaccard |
|---|---|---:|
| CCS Financial Services, Inc. | MAS Financial Services, Inc. | 0.85 |
| United Collection Bureau | United Collection Bureau, Inc. | 0.85 |
| Receivable Management Group, Inc. | Receivables Management Group, Inc. | 0.84 |
| C&F Mortgage Corporation | GSF Mortgage Corporation | 0.83 |
| Financial Asset Management, Inc. | First Financial Asset Management, Inc. | 0.82 |
| CN Collections, Inc. | Collections Inc | 0.81 |

And the caveat that belongs beside them: ground truth here is *string similarity*, not
"genuinely the same company". Two subsidiaries of one group can score highly; a company that
rebranded scores low. Every recall number above measures agreement with exact string search,
not with reality.

---

**Live demo:** [https://pranjal101shrivastava.github.io/Projects/#/p/similarity-search](https://pranjal101shrivastava.github.io/Projects/#/p/similarity-search)
**Code:** [github.com/Pranjal101Shrivastava/Projects](https://github.com/Pranjal101Shrivastava/Projects)
