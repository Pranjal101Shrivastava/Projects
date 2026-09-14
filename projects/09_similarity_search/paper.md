# Measuring the Cost of Approximation in Locality-Sensitive Hashing

*Entity resolution over 1,534 real company names*

## 1. Problem

Deduplicating entity names is quadratic in the number of entities. At
1,534 names that is 1,175,811 comparisons — tractable,
but a hundred thousand names would not be. LSH is the standard answer, and it is an
*approximation*: it trades recall for time. The question this study asks is how much.

## 2. Data

28,156 CFPB consumer complaint records across 11 products and
89 issue categories yield 1,534 distinct company
name strings, mean length 25.4 characters (median
26, range 4–60).

Lowercased, punctuation replaced by spaces, whitespace collapsed. Corporate suffixes (LLC, Inc, Corporation) are deliberately NOT stripped: removing them would make matching artificially easy and obscure the question of whether the approximation finds what exact search finds.

## 3. Method

### 3.1 Shingling

Each name is decomposed into overlapping character 3-grams — mean
21.91 per name, range 2–51.
Jaccard similarity is computed over these shingle sets.

### 3.2 MinHash

k = 128 independent hash permutations. For each, the minimum hash value over
a set's shingles is retained; the fraction of agreeing minima across permutations is an
unbiased estimator of Jaccard similarity. Signature construction over all
1,534 names took 0.073s.

The estimator's standard deviation is √(s(1−s)/k), maximal at s = 0.5 where it equals
0.5/√k = 0.0442.

### 3.3 Banded LSH

The signature is split into b bands of r rows (b·r = k). Two items become candidates if any
band matches exactly. The probability of candidacy is

    P(s) = 1 − (1 − sʳ)ᵇ

a sigmoid with its steep region near (1/b)^(1/r).

### 3.4 Ground truth

All 1,175,811 pairs were compared exactly (2.682s), giving
1,659 pairs at Jaccard ≥ 0.5. Every LSH configuration is scored
against this set.

## 4. Results

| Bands | Rows | Threshold | Candidates | Recall | Missed | Banding | Verify | Total | Speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 16 | 0.878 | 14 | 0.8% | 1,645 | 0.007s | 0.0001s | 0.007s | 359.7× |
| 16 | 8 | 0.707 | 335 | 16.8% | 1,380 | 0.012s | 0.0016s | 0.014s | 191.1× |
| 26 | 4 | 0.443 | 8,966 | 90.0% | 166 | 0.023s | 0.0363s | 0.059s | 45.2× |
| 32 | 4 | 0.420 | 10,260 | 93.5% | 107 | 0.235s | 0.0440s | 0.280s | 9.6× |
| 64 | 2 | 0.125 | 144,280 | 100.0% | 0 | 0.337s | 0.5626s | 0.900s | 3.0× |

Precision is 1.000 throughout because candidates are verified exactly. Verification time is
included in the total; excluding it, as is common, would misstate the trade.

### 4.1 Estimator accuracy, stratified

| Band | Pairs | Mean *s* | Observed sd | Theory | Mean error |
|---|---:|---:|---:|---:|---:|
| `[0.00, 0.01)` | 2,105 | 0.0000 | 0.0000 | 0.0000 | +0.00000 |
| `[0.01, 0.10)` | 1,483 | 0.0434 | 0.0174 | 0.0180 | +0.00513 |
| `[0.10, 0.30)` | 364 | 0.1747 | 0.0307 | 0.0336 | +0.00509 |
| `[0.30, 0.50)` | 36 | 0.3717 | 0.0398 | 0.0427 | +0.00223 |
| `[0.50, 1.01)` | 6 | 0.5747 | 0.0265 | 0.0437 | +0.00602 |

Observed spread tracks √(s(1−s)/k) within every band. The aggregate figure
(0.0149) is not comparable to the peak theoretical value because
89.8% of sampled pairs lie below s = 0.1.

## 5. Discussion

The configuration maximising F1 is 64 bands × 2 rows, achieving
100.0% recall at 3.0×. It is not obviously the right
operating point: 26 × 4 gives
45.2× for 166 missed
pairs. F1 weights a missed pair and a wasted comparison equally, which no real application
does.

The degenerate configurations are instructive: 8 × 16 reaches
360× while recovering 0.8% of true pairs. A
speedup reported without recall can be made arbitrarily large.

## 6. Limitations

- Ground truth is 'Jaccard above a threshold', not 'genuinely the same company'. String similarity is a proxy: two distinct subsidiaries of one group can score highly, and a company that rebranded entirely will score low. The recall figures measure agreement with exact string search, not with reality.
- No configuration reaches 100% recall. The best here misses 0 of 1659 true pairs. That is not a defect to be tuned away — it is the guarantee LSH offers: probabilistic, not exhaustive. Any application where a missed pair is unacceptable needs exact search or a different method.
- Speedup is measured at n=1,534, where exact search is already fast. The asymptotic argument is what matters: exact search grows as O(n²) while LSH grows roughly linearly, so the advantage widens with scale and these figures understate it.
- Both stages run in one process on one core. A distributed implementation would change the constants substantially.

*Generated at commit `200f444` · seed 42 · 4.4s · Python 3.11.15 · numpy 2.4.6 · pandas 3.0.5 · sklearn 1.9.1 · lightgbm 4.7.0 · torch 2.14.0+cu130*
