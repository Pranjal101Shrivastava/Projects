# Frequent Itemset Mining with Statistical Rule Validation

*A study on 9,835 real grocery transactions*

## 1. Problem

Cross-sell placement and bundle promotion depend on knowing which products genuinely pull
each other into the basket. The commercial risk is acting on a spurious pattern: rearranging
an aisle around a rule that was noise costs real money and is difficult to detect after the
fact.

## 2. Algorithms

### 2.1 Apriori

Apriori rests on the downward-closure property: if an itemset is frequent, every subset of it
is frequent. Contrapositively, any candidate containing an infrequent subset can be discarded
before counting it. The cost is one full database pass per level.

Measured on this data: 267,573 candidates generated across
4 levels, of which 72.0% were
eliminated by downward closure before counting.

### 2.2 FP-Growth

FP-Growth reads the database exactly twice — once to count item frequencies, once to build a
prefix tree ordered by descending frequency to maximise sharing — then mines entirely in
memory by recursively constructing conditional trees. No candidates are generated.

Measured: 121,317 tree nodes,
2 database scans.

### 2.3 Cross-validation of implementations

Both were run on identical input and their outputs compared exactly. They agree on all
13,106 frequent itemsets. The pipeline raises `AssertionError` on any
divergence rather than publishing results from an algorithm that disagrees with itself.

| Metric | Apriori | FP-Growth | Ratio |
|---|---:|---:|---:|
| Seconds | 117.943 | 0.624 | 189× |
| Database scans | 4 | 2 | — |
| Candidates | 267,573 | 0 | — |

## 3. Rule scoring

Six measures were computed for every rule, because each fails in a different way.

Let X be the antecedent and Y the consequent.

- Support: P(X ∧ Y)
- Confidence: P(Y|X) — ignores the marginal frequency of Y
- Lift: P(Y|X)/P(Y) — corrects for that, but is symmetric
- Conviction: P(X)P(¬Y)/P(X ∧ ¬Y) — directional
- Leverage: P(X ∧ Y) − P(X)P(Y) — absolute excess co-occurrence
- Zhang's metric ∈ [−1, 1] — distinguishes association from dissociation

## 4. Statistical validation

52,285 rules constitutes a large simultaneous-testing problem. Each rule's
2×2 contingency table was subjected to a one-sided Fisher exact test, and p-values corrected
by the Benjamini-Hochberg step-up procedure at α = 0.05.

52,285 candidate rules were tested. 5,144 fail FDR control at α=0.05 and are excluded from the published shortlist. Reporting only the survivors — and stating how many did not survive — prevents the top-by-lift table from being read as though every rule in it were real.

Bonferroni was considered and rejected: controlling family-wise error across
52,285 tests would reject genuine affinities along with the noise. BH
bounds the expected proportion of false discoveries among reported rules, which matches how
the output is used — as a ranked shortlist for human judgement.

The BH implementation was verified against `statsmodels.multipletests`: maximum absolute
q-value difference 1.1 × 10⁻¹⁶ and identical rejection sets on a 1,000-hypothesis test case.

## 5. Results

| Rule | Baskets | Conf | Lift | Leverage | Zhang | q |
|---|---:|---:|---:|---:|---:|---:|
| `{liquor}` → `{bottled beer, red/blush wine}` | 19 | 0.174 | **35.72** | 0.00188 | 0.977 | 5.77e-23 |
| `{bottled beer, red/blush wine}` → `{liquor}` | 19 | 0.396 | **35.72** | 0.00188 | 0.983 | 5.77e-23 |
| `{Instant food products}` → `{hamburger meat, soda}` | 12 | 0.152 | **26.21** | 0.00117 | 0.968 | 1.06e-12 |
| `{hamburger meat, soda}` → `{Instant food products}` | 12 | 0.211 | **26.21** | 0.00117 | 0.970 | 1.06e-12 |
| `{processed cheese}` → `{ham, white bread}` | 19 | 0.117 | **22.93** | 0.00185 | 0.961 | 2.55e-19 |
| `{ham, white bread}` → `{processed cheese}` | 19 | 0.380 | **22.93** | 0.00185 | 0.973 | 2.55e-19 |
| `{red/blush wine}` → `{bottled beer, liquor}` | 19 | 0.101 | **21.49** | 0.00184 | 0.958 | 6.27e-19 |
| `{bottled beer, liquor}` → `{red/blush wine}` | 19 | 0.413 | **21.49** | 0.00184 | 0.972 | 6.27e-19 |

Negative associations:

| Rule | Baskets | Conf | Lift | Leverage | Zhang | q |
|---|---:|---:|---:|---:|---:|---:|
| `{canned beer}` → `{whole milk}` | 87 | 0.114 | **0.45** | -0.01100 | -0.626 | 1.00e+00 |
| `{UHT-milk}` → `{whole milk}` | 39 | 0.119 | **0.46** | -0.00458 | -0.608 | 1.00e+00 |
| `{canned beer}` → `{root vegetables}` | 40 | 0.052 | **0.48** | -0.00440 | -0.548 | 1.00e+00 |
| `{canned beer}` → `{yogurt}` | 53 | 0.069 | **0.50** | -0.00545 | -0.540 | 1.00e+00 |
| `{white wine}` → `{whole milk}` | 26 | 0.139 | **0.54** | -0.00221 | -0.529 | 1.00e+00 |

## 6. Support threshold selection

At 97.4% sparsity, a conventional 1% minimum support admits only staples.
0.1% (≈9 baskets) was used instead, producing far more
candidates whose validity is then controlled statistically rather than by an arbitrary
frequency cutoff.

## 7. Limitations

- One month from one outlet. Seasonal affinities and store-specific assortment are baked in and will not generalise to other stores.
- Items are categories, not SKUs, so within-category substitution (two brands of yogurt) is invisible.
- Association is not causation. A lift of 3 between two items does not mean moving one next to the other will raise sales of the other; it may only mean both are bought on the same kind of shopping trip.
- Rules are mined and evaluated on the same transactions. A holdout split would test whether affinities persist across periods, which one month of data does not support.

*Generated at commit `b949297` · seed 42 · 130.63s · Python 3.11.15 · numpy 2.4.6 · pandas 3.0.5 · sklearn 1.9.1 · lightgbm 4.7.0*
