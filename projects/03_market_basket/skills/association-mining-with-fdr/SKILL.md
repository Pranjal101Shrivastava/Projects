---
name: association-mining-with-fdr
description: Mine association rules and control the false discovery rate over them
---

# Mine association rules and control the false discovery rate over them

*Derived from [`projects/03_market_basket`](../../), which applied this procedure to: Groceries Market Basket Transactions.*

This is a **transferable procedure**, not instructions for rerunning the script in this
repository. It states what to do on a new dataset, where the judgement calls are, and which
failure modes this project actually hit.

## When to use this

You have transaction baskets and want item affinities that survive scrutiny.

## Procedure

### 1. Choose the algorithm by data shape

**Apriori** — breadth-first, one database pass per level, prunes candidates by downward
closure. Simple to reason about, expensive on dense data.

**FP-Growth** — two database passes total, builds a frequency-ordered prefix tree, mines
conditional trees recursively, generates no candidates.

On sparse basket data FP-Growth is typically 100–200× faster. Implement both once if you are
learning; the cross-check is worth more than the speed.

### 2. Set the support floor from the sparsity, not from convention

At high sparsity a conventional 1% floor admits only staples and every rule becomes a
variation on "people buy the most popular item". Lower it, and control validity statistically
instead.

### 3. Compute six measures, not one

| Measure | Fails at |
|---|---|
| Confidence | Ignores the marginal frequency of the consequent |
| Lift | Symmetric; unstable at low support |
| Conviction | — (directional) |
| Leverage | — (absolute volume) |
| Zhang's metric | — (distinguishes dissociation) |
| q-value | — (distinguishes from noise) |

Ranking by confidence surfaces whatever is popular. Ranking by lift surfaces rare
coincidences.

### 4. Correct for multiple comparisons

Tens of thousands of simultaneous tests guarantee striking lifts by chance.

```python
from scipy.stats import fisher_exact
_, p = fisher_exact([[a, b], [c, d]], alternative="greater")
# then Benjamini-Hochberg step-up over all rules
```

Use **Benjamini-Hochberg, not Bonferroni**. At this scale Bonferroni rejects real affinities
along with the noise. BH bounds the expected proportion of false discoveries in the reported
list, which matches how a ranked shortlist is used.

**Report how many rules you rejected.** A table of survivors without that count invites the
reader to believe every rule ever tested was real.

### 5. Look for dissociations

Lift below 1 compresses all negative association into [0, 1). Zhang's metric spans [−1, 1] and
separates them. Dissociations identify distinct shopping occasions, which is actionable.

## Decision points requiring judgement

**Association is not causation.** High lift between two items may only mean both belong to the
same kind of trip. Co-locating them may change nothing.

**Verify your correction.** Check your BH implementation against
`statsmodels.multipletests` on synthetic p-values before trusting it.

---

## Reproduce the reference implementation

```bash
export PYTHONPATH=lib
python3 projects/03_market_basket/pipeline/build.py
python3 tools/audit.py
```

Results, with their run provenance, land in
[`projects/03_market_basket/artifacts/`](../../artifacts/).
