# `dsx` — shared data science toolkit

Every project in this portfolio imports this one library rather than carrying its own copy
of the same helpers. For a portfolio whose central claim is methodological rigour, that has
a concrete payoff: the leakage-safe split, the imbalance-aware metric and the
provenance-stamped artifact writer are implemented **once**, so an auditor verifies them
once and then knows the property holds everywhere they are imported.

| Module | Responsibility |
|---|---|
| `data.py` | Dataset registry. Declares every source with true origin, licence and a `REAL`/`SIMULATED` marker. Caches to `.data/`, pins content by SHA-256. |
| `splits.py` | Splitting that resists preprocessing, temporal and group leakage. Returns an auditable `SplitReport`. |
| `metrics.py` | Scores reported next to their no-skill baseline, so a figure reads as skill rather than as an impressive constant. |
| `artifacts.py` | Run seeding, provenance stamping, JSON artifact IO. Every number in the docs originates here. |
| `crispdm.py` | Records the six CRISP-DM phases as machine-readable evidence, including rejected alternatives. |

## Usage

```bash
export PYTHONPATH=lib
```

```python
from dsx import data, splits, metrics, artifacts

df = data.load_csv("titanic")
train_idx, test_idx, report = splits.stratified_split(df.Survived.values)

with artifacts.run("my_project", seed=42) as ctx:
    ...
    artifacts.write("projects/xx/artifacts/results.json", payload, context=ctx)
```

## Design rules these modules enforce

1. **No undeclared data.** A model script cannot reach an arbitrary URL; it asks the
   registry for a declared id. Unknown ids raise.
2. **No relabelled simulations.** `kind` is `REAL` or `SIMULATED`, with no third category
   and no euphemism. The provenance table printed into each README uses that field
   directly.
3. **No hand-typed metrics.** Documentation reads numbers back from committed JSON. If a
   paper and an artifact disagree, the artifact is correct and the paper is a bug.
4. **No bare accuracy on imbalanced targets.** `classification_report` always emits
   `prevalence` and `always_negative_accuracy` beside the model's own score.
