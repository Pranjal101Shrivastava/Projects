# Skills

Twelve reproduction guides, one per project. Each states a **transferable procedure** for a
different dataset — not instructions for rerunning the script already in this repository,
which would convey nothing.

Each guide is organised the same way: when to use it, the procedure step by step, and the
decision points where a practitioner has to think rather than follow.

| Skill | Project | What it teaches |
|---|---|---|
| [`spatio-temporal-demand-forecasting`](./projects/01_nyc_mobility/skills/spatio-temporal-demand-forecasting/SKILL.md) | [01](./projects/01_nyc_mobility/) | Forecasting counts over a spatial grid and a time index, with honest uncertainty |
| [`stability-validated-segmentation`](./projects/02_customer_segmentation/skills/stability-validated-segmentation/SKILL.md) | [02](./projects/02_customer_segmentation/) | Selecting a clustering by reproducibility rather than by internal validity indices |
| [`association-mining-with-fdr`](./projects/03_market_basket/skills/association-mining-with-fdr/SKILL.md) | [03](./projects/03_market_basket/) | Mining association rules and controlling the false discovery rate over them |
| [`rare-event-detection`](./projects/04_fraud_detection/skills/rare-event-detection/SKILL.md) | [04](./projects/04_fraud_detection/) | Building and evaluating a detector when the positive class is under 1% |
| [`rolling-origin-forecast-evaluation`](./projects/05_timeseries_forecasting/skills/rolling-origin-forecast-evaluation/SKILL.md) | [05](./projects/05_timeseries_forecasting/) | Comparing forecasters across series without fooling yourself with one split |
| [`target-leakage-detection`](./projects/06_automl_tournament/skills/target-leakage-detection/SKILL.md) | [06](./projects/06_automl_tournament/) | Finding the feature that cannot exist at scoring time, before it reaches a leaderboard |
| [`transformer-from-tensor-ops`](./projects/07_nano_transformer/skills/transformer-from-tensor-ops/SKILL.md) | [07](./projects/07_nano_transformer/) | Implementing a decoder from first principles and evaluating it against real baselines |
| [`teaching-from-real-data`](./projects/08_crispdm_academy/skills/teaching-from-real-data/SKILL.md) | [08](./projects/08_crispdm_academy/) | Building teaching material where assumption violations are the content |
| [`approximate-similarity-search`](./projects/09_similarity_search/skills/approximate-similarity-search/SKILL.md) | [09](./projects/09_similarity_search/) | Deploying LSH for near-duplicate detection and measuring what the approximation costs |
| [`group-fairness-audit`](./projects/10_fairness_audit/skills/group-fairness-audit/SKILL.md) | [10](./projects/10_fairness_audit/) | Auditing a scoring system against fairness criteria that cannot all hold at once |
| [`dependency-graph-scheduling`](./projects/11_pipeline_dag/skills/dependency-graph-scheduling/SKILL.md) | [11](./projects/11_pipeline_dag/) | Deciding whether parallelising a pipeline can help before building the orchestration |
| [`honest-backtesting`](./projects/12_market_backtest/skills/honest-backtesting/SKILL.md) | [12](./projects/12_market_backtest/) | Evaluating a forward-looking model so that a negative result survives to be reported |

---

## The one to read first

**[`target-leakage-detection`](./projects/06_automl_tournament/skills/target-leakage-detection/SKILL.md)**
is a review procedure rather than a modelling one, and it applies to every project anyone
shows you.

Its central instruction is a single question asked of every column:

> **Would this value exist at the moment the prediction must be made?**

It requires no statistics, takes ten minutes, and catches leaks that cross-validation,
calibration curves, confusion matrices and learning curves all miss — because those diagnose
whether the model fits the data, and a leak is a property of the data.

In Project 06 it was worth **39.9% of PR-AUC**.

The companion to it is
**[`honest-backtesting`](./projects/12_market_backtest/skills/honest-backtesting/SKILL.md)**,
which generalises the same instinct to anything that predicts forward in time: measure signal
before measuring money, purge overlapping labels out of the split, and prove the absence of
look-ahead by perturbing the future rather than asserting it. Project 12 applied it and found
nothing — which is the point. A protocol that can only report successes is not a protocol.

---

## The rules these guides share

Each skill restates the same five commitments in its own domain. They are implemented once, in
[`lib/dsx/`](./lib/), and enforced by [`tools/audit.py`](./tools/audit.py).

1. **Declare every data source, and mark it `REAL` or `SIMULATED`.** No third category, no
   euphemism.
2. **Make leakage structurally impossible, not a thing to remember.** Preprocessing inside
   pipelines, temporal splits that cannot shuffle, group splits that keep entities whole.
3. **Report every metric beside its no-skill baseline.** A score without a reference point is
   not a result.
4. **Record rejected alternatives, not just choices.** A write-up listing only what was
   chosen is unfalsifiable.
5. **Never type a number into prose.** Generate documentation from artifacts, so text and
   measurement cannot drift apart.

---

## Regenerating these

```bash
python3 tools/skills.py
```

Guides are generated from [`tools/skills.py`](./tools/skills.py), which reads each project's
provenance artifact so the datasets named in each guide stay accurate.
