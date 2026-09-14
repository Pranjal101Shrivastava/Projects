# Video scripts

One script per project — 12 separate videos, each standing on its own.

Every figure spoken in these scripts is read from a committed artifact by
[`tools/video_scripts.py`](../../tools/video_scripts.py), the same way the READMEs and the
site are generated. Re-run a pipeline and the scripts update, so what you say on camera
cannot drift away from what is on screen behind you.

| # | Script | Runtime |
|:---:|---|---|
| 01 | [NYC Ride-Hail Demand](./01_nyc_mobility.md) | ~6:41 |
| 02 | [Customer Segmentation](./02_customer_segmentation.md) | ~4:40 |
| 03 | [Market Basket Mining](./03_market_basket.md) | ~6:55 |
| 04 | [Fraud Detection](./04_fraud_detection.md) | ~6:03 |
| 05 | [Forecasting Tournament](./05_timeseries_forecasting.md) | ~4:25 |
| 06 | [AutoML & the Leak](./06_automl_tournament.md) | ~4:50 |
| 07 | [Nano Transformer](./07_nano_transformer.md) | ~5:31 |
| 08 | [CRISP-DM Academy](./08_crispdm_academy.md) | ~6:19 |
| 09 | [Sub-Linear Similarity Search](./09_similarity_search.md) | ~6:19 |
| 10 | [Fairness Audit · COMPAS](./10_fairness_audit.md) | ~7:10 |
| 11 | [Pipeline DAG Engine](./11_pipeline_dag.md) | ~5:53 |
| 12 | [Market Backtest](./12_market_backtest.md) | ~7:27 |

## How these are written

- **Plain text is spoken.** Lines marked *On screen* and *Note to self* are directions.
- **One idea per sentence**, because a listener cannot re-read a clause.
- **Numbers are said the way you'd say them aloud**, not read off like a table.
- Each script opens with a **hook** and closes by **handing off to the next video**, so the
  twelve work as a series as well as individually.
- Each one contains at least one **thing that went wrong** — a wrong claim corrected, a
  result that disagreed with expectation, or a limitation that cannot be engineered away.
  That is the argument of the portfolio, and it is what makes a walkthrough worth watching
  rather than a tour of green checkmarks.

## Suggested recording order

Record **06 · AutoML & the Leak** first. It is the strongest single idea in the portfolio
and the one most worth getting right while you are fresh. Record **01** second, since it
establishes the baseline argument that every later video leans on.

## Regenerating

```bash
python3 tools/video_scripts.py
```
