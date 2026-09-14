# Applied Data Science Portfolio

**Twelve end-to-end data science systems built on real, publicly documented data** — with
leakage controls that are enforced in code, baselines reported beside every metric, and the
results that did not work kept in.

🔗 **Live site: [https://pranjal101shrivastava.github.io/Projects](https://pranjal101shrivastava.github.io/Projects)**
<sub>The interactive site deploys automatically once GitHub Pages is enabled for this
repository (Settings → Pages → Source: **GitHub Actions**). Until then the link 404s —
but nothing on this page depends on it: every screenshot below is committed to the repo
and renders inline on GitHub, and every project runs locally.</sub>

[![Verify](https://github.com/Pranjal101Shrivastava/Projects/actions/workflows/verify.yml/badge.svg)](https://github.com/Pranjal101Shrivastava/Projects/actions/workflows/verify.yml)
[![Pages](https://github.com/Pranjal101Shrivastava/Projects/actions/workflows/pages.yml/badge.svg)](https://github.com/Pranjal101Shrivastava/Projects/actions/workflows/pages.yml)
![Real data](https://img.shields.io/badge/data-13%2F13%20REAL-brightgreen)
![Audit](https://img.shields.io/badge/leakage%20audit-57%20checks%20passed-brightgreen)
![Critical](https://img.shields.io/badge/critical%20findings-0-brightgreen)

---

## The projects

| # | Project | Domain | Data | Headline result |
|:---:|---|---|---|---|
| **[01](https://pranjal101shrivastava.github.io/Projects/#/p/nyc-mobility)** | [NYC Ride-Hail Demand](./projects/01_nyc_mobility/) | Spatio-temporal regression | 4.5M real NYC TLC dispatches | MAE **15.65** vs **25.70** naive (39.1% skill) |
| **[02](https://pranjal101shrivastava.github.io/Projects/#/p/segmentation)** | [Customer Segmentation](./projects/02_customer_segmentation/) | Unsupervised clustering | 41,188 real bank marketing contacts | conversion **4.15% → 63.83%** (χ²=4786) |
| **[03](https://pranjal101shrivastava.github.io/Projects/#/p/market-basket)** | [Market Basket Mining](./projects/03_market_basket/) | Association rule mining | 9,835 real grocery baskets | **189× speedup**, identical output; **5,144** rules rejected by FDR |
| **[04](https://pranjal101shrivastava.github.io/Projects/#/p/fraud)** | [Fraud Detection](./projects/04_fraud_detection/) | Imbalanced classification | 284,807 real ULB card transactions (0.17% fraud) | PR-AUC **0.7657** vs **0.00132** floor; standard advice cost **80×** |
| **[05](https://pranjal101shrivastava.github.io/Projects/#/p/forecasting)** | [Forecasting Tournament](./projects/05_timeseries_forecasting/) | Time series | 4 real series, 605 observations | **1 of 4** series where nothing beats naive |
| **[06](https://pranjal101shrivastava.github.io/Projects/#/p/automl)** | [AutoML & the Leak](./projects/06_automl_tournament/) | AutoML / data leakage | 41,188 real bank marketing contacts | one column inflated PR-AUC **0.4961 → 0.6939** (39.9%) |
| **[07](https://pranjal101shrivastava.github.io/Projects/#/p/transformer)** | [Nano Transformer](./projects/07_nano_transformer/) | Deep learning | 1,115,394 characters of Shakespeare | perplexity **4.55** vs **27.5** unigram (6.0×) |
| **[08](https://pranjal101shrivastava.github.io/Projects/#/p/academy)** | [CRISP-DM Academy](./projects/08_crispdm_academy/) | Statistical education | 5 real datasets, 6 modules | gradient check max error **1.4e-08** |
| **[09](https://pranjal101shrivastava.github.io/Projects/#/p/similarity-search)** | [Sub-Linear Similarity Search](./projects/09_similarity_search/) | Approximate nearest neighbour | 1,534 real company names (1,175,811 pairs) | **3.0×** at 100% recall; **45×** at 90.0% |
| **[10](https://pranjal101shrivastava.github.io/Projects/#/p/fairness)** | [Fairness Audit · COMPAS](./projects/10_fairness_audit/) | Algorithmic fairness | 6,172 real COMPAS defendants | FPR **42.3% vs 19.4%**; removing race leaves **0.1425** of a 0.2296 gap |
| **[11](https://pranjal101shrivastava.github.io/Projects/#/p/dag-engine)** | [Pipeline DAG Engine](./projects/11_pipeline_dag/) | Orchestration | this repository's 18-task graph (48 edges, 5 levels) | speedup ceiling **1.51×** — one task is 97.7% of the critical path |
| **[12](https://pranjal101shrivastava.github.io/Projects/#/p/backtest)** | [Market Backtest](./projects/12_market_backtest/) | Quantitative finance | 506 real AAPL daily bars | **0 of 2** strategies beat buy-and-hold; price R² **0.9753** vs return R² **-0.0001** |

Each project directory contains its pipeline, its committed artifacts, a README, an
`abstract.md`, a `paper.md`, a Medium-style `article.md` and its own `audit.md`.

---

## What makes this different from a typical portfolio

### 1 · The data is real, and its provenance is machine-readable

Every dataset is declared once in [`lib/dsx/data.py`](./lib/dsx/data.py) with its true origin,
licence, and an explicit `REAL` or `SIMULATED` marker. There is no third category and no
euphemism. Downloads are cached and **pinned by SHA-256**, so an upstream file that changes
underneath the work fails loudly instead of quietly shifting every metric downstream.

All **13 of 13** declared sources are `REAL`.

This matters because the reference portfolio this work responds to
([`dlmastery/data_science_examples`](https://github.com/dlmastery/data_science_examples))
labels its datasets "Kaggle" while generating all of them with `numpy.random` — see
[PROMPTS.md](./PROMPTS.md) for the evidence.

### 2 · Leakage prevention is structural, and the enforcement is tested

[`lib/dsx/splits.py`](./lib/dsx/splits.py) defends against the three leakage modes that
actually bite:

| Mode | Defence |
|---|---|
| Preprocessing | `assert_pipeline_safe()` refuses any estimator not wrapped in a `Pipeline`, so transforms refit inside each fold |
| Temporal | `temporal_split()` never shuffles and supports a boundary embargo for rolling-window features |
| Group | `grouped_split()` keeps entities whole across the split |

[`tools/audit.py`](./tools/audit.py) then walks the AST of every pipeline to verify it.

**The scanner is itself tested against known-bad code.** Its first version reported zero
findings across every project, which looked like success and was a bug — it matched only
`StandardScaler().fit(X)` inline and missed `scaler = StandardScaler(); scaler.fit(X)`. A
deliberately leaky fixture found that in one run.
[`tools/tests/`](./tools/tests/) now holds 3
files asserting each leak class is still detected.

Current state: **57 structural checks passed, 0 unacknowledged critical
findings, 0 warnings, 5 acknowledged** (rule fired, author recorded a written
reason — a bare suppression is rejected). Full report: [AUDIT.md](./AUDIT.md).

### 3 · Every metric appears beside its no-skill baseline

A score without a reference point is not a result. `classification_report()` always emits
prevalence and always-negative accuracy; `regression_report()` takes an explicit baseline;
`interval_coverage()` reports realised coverage next to the nominal level, which is the only
way a prediction interval can be falsified.

### 4 · Negative results are kept

A portfolio where everything worked is evidence of selective reporting, not of skill.

| Finding | Project |
|---|---|
| A conformal interval that under-covered — 71.2% at the nominal 80% level (1 of 3 levels miscalibrated), because demand growth broke exchangeability | [01](./projects/01_nyc_mobility/) |
| Clustering algorithms agreeing at only ARI 0.49 — much of the structure is the algorithm's assumption, not the data | [02](./projects/02_customer_segmentation/) |
| Standard advice for imbalanced boosting made the model 80× worse | [04](./projects/04_fraud_detection/) |
| 1 of 4 series where no learned model beats the naive baseline (sunspots) | [05](./projects/05_timeseries_forecasting/) |
| A bias-variance curve that is not a U — variance is only 3.6% of test error at maximum capacity, and training error is not monotone | [08](./projects/08_crispdm_academy/) |
| Full recall costs almost all of LSH's advantage — 3.0× where 90.0% recall buys 45× | [09](./projects/09_similarity_search/) |
| Removing race from the model left 62.1% of the false-positive-rate gap in place | [10](./projects/10_fairness_audit/) |
| A DAG scheduler that cannot beat 1.51× at any worker count, because one task is 97.7% of the critical path | [11](./projects/11_pipeline_dag/) |
| A trading study with no measurable signal — 0 of 2 strategies beat buy-and-hold — published as the result | [12](./projects/12_market_backtest/) |

Every figure in that table is read from an artifact too — including the inconvenient ones.

### 5 · No number in any document was typed by hand

Every figure in every README, paper, abstract and article is generated from a committed JSON
artifact by [`tools/docs.py`](./tools/docs.py). Each artifact carries a `_run` block recording
the git commit, library versions, seed and duration that produced it. **If prose and artifact
disagree, the artifact is correct and the prose is a bug.**

---

## Repository layout

```
lib/dsx/              shared toolkit — data registry, leakage-safe splits, honest metrics,
                      artifact IO, CRISP-DM recording
projects/NN_name/
  pipeline/build.py   the whole analysis, end to end
  artifacts/*.json    every number this project claims, with run provenance
  README.md           generated from those artifacts
  paper.md            formal write-up, generated
  abstract.md         publication-style abstract, generated
  article.md          Medium-style narrative, generated
  audit.md            per-project leakage audit
tools/
  audit.py            AST leakage scanner
  tests/              regression tests proving the scanner works
  docs.py             documentation generator
  readme.py           this file's generator
  screenshots.py      browser verification + screenshot capture
  sync_artifacts.py   artifacts → web
web/                  unified React + TypeScript site for all twelve projects
docs/screenshots/     verified captures of every page
.github/workflows/    CI verification and Pages deployment
```

---

## Video walkthrough

<!-- YOUTUBE-PLACEHOLDER -->
> 🎬 **A video walkthrough will be linked here.**
>
> It was explicitly deferred for this submission. In the meantime, the
> [live site](https://pranjal101shrivastava.github.io/Projects) is fully interactive, and
> [`docs/screenshots/`](./docs/screenshots/) contains browser-verified captures of every page
> — each one asserted to have rendered real artifact data with zero console errors before it
> was saved.

---

## Running it

```bash
git clone https://github.com/Pranjal101Shrivastava/Projects
cd Projects
pip install -r requirements.txt
export PYTHONPATH=lib

# Any single pipeline
python3 projects/04_fraud_detection/pipeline/build.py

# Everything
for p in projects/*/pipeline/build.py; do python3 "$p"; done
python3 tools/audit.py --write
python3 -m pytest tools/tests/ -q
python3 tools/sync_artifacts.py
python3 tools/docs.py
python3 tools/readme.py
cd web && npm install && npm run build
cd .. && python3 tools/screenshots.py
```

Datasets download on first use into `.data/` (gitignored) and are verified against their
recorded hashes thereafter. Every pipeline is seeded.

---

## Further reading

- **[PROMPTS.md](./PROMPTS.md)** — the prompts that produced this repository, and the
  investigation showing the reference portfolio's "Kaggle" datasets are `numpy.random`
- **[AUDIT.md](./AUDIT.md)** — full leakage and reproducibility audit
- **[`lib/README.md`](./lib/README.md)** — the shared toolkit and the rules it enforces
- **[Methodology](https://pranjal101shrivastava.github.io/Projects/#/methodology)** — the standing argument, on the live site

---

*Built with [Claude Code](https://claude.ai/code). Every figure above is read from a committed
artifact produced by a pipeline in this repository.*
