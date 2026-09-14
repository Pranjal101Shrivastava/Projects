# Walkthrough

A guided tour of the repository, for a reader who wants to check the claims rather than take
them on trust. Each section names the file that implements the thing being described.

---

## Start here if you have five minutes

1. **[`lib/dsx/splits.py`](./lib/dsx/splits.py)** — the leakage defences, in about 200 lines.
   `assert_pipeline_safe()` is the one that does the most work: it refuses any estimator not
   wrapped in a `Pipeline`, which makes preprocessing leakage structurally impossible rather
   than a thing to remember.

2. **[`tools/audit.py`](./tools/audit.py)** — the AST scanner that verifies the above across
   every pipeline. Read its docstring first: it states plainly what static analysis *cannot*
   establish.

3. **[`tools/tests/test_audit.py`](./tools/tests/test_audit.py)** — the tests proving the
   scanner works. This exists because the scanner's first version reported zero findings and
   that was a bug, not a clean bill of health.

4. **[`projects/06_automl_tournament/README.md`](./projects/06_automl_tournament/README.md)** —
   the single most useful result in the repository: one column, worth 40% of PR-AUC, invisible
   to every diagnostic except reasoning about causal timing.

---

## The shared toolkit

`lib/dsx/` is imported by all eight pipelines. One implementation, audited once.

| Module | What it guarantees |
|---|---|
| [`data.py`](./lib/dsx/data.py) | No pipeline can fetch an undeclared URL. Sources carry a `REAL`/`SIMULATED` marker and a SHA-256. |
| [`splits.py`](./lib/dsx/splits.py) | Preprocessing, temporal and group leakage each have a named defence, and every split emits an auditable report. |
| [`metrics.py`](./lib/dsx/metrics.py) | No metric is reported without its no-skill baseline. |
| [`artifacts.py`](./lib/dsx/artifacts.py) | Every run is seeded and every artifact is stamped with the commit that produced it. |
| [`crispdm.py`](./lib/dsx/crispdm.py) | Phases are structured evidence including *rejected alternatives*, not headings. |

### The one function worth reading in full

```python
def assert_pipeline_safe(estimator) -> None:
    if not isinstance(estimator, Pipeline):
        raise TypeError(
            "Estimator must be a sklearn Pipeline so that preprocessing is fitted "
            "strictly inside each training fold. ..."
        )
```

Preprocessing leakage produces no error and no metric anomaly — the score simply comes out a
little too high. The only reliable defence is structural, and this is it.

---

## How a pipeline is organised

Every `projects/NN_*/pipeline/build.py` follows the same six-phase shape, and records each
phase as it goes rather than describing it afterwards:

```python
crisp.record(Phase(
    name="data_preparation",
    summary="...",
    decisions=[Decision(
        question="How are rolling features protected from lookahead?",
        choice="shift(1) applied before .rolling(), never after.",
        rationale="A rolling mean computed without the prior shift includes the target "
                  "hour in its own predictor. The resulting fit looks excellent and "
                  "collapses in production.",
        alternatives_rejected=[
            ".rolling(w).mean() directly on the demand column — leaks the target into "
            "its own feature.",
            "centered=True rolling windows — leaks the future outright.",
        ],
    )],
    evidence=prep,
    risks=["K-means assumes isotropic clusters in degrees; ..."],
))
```

The `alternatives_rejected` field is the one doing real work. A write-up listing only what was
chosen is unfalsifiable.

---

## Tracing a number from artifact to screen

Take the fraud model's PR-AUC.

1. **Computed** in [`lib/dsx/metrics.py`](./lib/dsx/metrics.py) by
   `classification_report()`, which also emits `prevalence` and `always_negative_accuracy` so
   the figure cannot be quoted without its baseline.
2. **Written** to `projects/04_fraud_detection/artifacts/models.json` with a `_run` block
   recording commit, seed, library versions and duration.
3. **Synced** to `web/public/data/` by [`tools/sync_artifacts.py`](./tools/sync_artifacts.py),
   which validates the JSON as it copies.
4. **Fetched** at runtime by `useArtifacts()` in [`web/src/lib/data.ts`](./web/src/lib/data.ts)
   — not bundled, so each page loads only its own data.
5. **Rendered** by [`web/src/pages/views/Fraud.tsx`](./web/src/pages/views/Fraud.tsx). No
   component hard-codes a metric.
6. **Written into prose** by [`tools/docs.py`](./tools/docs.py), which reads the same JSON.

At no point is the number typed by a human. That is the mechanism behind "if prose and
artifact disagree, the artifact is correct".

---

## Verifying the claims yourself

```bash
export PYTHONPATH=lib

# 1. Does the leakage scanner actually detect leakage?
python3 -m pytest tools/tests/ -q
#    Two fixtures contain known-bad code. The tests assert each leak class is caught, that a
#    suppression without a written reason is rejected, and that this repository has no
#    unacknowledged critical findings.

# 2. Is the repository clean right now?
python3 tools/audit.py
#    39 checks passed, 0 critical, 1 acknowledged (with its reason printed).

# 3. Do the numbers reproduce?
python3 projects/03_market_basket/pipeline/build.py
#    ~2 minutes. Seeded, so it produces the same 13,106 itemsets and the same 189× speedup.

# 4. Does the site render real data?
cd web && npm ci && npm run build && cd ..
python3 tools/screenshots.py
#    Asserts every page rendered artifact data — not a loading or error placeholder — with
#    zero console errors, then captures it.
```

---

## The results that did not work

These are the ones worth reading, because they are where the methodology earned its keep.

### A conformal interval that under-covered · [Project 01](./projects/01_nyc_mobility/)

Split-conformal intervals held at 90% and 95% but realised only 71.2% at the nominal 80%
level. The cause is in the data: demand grew 82.9% across the window, so calibration residuals
(from the training period) and test residuals are not exchangeable. Wide bands absorb the
drift; the tight one cannot.

Reporting only the two calibrated levels would have produced a clean table.

### Algorithms that only half-agreed · [Project 02](./projects/02_customer_segmentation/)

K-means agrees with a Gaussian mixture at ARI 0.49 and with Ward linkage at 0.50. Roughly half
the apparent structure is the spherical assumption, not the data. Segment boundaries are one
defensible partition, not discovered natural kinds.

### Standard advice that made a model 82× worse · [Project 04](./projects/04_fraud_detection/)

`scale_pos_weight = n_neg/n_pos` is the conventional recommendation for imbalanced boosting.
It collapsed PR-AUC from 0.7359 to 0.0089. The ablation was run because a single configuration
produced an implausibly bad result and the reflex — quietly tune until it looks better — would
have hidden a real finding.

### A series where nothing beat naive · [Project 05](./projects/05_timeseries_forecasting/)

The sunspot cycle averages ~11 years but drifts and is not calendar-anchored. Every learned
model ranked below seasonal naive. A single-series study on airline data would have concluded
"SARIMA wins" and been wrong about the general case.

### A bias-variance curve that is not a U · [Project 08](./projects/08_crispdm_academy/)

Variance accounts for 3.6% of test error even at degree 15; bias dominates throughout.
Training error is also non-monotone under bootstrap averaging. Both departures from the
textbook figure are detected in code, not asserted from the textbook.

---

## Screens

Every capture below was taken by [`tools/screenshots.py`](./tools/screenshots.py), which
asserts the page rendered real artifact data with no console errors before saving.

| View | Screenshot |
|---|---|
| Portfolio index | [`00_home.png`](./docs/screenshots/00_home.png) |
| Methodology | [`00_methodology.png`](./docs/screenshots/00_methodology.png) |
| 01 · NYC demand | [`01_nyc_mobility.png`](./docs/screenshots/01_nyc_mobility.png) |
| 01 · Data provenance tab | [`01_nyc_mobility_data.png`](./docs/screenshots/01_nyc_mobility_data.png) |
| 02 · Segmentation | [`02_segmentation.png`](./docs/screenshots/02_segmentation.png) |
| 03 · Market basket | [`03_market_basket.png`](./docs/screenshots/03_market_basket.png) |
| 04 · Fraud | [`04_fraud.png`](./docs/screenshots/04_fraud.png) |
| 04 · CRISP-DM tab | [`04_fraud_method.png`](./docs/screenshots/04_fraud_method.png) |
| 05 · Forecasting | [`05_forecasting.png`](./docs/screenshots/05_forecasting.png) |
| 06 · AutoML & the leak | [`06_automl.png`](./docs/screenshots/06_automl.png) |
| 07 · Nano transformer | [`07_transformer.png`](./docs/screenshots/07_transformer.png) |
| 08 · CRISP-DM Academy | [`08_academy.png`](./docs/screenshots/08_academy.png) |
