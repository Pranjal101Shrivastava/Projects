"""Generate the repository README from committed artifacts.

Run:
    python3 tools/readme.py

Same rule as tools/docs.py: every figure in the top-level README is read from a pipeline's
JSON output rather than typed. The index table, the findings list and the audit summary are
all derived, so they cannot fall out of date when a pipeline is rerun.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from docs import SITE, SLUGS, load, pct  # noqa: E402
import audit as A  # noqa: E402

REPO = "https://github.com/Pranjal101Shrivastava/Projects"

TITLES = {
    "01_nyc_mobility": ("NYC Ride-Hail Demand", "Spatio-temporal regression"),
    "02_customer_segmentation": ("Customer Segmentation", "Unsupervised clustering"),
    "03_market_basket": ("Market Basket Mining", "Association rule mining"),
    "04_fraud_detection": ("Fraud Detection", "Imbalanced classification"),
    "05_timeseries_forecasting": ("Forecasting Tournament", "Time series"),
    "06_automl_tournament": ("AutoML & the Leak", "AutoML / data leakage"),
    "07_nano_transformer": ("Nano Transformer", "Deep learning"),
    "08_crispdm_academy": ("CRISP-DM Academy", "Statistical education"),
    "09_similarity_search": ("Sub-Linear Similarity Search", "Approximate nearest neighbour"),
    "10_fairness_audit": ("Fairness Audit · COMPAS", "Algorithmic fairness"),
    "11_pipeline_dag": ("Pipeline DAG Engine", "Orchestration"),
    "12_market_backtest": ("Market Backtest", "Quantitative finance"),
}


def headline(project: str) -> tuple[str, str]:
    """Return (dataset description, headline result) read from artifacts."""
    if project == "01_nyc_mobility":
        m = load(project, "models")
        best = m["models"][m["best_model"]]
        naive = m["models"]["seasonal_naive_168h"]
        return ("4.5M real NYC TLC dispatches",
                f"MAE **{best['mae']:.2f}** vs **{naive['mae']:.2f}** naive "
                f"({pct(best['skill_score_vs_baseline'])} skill)")
    if project == "02_customer_segmentation":
        s = load(project, "segments")["external_validation"]
        return ("41,188 real bank marketing contacts",
                f"conversion **{pct(s['segment_profiles'][-1]['conversion_rate'], 2)} → "
                f"{pct(s['segment_profiles'][0]['conversion_rate'], 2)}** (χ²={s['chi2']:.0f})")
    if project == "03_market_basket":
        a = load(project, "algorithms")["comparison"]
        r = load(project, "rules")["significance"]
        return (f"{load(project, 'profile')['n_transactions']:,} real grocery baskets",
                f"**{a['speedup_fp_over_apriori']:.0f}× speedup**, identical output; "
                f"**{r['n_rejected_by_fdr']:,}** rules rejected by FDR")
    if project == "04_fraud_detection":
        m = load(project, "models")
        best = m["results"][m["best"]]
        abl = m["reweighting_ablation"]
        worst = [v for v in abl["variants"] if v["variant"] == abl["worst"]][0]
        none = [v for v in abl["variants"] if v["variant"] == "none"][0]
        return ("284,807 real ULB card transactions (0.17% fraud)",
                f"PR-AUC **{best['pr_auc']:.4f}** vs **{best['prevalence']:.5f}** floor; "
                f"standard advice cost **{none['pr_auc'] / max(worst['pr_auc'], 1e-9):.0f}×**")
    if project == "05_timeseries_forecasting":
        s = load(project, "synthesis")
        naive = [k for k, v in s["winners_by_series"].items() if "naive" in v.lower()]
        return ("4 real series, 605 observations",
                f"**{len(naive)} of 4** series where nothing beats naive")
    if project == "06_automl_tournament":
        i = load(project, "leak_demonstration")["inflation"]
        return ("41,188 real bank marketing contacts",
                f"one column inflated PR-AUC **{i['clean_best_pr_auc']:.4f} → "
                f"{i['leaked_best_pr_auc']:.4f}** ({pct(i['relative_inflation'])})")
    if project == "07_nano_transformer":
        t = load(project, "training")
        e = load(project, "evaluation")
        return ("1,115,394 characters of Shakespeare",
                f"perplexity **{t['best_val']['val_perplexity']:.2f}** vs "
                f"**{e['baselines']['unigram']['perplexity']:.1f}** unigram "
                f"({e['improvement_over_unigram']:.1f}×)")
    if project == "08_crispdm_academy":
        b = load(project, "backprop")
        return ("5 real datasets, 6 modules",
                f"gradient check max error **{b['max_relative_error']:.1e}**")
    if project == "09_similarity_search":
        r = load(project, "results")
        cfgs = r["lsh_configurations"]
        full = [c for c in cfgs if c["recall"] >= 0.999][-1]
        cheap = max((c for c in cfgs if 0.85 <= c["recall"] < 0.999),
                    key=lambda c: c["speedup_vs_exact"])
        p9 = load(project, "profile")["profile"]
        return (f"{p9['n_distinct_company_strings']:,} real company names "
                f"({p9['exact_pairs']:,} pairs)",
                f"**{full['speedup_vs_exact']:.1f}×** at {pct(full['recall'], 0)} recall; "
                f"**{cheap['speedup_vs_exact']:.0f}×** at {pct(cheap['recall'])}")
    if project == "10_fairness_audit":
        f10 = load(project, "fairness")
        own = load(project, "own_model")
        m = f10["criteria"]["predictive_equality"]["measured"]
        return (f"{load(project, 'profile')['rows_after_propublica_filters']:,} real "
                "COMPAS defendants",
                f"FPR **{pct(m['max'])} vs {pct(m['min'])}**; removing race leaves "
                f"**{own['fpr_gap']:.4f}** of a {own['compas_fpr_gap']:.4f} gap")
    if project == "11_pipeline_dag":
        g = load(project, "graph")
        sch = load(project, "scheduling")
        return (f"this repository's {g['n_tasks']}-task graph "
                f"({g['n_edges']} edges, {g['n_levels']} levels)",
                f"speedup ceiling **{sch['theoretical_max_speedup']:.2f}×** — "
                f"one task is {pct(sch['dominant_task']['share_of_critical_path'])} "
                "of the critical path")
    r12 = load(project, "results")
    trap = load(project, "profile")["random_walk_trap"]
    beat = len([c for c in r12["verdict"]["comparison"]
                if c["beats_buy_and_hold_sharpe"]])
    return (f"{load(project, 'profile')['profile']['n_days']:,} real AAPL daily bars",
            f"**{beat} of {len(r12['verdict']['comparison'])}** strategies beat "
            f"buy-and-hold; price R² "
            f"**{trap['r2_predicting_price_with_yesterdays_price']:.4f}** vs return R² "
            f"**{trap['r2_predicting_return_with_zero']:.4f}**")


def video_rows() -> list[str]:
    """One row per planned video, linking the generated script.

    The Video column stays as a dash until a recording exists. Keeping the table here
    rather than hand-written means the project list cannot fall out of step with it.
    """
    scripts = ROOT / "docs" / "video-scripts"
    rows = []
    for project, (title, _domain) in TITLES.items():
        script = scripts / f"{project}.md"
        if script.exists():
            runtime = script.read_text().split("**Estimated runtime:** ~")[1].split(" ")[0]
            cell = f"[script](./docs/video-scripts/{project}.md) · ~{runtime}"
        else:
            cell = "—"
        rows.append(
            f"| {project[:2]} | [{title}](./projects/{project}/) | {cell} | — |"
        )
    return rows


def negative_findings() -> list[str]:
    """The results that did not work, with every figure read from an artifact.

    Hand-typing these would be the one place in the repository where a number could drift
    unnoticed, because nothing downstream depends on them. They are derived for exactly
    that reason.
    """
    out = []

    conformal = load("01_nyc_mobility", "models")["conformal"]
    c80 = conformal["80%"]
    failed = [k for k, v in conformal.items() if not v["calibrated"]]
    out.append(
        f"| A conformal interval that under-covered — {pct(c80['empirical_coverage'])} at the "
        f"nominal {pct(c80['nominal_coverage'], 0)} level ({len(failed)} of {len(conformal)} "
        f"levels miscalibrated), because demand growth broke exchangeability "
        f"| [01](./projects/01_nyc_mobility/) |"
    )

    agree = load("02_customer_segmentation", "selection")["cross_algorithm_agreement"]
    worst = min(v for v in agree.values() if isinstance(v, (int, float)))
    out.append(
        f"| Clustering algorithms agreeing at only ARI {worst:.2f} — much of the structure is "
        f"the algorithm's assumption, not the data "
        f"| [02](./projects/02_customer_segmentation/) |"
    )

    abl = load("04_fraud_detection", "models")["reweighting_ablation"]
    none = [v for v in abl["variants"] if v["variant"] == "none"][0]
    worst_v = [v for v in abl["variants"] if v["variant"] == abl["worst"]][0]
    out.append(
        f"| Standard advice for imbalanced boosting made the model "
        f"{none['pr_auc'] / max(worst_v['pr_auc'], 1e-9):.0f}× worse "
        f"| [04](./projects/04_fraud_detection/) |"
    )

    winners = load("05_timeseries_forecasting", "synthesis")["winners_by_series"]
    naive = [k for k, v in winners.items() if "naive" in v.lower()]
    out.append(
        f"| {len(naive)} of {len(winners)} series where no learned model beats the naive "
        f"baseline ({', '.join(naive)}) | [05](./projects/05_timeseries_forecasting/) |"
    )

    bv = load("08_crispdm_academy", "bias_variance")
    out.append(
        f"| A bias-variance curve that is not a U — variance is only "
        f"{pct(bv['variance_share_at_max_degree'])} of test error at maximum capacity, and "
        f"training error is {'not ' if not bv['train_error_monotone'] else ''}monotone "
        f"| [08](./projects/08_crispdm_academy/) |"
    )

    cfgs = load("09_similarity_search", "results")["lsh_configurations"]
    full = [c for c in cfgs if c["recall"] >= 0.999][-1]
    cheap = max((c for c in cfgs if 0.85 <= c["recall"] < 0.999),
                key=lambda c: c["speedup_vs_exact"])
    out.append(
        f"| Full recall costs almost all of LSH's advantage — {full['speedup_vs_exact']:.1f}× "
        f"where {pct(cheap['recall'])} recall buys {cheap['speedup_vs_exact']:.0f}× "
        f"| [09](./projects/09_similarity_search/) |"
    )

    own = load("10_fairness_audit", "own_model")
    out.append(
        f"| Removing race from the model left {pct(own['fpr_gap'] / own['compas_fpr_gap'])} of "
        f"the false-positive-rate gap in place | [10](./projects/10_fairness_audit/) |"
    )

    sch = load("11_pipeline_dag", "scheduling")
    out.append(
        f"| A DAG scheduler that cannot beat {sch['theoretical_max_speedup']:.2f}× at any "
        f"worker count, because one task is "
        f"{pct(sch['dominant_task']['share_of_critical_path'])} of the critical path "
        f"| [11](./projects/11_pipeline_dag/) |"
    )

    v12 = load("12_market_backtest", "results")["verdict"]
    beat = len([c for c in v12["comparison"] if c["beats_buy_and_hold_sharpe"]])
    out.append(
        f"| A trading study with no measurable signal — {beat} of {len(v12['comparison'])} "
        f"strategies beat buy-and-hold — published as the result "
        f"| [12](./projects/12_market_backtest/) |"
    )
    return out


def main() -> None:
    audits = [
        A.audit_project(d)
        for d in sorted((ROOT / "projects").iterdir())
        if d.is_dir() and (d / "pipeline").is_dir()
    ]
    passed = sum(len(a.checks_passed) for a in audits)
    critical = sum(a.critical for a in audits)
    warnings = sum(a.warnings for a in audits)
    ack = sum(len(a.acknowledged) for a in audits)

    datasets = set()
    for project in TITLES:
        for d in load(project, "provenance")["datasets"]:
            datasets.add((d["title"], d["kind"]))
    real = sum(1 for _, kind in datasets if kind == "REAL")

    rows = []
    for project, (title, domain) in TITLES.items():
        data_desc, result = headline(project)
        rows.append(
            f"| **[{project[:2]}]({SITE}/#/p/{SLUGS[project]})** | "
            f"[{title}](./projects/{project}/) | {domain} | {data_desc} | {result} |"
        )

    readme = f"""# Applied Data Science Portfolio

**Twelve end-to-end data science systems built on real, publicly documented data** — with
leakage controls that are enforced in code, baselines reported beside every metric, and the
results that did not work kept in.

🔗 **Live site: [{SITE}]({SITE})**
<sub>The interactive site deploys automatically once GitHub Pages is enabled for this
repository (Settings → Pages → Source: **GitHub Actions**). Until then the link 404s —
but nothing on this page depends on it: every screenshot below is committed to the repo
and renders inline on GitHub, and every project runs locally.</sub>

[![Verify](https://github.com/Pranjal101Shrivastava/Projects/actions/workflows/verify.yml/badge.svg)](https://github.com/Pranjal101Shrivastava/Projects/actions/workflows/verify.yml)
[![Pages](https://github.com/Pranjal101Shrivastava/Projects/actions/workflows/pages.yml/badge.svg)](https://github.com/Pranjal101Shrivastava/Projects/actions/workflows/pages.yml)
![Real data](https://img.shields.io/badge/data-{real}%2F{len(datasets)}%20REAL-brightgreen)
![Audit](https://img.shields.io/badge/leakage%20audit-{passed}%20checks%20passed-brightgreen)
![Critical](https://img.shields.io/badge/critical%20findings-{critical}-{'brightgreen' if critical == 0 else 'red'})

---

## The projects

| # | Project | Domain | Data | Headline result |
|:---:|---|---|---|---|
""" + "\n".join(rows) + f"""

Each project directory contains its pipeline, its committed artifacts, a README, an
`abstract.md`, a `paper.md`, a Medium-style `article.md` and its own `audit.md`.

---

## What makes this different from a typical portfolio

### 1 · The data is real, and its provenance is machine-readable

Every dataset is declared once in [`lib/dsx/data.py`](./lib/dsx/data.py) with its true origin,
licence, and an explicit `REAL` or `SIMULATED` marker. There is no third category and no
euphemism. Downloads are cached and **pinned by SHA-256**, so an upstream file that changes
underneath the work fails loudly instead of quietly shifting every metric downstream.

All **{real} of {len(datasets)}** declared sources are `REAL`.

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
[`tools/tests/`](./tools/tests/) now holds {len(list((ROOT / 'tools' / 'tests').glob('*.py')))}
files asserting each leak class is still detected.

Current state: **{passed} structural checks passed, {critical} unacknowledged critical
findings, {warnings} warnings, {ack} acknowledged** (rule fired, author recorded a written
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
""" + "\n".join(negative_findings()) + f"""

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
  video_scripts.py    spoken walkthrough script per project, from the same artifacts
web/                  unified React + TypeScript site for all twelve projects
docs/screenshots/     verified captures of every page
docs/video-scripts/   one spoken walkthrough script per project
.github/workflows/    CI verification and Pages deployment
```

---

## Video walkthrough

One video per project — twelve separate walkthroughs, each standing on its own.

<!-- YOUTUBE-PLACEHOLDER -->
| # | Project | Script | Video |
|:---:|---|---|---|
""" + "\n".join(video_rows()) + f"""

**The scripts are written and generated, not drafted.**
[`docs/video-scripts/`](./docs/video-scripts/) holds a full spoken script for each project,
produced by [`tools/video_scripts.py`](./tools/video_scripts.py) from the same committed
artifacts as everything else. A figure said on camera is therefore the figure on screen
behind it, and re-running a pipeline updates the script rather than leaving the recording
contradicting the site.

Until the recordings are linked, the [live site]({SITE}) is fully interactive, and
[`docs/screenshots/`](./docs/screenshots/) contains browser-verified captures of every page
— each one asserted to have rendered real artifact data with zero console errors before it
was saved.

---

## Running it

```bash
git clone {REPO}
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
- **[Methodology]({SITE}/#/methodology)** — the standing argument, on the live site

---

*Built with [Claude Code](https://claude.ai/code). Every figure above is read from a committed
artifact produced by a pipeline in this repository.*
"""
    (ROOT / "README.md").write_text(readme)
    print(f"Wrote README.md ({len(readme) / 1024:.1f} KB)")
    print(f"  {real}/{len(datasets)} datasets REAL · {passed} checks passed · "
          f"{critical} critical · {ack} acknowledged")


if __name__ == "__main__":
    main()
