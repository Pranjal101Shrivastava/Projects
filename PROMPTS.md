# Prompts

This repository was built with **Claude Code** (Claude Opus 5) in a single session. This file
records the prompts that produced it, verbatim, in the order they were given.

The reference portfolio this work responds to —
[`dlmastery/data_science_examples`](https://github.com/dlmastery/data_science_examples) —
publishes its own [`PROMPTS.md`](https://github.com/dlmastery/data_science_examples/blob/main/PROMPTS.md),
and this file follows that convention.

---

## The brief

```text
Replicate the data science experiments i did in your favorite coding assistant - i provided
the prompts in the repository

https://github.com/dlmastery/data_science_examples

make sure you publish them in github repo and walk thru these in a youtube video and upload
and link to README.md

Note: use your creativity and not need to do exactly like above - improvise if possible. the
minimum bar is above

Checking all artifacts

Publish a link to YouTube video which walkthrough the code and ux of each of the projects
thoroughly in readme.md

I provided prompts.md with what prompts I used

Use your creativity to do more in each app and explain

skip the video part for now
before starting tell me what the requirement is and how you are going to acheive it in bullets
do not auto assume ask if there is a confusion
do not make any change in any other repository on my github
```

## Clarifications requested before starting

Four decisions were put to the user before any code was written, because each would have
materially changed the work:

| Question | Answer |
|---|---|
| How many projects — breadth or depth? | *"answer me what does the requirement say on this? if nothing stated choose first"* → the brief specifies no count, so **8 deep projects** |
| Kaggle is unreachable from the build environment | *"You'll supply Kaggle credentials"* → see below |
| How should a grader see the apps working, with the video skipped? | **GitHub Pages live demos + Playwright screenshots** |
| What stack? | *"Let me pick per project"* |

### On Kaggle

The user offered credentials. They would not have helped, and saying so was more useful than
accepting them:

```
CONNECT tunnel failed, response 403
"kind": "connect_rejected",
"detail": "gateway answered 403 to CONNECT (policy denial or upstream failure)",
"host": "www.kaggle.com:443"
```

The environment's network policy refuses the tunnel to `kaggle.com` before any authentication
occurs, so credentials are never transmitted. The user then asked:

```text
is kaggle madatory ?
```

Investigation of the reference repository showed it is not, and that the reference itself does
not use Kaggle. Grepping every `.py`, `.js` and `.ts` file across its 1,048 files for
`requests.get`, `urlretrieve`, `read_csv("http...`, `yfinance`, `read_parquet` and any Kaggle
API call returned **zero matches**. Its datasets are generated:

```python
# dlmastery/data_science_examples — 05_data_science_skills_lab/core/datasets.py
# Kaggle Benchmark Datasets Synthesizer for Data Science Skills Mastery Lab
def get_titanic_dataset(n_samples: int = 891, random_state: int = 42) -> pd.DataFrame:
    """Kaggle Titanic: Machine Learning from Disaster benchmark."""
    np.random.seed(random_state)
    pclass = np.random.choice([1, 2, 3], size=n_samples, p=[0.24, 0.21, 0.55])
```

```python
# 01_nyc_taxi_trip_prediction/ml/data_loader.py
# Dataset Synthesizer & Loader for NYC Taxi Trip Duration Model
```

```python
# 10_crispdm_masters_curriculum/backend/main.py
# Synthetic High-Fidelity Dataset Generation (Kaggle Adult Income Benchmark)
```

The word "Kaggle" survives only in docstrings and UI labels. Every dataset is a seeded NumPy
RNG.

This repository therefore uses **real data throughout** — including the genuine ULB credit
card fraud dataset, mirrored on a reachable host. Every source is declared with an explicit
`REAL` or `SIMULATED` marker in [`lib/dsx/data.py`](./lib/dsx/data.py); all eleven are `REAL`.

## Authorship

```text
So when the commits will be made, uh, in the GitHub repo, will it be under my name, or will
it be shown that Claude committed?
```

Commits are authored as `Pranjal Shrivastava <131521253+Pranjal101Shrivastava@users.noreply.github.com>`,
matching the repository's existing initial commit, so they appear under the user's profile and
count toward their contribution graph. Each commit message carries a
`Co-Authored-By: Claude Opus 5` trailer disclosing AI assistance — kept deliberately, since
building with a coding assistant is the explicit premise of the assignment.

---

## How the work was actually directed

Beyond the brief above, the session was self-directed. The decisions that shaped the result
were made while building, and each is recorded in the artifact it affected rather than here.
The ones that mattered most:

1. **Find out what the data actually contains before choosing a target.** The NYC TLC release
   has no trip-duration column, so Project 01 forecasts demand instead of synthesising a
   target it cannot observe.

2. **Establish the baseline before building the model.** Every project reports what doing
   nothing achieves, first. Several results only make sense against that reference — a
   PR-AUC of 0.77 is meaningless until you know the no-skill floor is 0.0017.

3. **Run the ablation instead of following the advice.** Standard guidance for imbalanced
   boosting made the fraud model 82× worse. That was discovered by testing four variants
   after a single configuration produced an implausibly bad result.

4. **Test the tooling against known-bad input.** The leakage scanner initially reported zero
   findings across all eight projects. That looked like success and was a bug — it missed the
   most common spelling of a preprocessing leak. Running it against a deliberately leaky
   fixture found the gap in one run.

5. **Keep the results that did not work.** A conformal interval that under-covers, a
   clustering whose algorithms only half-agree, a series where nothing beats naive, a
   bias-variance curve that refuses to be a U. These are reported as prominently as the
   successes, with their causes identified.

6. **Never type a number into prose.** Every figure in every README, paper, abstract and
   article is generated from a committed JSON artifact by
   [`tools/docs.py`](./tools/docs.py). If the text and the artifact disagree, the artifact is
   correct.

---

## Reproducing the whole repository

```bash
git clone https://github.com/Pranjal101Shrivastava/Projects
cd Projects
pip install -r requirements.txt
export PYTHONPATH=lib

for p in projects/*/pipeline/build.py; do python3 "$p"; done   # all eight pipelines
python3 tools/audit.py --write                                  # leakage audit
python3 -m pytest tools/tests/ -q                               # audit regression tests
python3 tools/sync_artifacts.py                                 # artifacts → web
python3 tools/docs.py                                           # regenerate all documentation
cd web && npm install && npm run build                          # build the site
cd .. && python3 tools/screenshots.py                           # verify render + capture
```

Every pipeline is seeded and stamps the producing commit into its artifacts, so a rerun at the
same commit reproduces the same numbers. Datasets download on first use into `.data/` and are
verified against their recorded SHA-256 thereafter.
