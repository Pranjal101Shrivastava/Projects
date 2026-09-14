"""Generate reproduction skill guides for each project.

Run:
    python3 tools/skills.py

The reference portfolio ships a `skills/` directory per project. This produces the equivalent:
a procedural guide for reproducing each analysis from scratch on a *different* dataset, which
is the only version of "skill" worth packaging. A guide that only tells you how to rerun the
script that is already in the repository conveys nothing.

Each guide states the transferable procedure, the decision points where a practitioner must
think rather than follow, and the specific failure modes that this project encountered.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from docs import load, pct  # noqa: E402

SKILLS = {
    "01_nyc_mobility": (
        "spatio-temporal-demand-forecasting",
        "Forecast demand over a spatial grid and a time index, with honest uncertainty",
    ),
    "02_customer_segmentation": (
        "stability-validated-segmentation",
        "Select a clustering by reproducibility rather than by internal validity indices",
    ),
    "03_market_basket": (
        "association-mining-with-fdr",
        "Mine association rules and control the false discovery rate over them",
    ),
    "04_fraud_detection": (
        "rare-event-detection",
        "Build and evaluate a detector when the positive class is under 1%",
    ),
    "05_timeseries_forecasting": (
        "rolling-origin-forecast-evaluation",
        "Compare forecasters across series without fooling yourself with one split",
    ),
    "06_automl_tournament": (
        "target-leakage-detection",
        "Find the feature that cannot exist at scoring time, before it reaches a leaderboard",
    ),
    "07_nano_transformer": (
        "transformer-from-tensor-ops",
        "Implement a decoder from first principles and evaluate it against real baselines",
    ),
    "08_crispdm_academy": (
        "teaching-from-real-data",
        "Build statistical teaching material where assumption violations are the content",
    ),
    "09_similarity_search": (
        "approximate-similarity-search",
        "Deploy LSH for near-duplicate detection and measure what the approximation costs",
    ),
    "10_fairness_audit": (
        "group-fairness-audit",
        "Audit a deployed scoring system against fairness criteria that cannot all hold",
    ),
    "11_pipeline_dag": (
        "dependency-graph-scheduling",
        "Decide whether parallelising a pipeline can help before building the orchestration",
    ),
    "12_market_backtest": (
        "honest-backtesting",
        "Evaluate a trading or any forward-looking model so that a negative result survives",
    ),
}

BODIES = {
    "01_nyc_mobility": """
## When to use this

You have event records with a timestamp and a location, and you need to forecast how many
events will occur in each region in each time bucket.

## Procedure

### 1. Establish what the data can support before choosing a target

Read the schema before writing the research question. The columns present determine what can
be predicted; the columns you wish were present determine what cannot.

If the target you intended is absent, **do not synthesise it**. A model fitted to a generated
target measures how well it recovers the generator.

### 2. Partition space by where events actually are

K-means over event coordinates, not a uniform grid. A uniform grid over any real
metropolitan bounding box is mostly water, parkland and industrial land; most cells carry
near-zero counts while a handful aggregate wildly different neighbourhoods.

Check the coordinate range first. Real location data routinely contains points from entirely
different markets.

### 3. Build the panel, then reindex it before lagging

```python
panel = events.groupby(["zone", "hour"]).size().reset_index(name="demand")

# Critical: reindex onto a COMPLETE time grid before computing any lag.
full = pd.date_range(panel.hour.min(), panel.hour.max(), freq="h")
grid = pd.MultiIndex.from_product([zones, full], names=["zone", "hour"])
panel = panel.set_index(["zone", "hour"]).reindex(grid, fill_value=0).reset_index()
```

Without the reindex, a missing hour silently causes `lag_24h` to reach twenty-five hours back.

### 4. Shift before rolling. Always.

```python
grouped = panel.groupby("zone")["demand"]

# WRONG — the window includes the value being predicted
panel["roll"] = grouped.rolling(24).mean()

# RIGHT — the window ends at t-1
panel["roll"] = grouped.shift(1).rolling(24).mean()
```

This is the single most common silent leak in demand forecasting. The fit looks excellent and
collapses in production, and no metric reveals it.

### 5. Split chronologically, with an embargo

The embargo width should equal your widest feature lag. Training rows within one lag-width of
the boundary share history with the first test rows.

### 6. Build the baseline ladder before the model

For periodic human activity, **test weekly before daily**. Weekday-versus-weekend is usually
a larger effect than day-to-day drift.

| Rung | Why |
|---|---|
| Seasonal naive at the dominant period | The number every model must beat |
| A second naive at a shorter period | Establishes which periodicity dominates |
| Regularised linear | Is non-linearity earning its complexity? |
| Gradient boosting | Only now |

### 7. Do not report R² alone

On any series with a strong daily cycle, everything scores high R². Report the error metric
and the skill score against the naive baseline.

## Decision points requiring judgement

**Duplicate records.** Check whether duplicate share correlates with the event rate. If it
rises with volume, they are collisions under coarse timestamp or coordinate resolution, not
faults — dropping them understates demand exactly at peak.

**Conformal intervals under trend.** Split conformal assumes calibration and test residuals
are exchangeable. If the series has a trend, calibration residuals drawn from the training
period will be smaller than test residuals and narrow bands will under-cover. Measure realised
coverage; do not assume the nominal level holds.
""",
    "02_customer_segmentation": """
## When to use this

You need customer or entity segments that someone will act on, and you need to know whether
they are real.

## Procedure

### 1. Exclude anything unknowable at segmentation time

Any feature generated by the outcome you eventually want to study must leave the feature
space, or your segments encode the answer.

Hold at least one outcome variable out entirely. It becomes your external validation.

### 2. Keep 'unknown' as a level

Missing-not-at-random categorical data is informative. An entity whose attribute was never
recorded differs systematically from one where it was. Mode imputation erases that and
manufactures certainty.

### 3. Sweep k against multiple indices

Silhouette rewards separation, Calinski-Harabasz rewards compactness against spread,
Davies-Bouldin penalises overlap. Let them disagree — a k winning on one and losing the others
is not a finding.

### 4. Bootstrap the stability. This is the step almost everyone skips.

```python
reference = KMeans(n_clusters=k, random_state=seed).fit_predict(X)
scores = []
for b in range(30):
    idx = rng.integers(0, len(X), len(X))          # resample with replacement
    shared = np.unique(idx)
    boot = KMeans(n_clusters=k, random_state=seed + b).fit(X[idx]).predict(X[shared])
    scores.append(adjusted_rand_score(reference[shared], boot))

stable = np.mean(scores) >= 0.75
```

A partition that changes when you resample the same population describes the sample.

### 5. Expect the criteria to disagree, and prefer reproducibility

Internal indices optimise a mathematical objective. They say nothing about whether the
partition survives a new sample or whether anyone can act on it.

### 6. Validate against the held-out outcome

Chi-square across segments. If conversion (or churn, or whatever you held out) does not differ
materially, the segmentation is statistically tidy and commercially useless.

### 7. Measure cross-algorithm agreement, and report it

Fit a Gaussian mixture and a hierarchical clustering at the same k. Compare by ARI. Agreement
near 0.5 means roughly half the structure is your algorithm's geometric assumption.

## Decision points requiring judgement

**Macro or contextual features.** Variables that change with the calendar rather than with the
entity will produce segments that are really time periods. Sweep both feature sets and choose
on stability.

**A high-converting segment defined by prior engagement** has usually rediscovered "people who
engaged before engage again". True, and not a latent group.
""",
    "03_market_basket": """
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
""",
    "04_fraud_detection": """
## When to use this

Rare-event detection: fraud, defects, intrusions, adverse events. Anything with a positive
rate below a few percent.

## Procedure

### 1. Compute the null-model accuracy first, and keep it visible

```python
prevalence = y.mean()
print(f"Always-negative accuracy: {1 - prevalence:.4%}")
```

Report it beside every accuracy figure you ever quote. This makes the metric impossible to
misuse.

### 2. Lead with PR-AUC, and show ROC-AUC beside it

ROC's false-positive rate divides by every negative in the dataset, so thousands of false
alarms barely move it. Precision divides by the model's own alert volume — what the operator's
queue contains.

Expect them to diverge sharply. That divergence is worth showing explicitly.

### 3. Split by time if events are bursty

Fraud, intrusions and defects cluster. A random split scatters one episode across both
partitions, so the model is scored on records whose siblings it trained on. Every metric rises
and none of them warns you.

### 4. Do not synthesise minority samples

SMOTE interpolates between minority neighbours. In high-dimensional space with few positives
the neighbours are far apart, so synthetic points land where no real positive has been
observed. The model learns a boundary around fabricated data.

Use class weighting, which rebalances without inventing observations.

### 5. Ablate the reweighting — do not follow the advice

The standard recommendation (`scale_pos_weight = n_neg/n_pos`) is derived from linear models
and does not transfer to high-capacity ensembles. With few positives, an extreme weight makes
every split chase the same handful of rows.

```python
for variant in [{}, {"scale_pos_weight": 10},
                {"scale_pos_weight": ratio}, {"is_unbalance": True}]:
    ...  # measure, do not assume
```

**Publish all variants**, not the winner.

### 6. Choose the threshold from costs, not from 0.5

Trace expected cost across thresholds under your FN:FP ratio. State the ratio as an
assumption and publish the whole curve so a reader with different costs can pick their own
point.

### 7. Report the alert-budget table

Operationally, the question is not "what threshold" but "how many alerts can we review". The
table showing recall and precision at 50, 100, 250, 500 and 1,000 alerts is more useful to a
decision-maker than any single metric.

## Decision points requiring judgement

**The no-skill floor is the prevalence of the set being scored**, not of the full dataset. A
chronological split does not preserve the base rate.

**Threshold selected on the test set is mildly optimistic.** Use a separate validation period
in production.
""",
    "05_timeseries_forecasting": """
## When to use this

You need to choose a forecasting method and want the choice to generalise beyond the one
series you tested on.

## Procedure

### 1. Use more than one series, chosen for structural diversity

One series tells you which model fits that series. Pick series that differ in *structure*:
trend with calendar seasonality, seasonality without trend, and — importantly — at least one
whose periodicity is **not calendar-anchored**.

Approximate, drifting cycles (solar, economic, epidemic, fashion) defeat every method that
assumes a fixed period. Without one in your set you will over-recommend seasonal models.

### 2. Use MASE, not MAPE or raw RMSE

    MASE = mean(|y − ŷ|) / mean(|y_t − y_{t−m}|)

Unit-free and anchored: 1.0 means "matched seasonal naive". This is the only sane way to
average across series with different units.

MAPE is undefined at zero, explosive near it, and asymmetric between over- and
under-forecasts.

### 3. Roll the origin; refit at each one

```python
for origin in origins:
    train, actual = values[:origin], values[origin:origin + h]
    prediction = model_fn(train, h)          # refit on this origin's data only
```

A single split reports where one arbitrary cut fell. Report the mean *and the spread* across
origins.

### 4. Use direct multi-step for ML forecasters

Recursive forecasting feeds a model its own predictions, so error compounds and the far
horizon becomes predictions of predictions. Train h separate models instead.

### 5. Run ADF and KPSS, and interpret them jointly

Their null hypotheses are opposites. A single test that fails to reject is ambiguous between
"the null holds" and "the test is underpowered". Together they distinguish trend-stationary
(detrend) from difference-stationary (difference).

Expect conflicts. Report them rather than picking whichever agreed with your expectation.

### 6. Report the series where nothing works

If a naive baseline wins on one of your series, that is the most informative row in the table.

## Decision points requiring judgement

**High-frequency seasonality.** A seasonal model at m=365 must estimate 365 seasonal lags —
intractable and unidentifiable. Aggregate to a coarser frequency that preserves the cycle.
Setting m=7 instead, silently, models a cycle that may not exist.

**Fixed hyperparameters favour classical methods.** Note it as a limitation rather than
claiming a fair fight.
""",
    "06_automl_tournament": """
## When to use this

Before trusting any leaderboard. This is a review procedure, not a modelling one.

## Procedure

### 1. Ask the causal-timing question of every column

> **Would this value exist at the moment the prediction must be made?**

This requires no statistics and catches leaks nothing else will. For each feature, identify
when it gets populated relative to the prediction event.

Classic offenders:

| Feature | Why it leaks |
|---|---|
| Call / session duration | Produced by the interaction being predicted |
| Final status fields | Written after the outcome |
| Aggregates over "all time" | Include the future |
| Row identifiers correlated with time | Encode ordering |
| Anything from a downstream system | Populated after the decision |

### 2. Screen univariate predictive power

```python
for column in features:
    print(column, roc_auc_score(y, df[column]))
```

Anomalously high single-feature scores warrant investigation. **Necessary but not sufficient**
— genuinely strong features exist, and leaks can be subtle.

### 3. Look for impossible cells

The decisive evidence is usually a degenerate case. If duration = 0 always means "no", the
feature is downstream of the outcome. Cross-tabulate extremes against the target.

### 4. Quantify it: run the tournament twice

Identical protocol, identical folds, identical seeds. One difference: the suspect column.

The gap is the price of the leak, and it is the only way to make the argument concretely to
someone who wants to keep the feature.

### 5. Understand what will NOT catch it

- Cross-validation — clean, because the model is correctly modelling the data it was given
- Generalisation gap — healthy
- Calibration curves — fine
- Confusion matrices — normal
- Learning curves — unremarkable

There is no statistical test. A leaking feature and a strong feature have the same
distributional signature.

### 6. When building the tournament itself

**Search families, not just hyperparameters.** A sweep over boosting depths is a tuner.
Include a linear model and a deliberately weak reference so you can see how much of the score
came from capacity.

**Preprocessing inside the pipeline**, so each fold refits it.

**Stack on out-of-fold predictions only.** Fitting the meta-learner on in-fold predictions
lets it see base-model outputs for rows those models memorised — leakage one level up.

## Decision points requiring judgement

Some leaks are legitimate features in a different framing. Prior-campaign outcome is knowable
before a call and is fine; call duration is not. The question is always *when*, never *how
predictive*.
""",
    "07_nano_transformer": """
## When to use this

Learning transformer internals, or building a small model where you need to understand every
component.

## Procedure

### 1. Implement the components, do not assemble them

`nn.TransformerEncoderLayer` hides exactly the parts worth understanding, and it does not
expose rotary embeddings, gated feed-forwards or pre-norm arrangement.

### 2. Rotary position embeddings

Rotate each (even, odd) coordinate pair of q and k by θ = p·ω_i. The dot product of two
rotated vectors depends only on the *difference* in position, so attention becomes a function
of relative offset — with nothing learned and no absolute index to overfit.

```python
x_even, x_odd = x[..., 0::2], x[..., 1::2]
rotated = torch.stack([x_even * cos - x_odd * sin,
                       x_even * sin + x_odd * cos], dim=-1)
return rotated.flatten(-2)
```

### 3. Pre-norm, not post-norm

`x + Attn(LN(x))`, never `LN(x + Attn(x))`. Post-norm routes the residual through the
normaliser, making gradient magnitude depth-dependent.

### 4. SwiGLU feed-forward

`(SiLU(W_gate x) ⊙ W_up x) W_down`. Consistently beats ReLU at matched parameter count.

### 5. Verify the causal mask

Use `is_causal=True` (or an explicit upper-triangular mask). Without it the model sees the
token it must predict, the loss collapses toward zero, and it learns nothing generative. A
suspiciously low training loss in the first hundred steps is the symptom.

### 6. Split continuous text contiguously

A random split interleaves validation windows between training windows, so validation context
is already memorised. Same reasoning as time series.

### 7. Evaluate against baselines, not against impressions

| Baseline | Meaning |
|---|---|
| Uniform over vocabulary | The ceiling — knowing nothing |
| Unigram (character frequencies) | Knowing only the marginal distribution |
| Your model | Must beat the unigram substantially |

**Decide your metrics before you see output.** A language model can always be made to look
good by selecting a sample.

### 8. Measure something falsifiable about generation

Real-word rate across temperatures is one option. State clearly what it does *not* measure:
orthography is not semantics, and a sample can be 95% real words and mean nothing.

### 9. Derive expected parameter count in code, and assert it

Documentation claiming a size that the architecture no longer produces is a silent
inaccuracy. Compute it from the configuration constants and fail the run on mismatch.

## Decision points requiring judgement

**Warmup is still needed with pre-norm.** Adam's second-moment estimate is unreliable in the
first few dozen steps.

**Character level versus subword.** Character keeps embeddings negligible so parameters sit in
the blocks, and makes spelling an observable achievement rather than the tokeniser's job.
""",
    "08_crispdm_academy": """
## When to use this

Building teaching material, or checking whether a concept you learned from a textbook figure
behaves that way in practice.

## Procedure

### 1. Use real data, and let it misbehave

Generated data satisfies its assumptions. That is the point of generating it, and it is why
students who learn from it are unprepared for the first real dataset.

### 2. Detect assumption violations in code, not in prose

Do not write "the bias-variance curve forms a U". Compute whether it does:

```python
variance_share = highest["variance"] / highest["total_test_mse"]
bias_dominates = variance_share < 0.2

train_monotone = all(rows[i]["train_mse"] >= rows[i + 1]["train_mse"] - tol
                     for i in range(len(rows) - 1))
```

Then branch the explanation on the measured result. This is the only way the lesson stays
correct when the data changes.

### 3. Quantify the violation rather than mentioning it

Naive Bayes assumes conditional independence. Measure Cramér's V between feature pairs *within
each class* and report how many pairs violate it, then show the consequence: ranking usually
survives, calibration usually does not.

### 4. Include the failure cases teaching figures omit

- A learning rate that **diverges**, not just one that converges slowly
- A sampling distribution still visibly skewed at n = 30
- A bias-variance curve with no visible U

These are more memorable and more useful than the clean versions.

### 5. Make gradient checking a first-class lesson

```python
numerical = (f(theta + eps) - f(theta - eps)) / (2 * eps)
relative_error = abs(numerical - analytic) / (abs(numerical) + abs(analytic))
assert relative_error < 1e-6
```

Correct gradients agree to ~1e-7. An error near 0.3 is a bug, not noise. This is the single
most useful debugging tool for a network that will not train.

### 6. Show the same model under two metrics

The clearest way to teach why a metric matters is to show one model scoring well on one and
badly on another. ROC-AUC 0.94 with PR-AUC 0.04 on the same predictions makes the point in a
way no explanation does.

### 7. Flag your own artefacts

If a low-n estimate is erratic because of sampling noise, say so rather than smoothing the
curve. The artefact is itself a lesson about estimation.

## Decision points requiring judgement

**One dataset per concept illustrates; it does not establish.** Say which you are doing.

**Single-answer quizzes check recall, not judgement** — which is the actual skill the material
argues for.
""",
    "09_similarity_search": """
## When to use this

You need to find near-duplicates — customer records, company names, documents, product
listings — and the number of pairs has made exhaustive comparison unattractive.

## Procedure

### 1. Compute the exact answer while you still can

This is the step almost every LSH write-up skips, and it is the one that makes the rest
meaningful. Below roughly a million pairs, exhaustive comparison is seconds of work. Do it,
save the result, and score every approximation against it.

If your data is already too large for this, **subsample it until it is not**. A 5,000-item
sample gives you a ground-truth recall curve that transfers to the full set far better than
no curve at all.

### 2. Normalise deliberately, and record what you chose not to do

Lowercasing and punctuation stripping are safe. Removing corporate suffixes (`Inc`, `LLC`,
`GmbH`) is *not* — it makes matching look better by deleting the exact tokens that
distinguish two legally separate entities. Whatever you decide, record it beside the results:
normalisation choices move recall more than parameter tuning does.

### 3. Shingle, then estimate

Character k-grams (k = 3 for short strings like names, larger for documents) turn strings
into sets so Jaccard applies. MinHash with k permutations estimates Jaccard in O(k) per item
instead of O(|set|) per pair.

### 4. Choose bands and rows as a threshold decision, not a tuning exercise

P(candidate) = 1 − (1 − sʳ)ᵇ, with the step near (1/b)^(1/r). Pick where you want the step —
that is the whole design. Sweep several (b, r) pairs and report the curve, not one point.

### 5. Include verification time in every speedup you report

LSH proposes; exact comparison disposes. A speedup computed from banding time alone is
measuring half the pipeline, and the half it omits grows with recall.

### 6. Report accuracy stratified by true similarity

The estimator's standard deviation is √(s(1−s)/k), maximal at s = 0.5 and near zero at the
extremes. Since most random pairs are dissimilar, any aggregate error figure is dominated by
the easy cases and looks far better than the estimator actually is on the pairs you care
about.

## Decision points requiring judgement

**What recall do you actually need?** Full recall typically costs most of the speed
advantage. Whether that matters depends on whether a missed pair is an inconvenience or a
compliance failure — and no benchmark can answer that.

**F1 is probably the wrong selector.** It weights a missed pair and a wasted comparison
equally. Almost no application does.

## Failure modes this project hit

**Comparing observed spread against 1/√k.** That rule of thumb is *twice* the true maximum of
0.5/√k. Combined with an unstratified aggregate, it made the estimator appear to beat its own
theoretical variance — an impossible result, and the signal that the theory line was wrong
rather than the measurement. If your estimator looks better than theory, you have the theory
wrong.

**Ground truth that is not ground truth.** "Jaccard above a threshold" is not "the same
entity". Recall figures measure agreement with exact string search, not with reality, and
saying so is part of reporting them.
""",
    "10_fairness_audit": """
## When to use this

A scoring system makes or informs decisions about people, and you need to establish whether
it treats groups differently — in a way that will survive scrutiny from someone who disagrees
with your conclusion.

## Procedure

### 1. Build one contingency table per group, and derive everything from it

TP, FP, FN, TN per group. Every fairness metric in the literature is a ratio of these four
numbers. Computing them first means you can evaluate any criterion later without re-running
anything, and it makes contradictions between criteria visible rather than mysterious.

### 2. Set a minimum group size and honour it

Groups below a few hundred members produce error rates with intervals too wide to interpret.
Report them as excluded, with their sizes, rather than silently dropping them or quoting
point estimates. State plainly that the smallest groups are the least likely to be audited
anywhere — that exclusion is a finding, not housekeeping.

### 3. Evaluate several criteria, and rank them rather than pass/fail them

Demographic parity, equal opportunity (FNR), predictive equality (FPR), predictive parity
(PPV). At any strict tolerance, a real system fails all of them — and a column of four
"VIOLATED" verdicts erases the entire substance of the argument. Rank by distance from parity
and report the ratio between the closest and the furthest.

### 4. Verify the impossibility on your own data

    FPR = (p / (1 − p)) · ((1 − PPV) / PPV) · (1 − FNR)

Compute the right-hand side from your observed counts and compare it to the observed FPR. The
discrepancy should be floating-point rounding. Once you have shown that on your own numbers,
"you cannot have equal error rates and equal predictive value when base rates differ" stops
being a citation and becomes a measurement.

### 5. Test the obvious fix so nobody has to ask

Retrain without the protected attribute. Show the remaining gap. Fairness through unawareness
is the first thing every stakeholder proposes, and demonstrating that correlated features
carry the same information is faster than arguing about it.

### 6. Report accuracy only to disarm it

Accuracy is usually near-identical across groups, which is exactly why a vendor can quote it
truthfully while a critic is also right. Show it, then show why a single number that averages
a false positive and a false negative hides who each error lands on.

## Decision points requiring judgement

**Which criterion matters is a policy question, not a statistical one.** Your job is to make
the trade-off legible and to say which one your context privileges — not to pick one and call
it fairness.

**The threshold is yours, and it moves everything.** A decile score becomes "high risk" only
once someone draws a line. Every error-rate figure depends on where.

## Failure modes this project hit

**Treating the recorded label as the ground truth it is named after.** Re-arrest is not
reoffending. Base rates measured through a policing process may themselves be biased — and
base rates are what drive the impossibility result. No analysis of this data can separate the
two, and the limitation belongs in the same breath as the finding.
""",
    "11_pipeline_dag": """
## When to use this

Someone has proposed parallelising a pipeline, adding CI runners, or adopting an
orchestrator, and you would like to know in advance whether it will help.

## Procedure

### 1. Write the graph down before optimising anything

Tasks and their real data dependencies. Not the order you happen to run them in — the order
they *must* run in. The difference between those two is the parallelism you are looking for.

### 2. Sort with Kahn's algorithm, not depth-first

DFS gives you one valid linear order and discards the width information. Kahn's algorithm
peels off layers of zero in-degree, and each layer's width is exactly how many tasks can run
at once. That is the number a scheduler consumes.

### 3. Compute the critical path first — it bounds everything

Longest-path relaxation over the topological order. Speedup can never exceed
sequential ÷ critical path, at any worker count, with any scheduler. Compute this number
before writing any scheduling code: it frequently ends the discussion.

### 4. Measure task durations, and look at their distribution

If one task dominates the critical path, no orchestration will help and the only useful moves
are making that task faster or caching it. This is common and rarely checked.

### 5. Normalise your idle metric

Raw idle worker-seconds grow with fleet size and read as a worsening problem when nothing has
changed. Report idle ÷ available capacity so the numbers are comparable across worker counts.

### 6. Make cycle detection report the cycle

Tri-colour DFS: an edge into a grey (in-progress) node closes a cycle, and the stack between
them is the cycle. `CycleError: cycle detected` on a hundred-node graph is barely better than
silence.

### 7. Chain cache fingerprints through dependencies

fingerprint(t) = H(content(t) ‖ fingerprints of dependencies). Invalidation becomes transitive
by construction — no separate graph walk, no bookkeeping to get wrong. Verify it by
perturbing a root node and confirming everything downstream invalidates.

## Decision points requiring judgement

**Level-synchronous scheduling is pessimistic.** A work-stealing executor starts a task the
moment its own dependencies finish rather than waiting for its whole level. Your measured
speedups are a lower bound — but the critical-path ceiling binds both.

**Durations are machine-specific.** Move to different hardware and the critical path may run
through entirely different tasks. The method transfers; the seconds do not.

## Failure modes this project hit

**Reporting a speedup without its ceiling.** 1.50× sounds like a poor result until you know
the maximum is 1.51×, at which point it is near-optimal. A scheduling number without the
ceiling beside it cannot be interpreted in either direction.

**Fingerprints that cover code but not the world.** A task reading a clock, an environment
variable or a network resource can produce different output under an unchanged fingerprint,
and the cache will serve the stale entry. Know that this is the failure mode you have
accepted.
""",
    "12_market_backtest": """
## When to use this

You are evaluating a model whose predictions would drive sequential decisions over time —
trading is the canonical case, but this applies to any forward-looking model where being
wrong is expensive and the temptation to flatter yourself is strong.

## Procedure

### 1. Check stationarity before choosing a target

Run an ADF test on both the level and the difference. If the level has a unit root, modelling
it directly produces enormous R² values that measure nothing but the target's own persistence.
This is the single most common way results in this domain are overstated.

The diagnostic: restate your model as a forecast of the *change*. If R² collapses from 0.97 to
0.00, you were never predicting anything.

### 2. Measure signal before you measure money

Compute the information coefficient — rank correlation between prediction and realised
outcome — with a p-value, before building any decision rule. An equity curve derived from
predictions with no measurable signal is arithmetic on noise, and it will occasionally look
excellent by chance.

### 3. Purge overlapping labels out of the CV boundary

A k-day forward return computed daily means consecutive rows share k−1 days of outcome.
Adjacent rows are nearly the same observation. Chronological ordering alone does not fix this:
remove a band of rows around each train/test boundary.

### 4. Prove the absence of look-ahead, do not assert it

Static rules cannot see through a `.pipe()` boundary or a helper function. Perturb the future
instead: multiply all inputs from row N onward by some factor, rebuild the entire feature
matrix, and assert that no feature value at any row before N changed. Maximum drift should be
exactly zero.

Bake this into the pipeline so it runs on every execution, not once during development.

### 5. Charge costs, and report gross beside net

State the cost assumption explicitly and compute the break-even: a strategy trading n times a
year at c bps needs n·c of gross alpha before it earns anything. At daily frequency costs are
usually the whole result.

### 6. Benchmark against doing nothing

Not against zero. Buy-and-hold, or the constant prediction, or the current process. A strategy
that returns 8% in a year the benchmark returned 20% lost.

### 7. Publish the result you got

This is the hard step, and it is the one that makes every step above worth doing.

## Decision points requiring judgement

**How many models did you try?** Every additional model family and feature set is another
hypothesis. If something comes out significant, the p-value needs correcting for all of them —
including the ones you abandoned.

**Instrument selection is a hindsight decision.** Anything with a long liquid history survived
to be chosen. That biases the benchmark upward and makes it harder to beat, which is worth
stating explicitly rather than hoping nobody notices.

## Failure modes this project hit

**A stunning R² that was the target's non-stationarity.** Yesterday's price predicts today's
price with R² ≈ 0.98. The identical model as a return forecast scores ≈ 0.00.

**Fold-to-fold variance wide enough to support any conclusion.** Directional accuracy swung
across a 30-point range between folds of the same model. Any single fold could have been
quoted as a triumph. Fixing the protocol in advance is what makes the average meaningful.
""",
}


def main() -> None:
    for project, (slug, tagline) in SKILLS.items():
        directory = ROOT / "projects" / project / "skills" / slug
        directory.mkdir(parents=True, exist_ok=True)

        provenance = load(project, "provenance")
        datasets = ", ".join(d["title"] for d in provenance["datasets"])

        content = f"""---
name: {slug}
description: {tagline}
---

# {tagline}

*Derived from [`projects/{project}`](../../), which applied this procedure to: {datasets}.*

This is a **transferable procedure**, not instructions for rerunning the script in this
repository. It states what to do on a new dataset, where the judgement calls are, and which
failure modes this project actually hit.
{BODIES[project]}
---

## Reproduce the reference implementation

```bash
export PYTHONPATH=lib
python3 projects/{project}/pipeline/build.py
python3 tools/audit.py
```

Results, with their run provenance, land in
[`projects/{project}/artifacts/`](../../artifacts/).
"""
        (directory / "SKILL.md").write_text(content)
        print(f"  projects/{project}/skills/{slug}/SKILL.md  ({len(content) / 1024:.1f} KB)")

    print(f"\nGenerated {len(SKILLS)} skill guides.")


if __name__ == "__main__":
    main()
