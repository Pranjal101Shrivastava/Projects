# 10 · Fairness Audit of COMPAS

An audit of the criminal-risk score ProPublica investigated, run against every major fairness
criterion on **6,172 real Broward County defendants** — and
a demonstration, computed on this data rather than cited, that the criteria **cannot all hold
at once**.

| Dataset | Kind | Size | Origin | Licence |
|---|:---:|---|---|---|
| [COMPAS Recidivism Risk Scores (Broward County, 2013-2014)](https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv) | 🟢 **REAL** | 7,214 defendants | Broward County, Florida criminal records joined to COMPAS risk scores, obtained by ProPublica under a public records request and published alongside their 2016 'Machine Bias' investigation. Each row is a real defendant: their COMPAS decile score, demographics, prior offences, and whether they were in fact rearrested within two years. | Released publicly by ProPublica for independent scrutiny (github.com/propublica/compas-analysis). |

## Both sides of the argument are in the same table

ProPublica said COMPAS was biased. Northpointe, its vendor, said it was not. This audit
computes both claims from one contingency table per group.

| Group | n | Reoffended | Scored high risk | FPR | FNR | PPV | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| African-American | 3,175 | 52.3% | 57.6% | **42.3%** | 28.5% | 65.0% | 64.9% |
| Caucasian | 2,103 | 39.1% | 33.1% | **22.0%** | 49.6% | 59.5% | 67.2% |
| Hispanic | 509 | 37.1% | 27.7% | **19.4%** | 58.2% | 56.0% | 66.2% |

**ProPublica's claim, measured:** among defendants who did *not* reoffend within two years,
African-American defendants were labelled high risk 42.3% of the time against
19.4% for Hispanic — a gap of 0.2296,
2.19×. A χ² test on the two largest groups gives χ² = 128.7,
p = 8.0e-30. Not noise.

**Northpointe's reply, measured:** among defendants labelled high risk, the share who
reoffended ranges 56.0%–65.0% across groups — a gap of
0.0892. The label means close to the same thing whoever receives it.

**Both are true.** The rest of this document is about why that is not a contradiction.

### Accuracy is the column to ignore

Accuracy varies by only 2.3 percentage points across groups — which is
exactly why a vendor can quote it truthfully while a journalist is also right. A single
accuracy number averages a false positive and a false negative into one figure, and here those
two errors fall on different people.

## Four criteria, ranked by distance from parity

| Rank | Criterion | Gap | Ratio | Range across groups | Within 0.05 tolerance |
|---:|---|---:|---:|---|:---:|
| 1 | **Predictive parity (PPV)** | 0.0892 | 1.16× | 0.560 (Hispanic) → 0.649 (African-American) | no |
| 2 | **Predictive equality (FPR parity)** | 0.2296 | 2.19× | 0.194 (Hispanic) → 0.423 (African-American) | no |
| 3 | **Equal opportunity (FNR parity)** | 0.2972 | 2.04× | 0.285 (African-American) → 0.582 (Hispanic) | no |
| 4 | **Demographic parity** | 0.2991 | 2.08× | 0.277 (Hispanic) → 0.576 (African-American) | no |

0 of 4 criteria are satisfied at a 0.05
tolerance — and reporting only that would have been the worse answer.
At a 5-point tolerance every criterion is rejected, which is true but uninformative: it erases the distinction the whole dispute rested on. The gaps are not comparable in size. calibration_ppv is off by 0.0892 while demographic_parity is off by 0.2991 — a factor of 3.4. Northpointe's defence was that calibration nearly holds; ProPublica's case was that error rates emphatically do not. Both readings are visible here, and a pass/fail table alone would hide the one that favours the developer.

The furthest-from-parity criterion (Demographic parity,
gap 0.2991) is
**3.35× further out** than the closest
(Predictive parity (PPV), gap
0.0892). A flat column of four "violated" verdicts would erase
precisely the distinction the public argument was about.

## The impossibility, verified rather than cited

    FPR = (p/(1-p)) · ((1-PPV)/PPV) · (1-FNR)

| Group | Base rate *p* | Observed FPR | FPR the identity forces | Discrepancy |
|---|---:|---:|---:|---:|
| African-American | 0.5231 | 0.4234 | 0.4233 | 6.0e-05 |
| Caucasian | 0.3909 | 0.2201 | 0.2202 | 7.0e-05 |
| Hispanic | 0.3713 | 0.1938 | 0.1937 | 7.0e-05 |

Maximum discrepancy across all groups: **7.0e-05** — the residual
is floating-point rounding, not slack.

Observed base rates differ by 0.1518 between groups. The identity above is an algebraic fact, not a modelling assumption: it holds on this data to within 0.00007. With PPV held equal across groups and p differing, FPR and FNR are forced apart. No amount of retraining, reweighting or threshold tuning escapes this — it is arithmetic. Any tool applied to groups with different base rates must violate either calibration or error-rate equality.

Base rates differ by 0.1518 between groups. While that holds, equal PPV and
equal FPR cannot both be achieved — not by COMPAS, not by a better model, not by any scoring
rule whatsoever. This is Kleinberg et al. (2016) and Chouldechova (2017), checked against real
counts instead of quoted.

## Would dropping race from the model fix it?

A replacement model was trained on 8 features
with `race` **deliberately excluded**.

| Group | COMPAS FPR | Own model FPR | COMPAS PPV | Own model PPV |
|---|---:|---:|---:|---:|
| African-American | 0.4234 | **0.3079** | 0.6495 | 0.7187 |
| Caucasian | 0.2201 | **0.1654** | 0.5948 | 0.5759 |
| Hispanic | 0.1938 | **0.1975** | 0.5603 | 0.6522 |

- FPR gap, COMPAS: **0.2296**
- FPR gap, own model (race never seen): **0.1425**

The replacement model never sees race, yet its false-positive rates still differ by 0.1425 across groups, against 0.2296 for COMPAS. Removing a protected attribute does not remove disparity, because prior convictions and age carry the same information. 'Fairness through unawareness' does not work, and this is what that looks like measured rather than asserted.

The model's own quality is reported beside its no-skill floor, as everywhere else in this
repository: PR-AUC 0.6948 against a prevalence floor of
0.4552, Brier 0.2081. And the accuracy
trap, stated so it cannot be quoted out of context: the model scores
62.5% accuracy where predicting "will not
reoffend" for everybody scores
54.5%.

## Who was excluded from the comparison, and why that matters

7,214 raw records reduce to 6,172 under
ProPublica's published filters. Groups below 500 people —
Other (n=343), Asian (n=31), Native American (n=11) — are
excluded from every comparison rather than reported with intervals too wide to mean anything.
That exclusion is itself a choice with consequences: the smallest groups are the ones least
likely to be audited anywhere.

## Screens

![10 fairness](../../docs/screenshots/10_fairness.png)

![10 fairness method](../../docs/screenshots/10_fairness_method.png)


## CRISP-DM record

> **Business question.** A risk score used in real sentencing and bail decisions predicts reoffending. Is it fair — and does that question even have a single answer?

### Business Understanding

COMPAS scores informed real bail and sentencing decisions in Broward County. ProPublica reported in 2016 that its errors fell unevenly by race; Northpointe replied that the scores were equally calibrated and therefore fair. Both claims are arithmetically true of the same data. The objective here is not to determine who was right but to measure every criterion and show why they cannot be satisfied simultaneously.

**Which fairness definition does this audit adopt?**

- **Chose:** None. All are measured; none is declared correct.
- **Why:** Kleinberg et al. (2016) and Chouldechova (2017) proved that calibration within groups and equal false-positive/false-negative rates are mutually incompatible whenever base rates differ, except in degenerate cases. No model can satisfy both. Selecting one is a judgement about which harm matters more — a wrongly detained person who would not have reoffended, or a wrongly released person who would have — and that is not a decision a pipeline can make.
- *Rejected:* Pick equalized odds and declare the tool unfair — defensible, but presents a value judgement as a measurement.
- *Rejected:* Pick calibration and declare it fair — the same error in the other direction.
- *Rejected:* Report a single 'fairness score' — collapses incompatible criteria into one number and hides the trade-off that is the entire issue.

**How is data about real individuals handled?**

- **Chose:** Names dropped on load; only aggregate group statistics reported.
- **Why:** These are real people with real criminal records. ProPublica published the data for scrutiny of the algorithm, not of its subjects. No individual is identified anywhere in the output.

### Data Understanding

7,214 Broward County records reduced to 6,172 by ProPublica's own published filters, applied so these results are directly comparable to theirs. 3 groups have at least 500 records and are compared; smaller groups are reported but not compared, because rate estimates on a few dozen people are too unstable to support a fairness claim.

**Limitations**

- The outcome variable is *rearrest*, not reoffending. Policing intensity differs by neighbourhood and by group, so arrest rates reflect both behaviour and enforcement. Every 'error rate' below therefore measures the tool against a target that is itself shaped by the system being audited. This affects every published analysis of this dataset, including ProPublica's and Northpointe's.
- Two years in one Florida county in 2013-2014. Nothing here establishes how COMPAS behaves elsewhere or now.

### Data Preparation

ProPublica's published filters applied verbatim so results are comparable to the original analysis. Identifying columns dropped before any analysis. High risk defined as decile score >= 5, matching ProPublica's cut.

### Modeling

No model is built to replace COMPAS as the object of the audit — COMPAS's own published scores are audited directly. A separate logistic model excluding race is trained solely to test whether omitting a protected attribute removes disparity.

**Does excluding race from the features make a model fair?**

- **Chose:** No — measured, not assumed.
- **Why:** The replacement model never sees race, yet its false-positive rates still differ by 0.1425 across groups, against 0.2296 for COMPAS. Removing a protected attribute does not remove disparity, because prior convictions and age carry the same information. 'Fairness through unawareness' does not work, and this is what that looks like measured rather than asserted.
- *Rejected:* Assume 'fairness through unawareness' works — the most common intuition, and demonstrably false on this data.

### Evaluation

Of four fairness criteria, 0 are satisfied and 4 are violated. That split is not a defect in the audit — Chouldechova's identity, verified on this data to within 0.00007, shows the criteria are mathematically incompatible once base rates differ.

**Limitations**

- This audit cannot say whether COMPAS is fair, because 'fair' is not a single measurable property. It can say precisely which criteria hold and which do not, and why no tool can satisfy all of them here.
- Rearrest is the proxy for reoffending. If policing differs by group, the 'ground truth' is itself affected by the disparity under investigation, and every error rate inherits that.
- Thresholding a decile score at 5 is ProPublica's choice, adopted for comparability. A different cut changes every rate reported here, though not the impossibility result, which holds at any threshold.

### Deployment

The audit ships as a dashboard where a reader can select a fairness criterion and see which groups it favours, with the impossibility identity shown alongside so no criterion can be read as the answer.


## Run it

```bash
git clone https://github.com/Pranjal101Shrivastava/Projects
cd Projects
pip install -r requirements.txt
export PYTHONPATH=lib

python3 projects/10_fairness_audit/pipeline/build.py   # rebuilds every artifact below
python3 tools/audit.py                          # static leakage audit
```

Datasets download on first run into `.data/` and are verified against their recorded
SHA-256 on every run thereafter. The pipeline is seeded, so a rerun at the same commit
reproduces the same numbers.

---

**Live:** [https://pranjal101shrivastava.github.io/Projects/#/p/fairness](https://pranjal101shrivastava.github.io/Projects/#/p/fairness) *(requires GitHub Pages enabled)* ·
**Method:** [`pipeline/build.py`](./pipeline/build.py) ·
**Audit:** [`audit.md`](./audit.md) ·
**Artifacts:** [`artifacts/`](./artifacts/)

*Generated at commit `200f444` · seed 42 · 0.3s · Python 3.11.15 · numpy 2.4.6 · pandas 3.0.5 · sklearn 1.9.1 · lightgbm 4.7.0 · torch 2.14.0+cu130*
