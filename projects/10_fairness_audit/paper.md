# Four Fairness Criteria, One Risk Score, and an Impossibility

*An audit of COMPAS on 6,172 Broward County defendants*

## 1. Background

In 2016 ProPublica reported that COMPAS, a proprietary recidivism-risk instrument used in
pretrial and sentencing decisions, produced substantially higher false-positive rates for Black
defendants. Northpointe replied that the instrument was calibrated: a given score carried the
same meaning across groups. Both analyses were competent and both conclusions were correct.

This study reproduces both, then shows why they had to coexist.

## 2. Data and filtering

| Step | Records |
|---|---:|
| Raw COMPAS release | 7,214 |
| After ProPublica screening filters | 6,172 |

Filters applied:

- `days_b_screening_arrest within [-30, 30]` — charge and screening must correspond
- `is_recid != -1` — drop records with no recidivism data
- `c_charge_degree != 'O'` — drop ordinary traffic offences, which carry no jail time
- `score_text != 'N/A'` — drop rows with no COMPAS score

Group sizes:

- African-American: 3,175
- Caucasian: 2,103
- Hispanic: 509
- Other: 343  *(excluded — below the 500 minimum)*
- Asian: 31  *(excluded — below the 500 minimum)*
- Native American: 11  *(excluded — below the 500 minimum)*

Identifying columns (`name`, `first`, `last`) were
dropped before any modelling.

## 3. Per-group performance

| Group | n | TP | FP | FN | TN | Base rate | FPR | FNR | PPV | NPV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| African-American | 3,175 | 1,188 | 641 | 473 | 873 | 0.5231 | 0.4234 | 0.2848 | 0.6495 | 0.6486 |
| Caucasian | 2,103 | 414 | 282 | 408 | 999 | 0.3909 | 0.2201 | 0.4964 | 0.5948 | 0.7100 |
| Hispanic | 509 | 79 | 62 | 110 | 258 | 0.3713 | 0.1938 | 0.5820 | 0.5603 | 0.7011 |

## 4. Criteria

| Rank | Criterion | Definition | Gap | Ratio |
|---:|---|---|---:|---:|
| 1 | Predictive parity (PPV) | Among those labelled high risk, P(reoffends) is equal across groups. | 0.0892 | 1.16× |
| 2 | Predictive equality (FPR parity) | False positive rate is equal across groups. | 0.2296 | 2.19× |
| 3 | Equal opportunity (FNR parity) | False negative rate is equal across groups. | 0.2972 | 2.04× |
| 4 | Demographic parity | P(predicted high risk) is equal across groups. | 0.2991 | 2.08× |

Demographic parity additionally fails the four-fifths rule: disparate impact ratio
0.4808 against a 0.80 threshold.

Each criterion carries a recorded interpretation:

**Predictive parity (PPV).** If satisfied, a 'high risk' label means the same thing whoever receives it. This is the criterion Northpointe's defence centred on.

**Predictive equality (FPR parity).** Concerns people who did NOT reoffend but were scored high risk — detained or sentenced more harshly for something they would not have done. This is the criterion ProPublica's analysis centred on.

**Equal opportunity (FNR parity).** Concerns people who did reoffend but were scored low risk. Unequal FNR means one group is more often released when they should not have been.

**Demographic parity.** Demographic parity ignores the outcome entirely. If two groups truly differ in base rate, enforcing it requires deliberately mis-scoring people — so failing this test is not on its own evidence of unfairness.

## 5. The impossibility result

For any classifier, within any group:

    FPR = (p / (1 − p)) · ((1 − PPV) / PPV) · (1 − FNR)

where p is the group's base rate. This is an algebraic identity, not an empirical finding.
Evaluated on the observed counts:

| Group | p | Observed FPR | Implied FPR | Discrepancy |
|---|---:|---:|---:|---:|
| African-American | 0.5231 | 0.4234 | 0.4233 | 6.0e-05 |
| Caucasian | 0.3909 | 0.2201 | 0.2202 | 7.0e-05 |
| Hispanic | 0.3713 | 0.1938 | 0.1937 | 7.0e-05 |

Maximum discrepancy 7.0e-05.

Observed base rates differ by 0.1518 between groups. The identity above is an algebraic fact, not a modelling assumption: it holds on this data to within 0.00007. With PPV held equal across groups and p differing, FPR and FNR are forced apart. No amount of retraining, reweighting or threshold tuning escapes this — it is arithmetic. Any tool applied to groups with different base rates must violate either calibration or error-rate equality.

## 6. Fairness through unawareness

A gradient-boosted replacement model was trained on
8 features
(`age`, `priors_count`, `juv_fel_count`, `juv_misd_count`, `juv_other_count`, `sex`, `c_charge_degree`, `age_cat`) with `race`
excluded.

Held-out performance: PR-AUC 0.6948 against a no-skill floor of
0.4552 (lift 1.53×),
ROC-AUC 0.7344, Brier 0.2081 on
1,852 held-out defendants.

| Group | COMPAS FPR | Own model FPR | COMPAS PPV | Own model PPV |
|---|---:|---:|---:|---:|
| African-American | 0.4234 | **0.3079** | 0.6495 | 0.7187 |
| Caucasian | 0.2201 | **0.1654** | 0.5948 | 0.5759 |
| Hispanic | 0.1938 | **0.1975** | 0.5603 | 0.6522 |

The replacement model never sees race, yet its false-positive rates still differ by 0.1425 across groups, against 0.2296 for COMPAS. Removing a protected attribute does not remove disparity, because prior convictions and age carry the same information. 'Fairness through unawareness' does not work, and this is what that looks like measured rather than asserted.

## 7. Statistical significance

Tests whether the FPR difference between the two largest groups could plausibly arise by chance. It does not test whether the difference is unjust — that is not a statistical question.

χ² = 128.67, p = 8.00e-30 for false positive rate between
African-American vs Caucasian.

## 8. Limitations

- The outcome variable is *rearrest*, not reoffending. Policing intensity differs by neighbourhood and by group, so arrest rates reflect both behaviour and enforcement. Every 'error rate' below therefore measures the tool against a target that is itself shaped by the system being audited. This affects every published analysis of this dataset, including ProPublica's and Northpointe's.
- Two years in one Florida county in 2013-2014. Nothing here establishes how COMPAS behaves elsewhere or now.
- This audit cannot say whether COMPAS is fair, because 'fair' is not a single measurable property. It can say precisely which criteria hold and which do not, and why no tool can satisfy all of them here.
- Rearrest is the proxy for reoffending. If policing differs by group, the 'ground truth' is itself affected by the disparity under investigation, and every error rate inherits that.
- Thresholding a decile score at 5 is ProPublica's choice, adopted for comparability. A different cut changes every rate reported here, though not the impossibility result, which holds at any threshold.

*Generated at commit `200f444` · seed 42 · 0.28s · Python 3.11.15 · numpy 2.4.6 · pandas 3.0.5 · sklearn 1.9.1 · lightgbm 4.7.0 · torch 2.14.0+cu130*
