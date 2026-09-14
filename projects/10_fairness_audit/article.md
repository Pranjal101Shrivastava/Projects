# ProPublica and Northpointe were both right, and I can show you the arithmetic

*An audit of COMPAS on 6,172 real defendants*

In 2016 ProPublica published an investigation of COMPAS, a risk score used in American
courtrooms, and found that Black defendants who did not go on to reoffend were labelled
high-risk far more often than white defendants who did not.

Northpointe, the company behind it, responded that the score was calibrated: a 7 means the same
probability of reoffending whoever gets it.

This looked like a factual dispute. It was not. Here is the same data, with both claims
computed side by side.

## Claim one: the error rates

| Group | Did not reoffend, labelled high risk |
|---|---:|
| African-American | **42.3%** |
| Caucasian | **22.0%** |
| Hispanic | **19.4%** |

2.19× between the extremes. χ² = 129, p = 8e-30.
This is not a sampling artefact.

## Claim two: the calibration

| Group | Labelled high risk, did reoffend |
|---|---:|
| African-American | 65.0% |
| Caucasian | 59.5% |
| Hispanic | 56.0% |

A spread of 0.089. The label means roughly the same thing whoever
receives it.

Both tables come from the same 6,172 rows. Nobody was
lying.

## Why both can be true

There is an identity that holds for any classifier at all:

    FPR = (p / (1 − p)) · ((1 − PPV) / PPV) · (1 − FNR)

where p is the group's base rate — the share who actually reoffended.

I did not want to cite this. I wanted to check it, so I computed the right-hand side from the
real numbers and compared it to the observed FPR:

| Group | Observed FPR | What the identity forces | Difference |
|---|---:|---:|---:|
| African-American | 0.5231 | 0.4234 | 0.4233 | 6.0e-05 |
| Caucasian | 0.3909 | 0.2201 | 0.2202 | 7.0e-05 |
| Hispanic | 0.3713 | 0.1938 | 0.1937 | 7.0e-05 |

Maximum difference across every group: **7.0e-05**. That is
rounding.

Now read the identity again. If two groups have different base rates — here they differ by
0.1518 — then holding PPV equal *forces* FPR to differ. Not because of bad
data, or a biased vendor, or a fixable modelling choice. Because of algebra.

You can have equal error rates, or you can have equal predictive value. Not both.

## So I tried the obvious fix

Everyone's first instinct: just don't give the model race.

I trained a gradient-boosted model on
8 features — age, priors, charge degree,
juvenile counts — with race **excluded entirely**.

| | FPR gap between groups |
|---|---:|
| COMPAS | 0.2296 |
| My model, race never seen | **0.1425** |

38% of the gap closed. The rest stayed.

The replacement model never sees race, yet its false-positive rates still differ by 0.1425 across groups, against 0.2296 for COMPAS. Removing a protected attribute does not remove disparity, because prior convictions and age carry the same information. 'Fairness through unawareness' does not work, and this is what that looks like measured rather than asserted.

Prior arrest counts carry the information race would have carried. You cannot delete a variable
out of a correlated world.

## The number I nearly published instead

My first version of the criteria table had four rows and one column: VIOLATED, VIOLATED,
VIOLATED, VIOLATED. All true. All useless.

At a 5-point tolerance every criterion is rejected, which is true but uninformative: it erases the distinction the whole dispute rested on. The gaps are not comparable in size. calibration_ppv is off by 0.0892 while demographic_parity is off by 0.2991 — a factor of 3.4. Northpointe's defence was that calibration nearly holds; ProPublica's case was that error rates emphatically do not. Both readings are visible here, and a pass/fail table alone would hide the one that favours the developer.

Ranked by distance from parity instead:

| Rank | Criterion | Gap |
|---:|---|---:|
| 1 | Predictive parity (PPV) | 0.0892 |
| 2 | Predictive equality (FPR parity) | 0.2296 |
| 3 | Equal opportunity (FNR parity) | 0.2972 |
| 4 | Demographic parity | 0.2991 |

3.4× between the closest and the furthest. That ordering is
the entire public argument, and a pass/fail column erases it.

## What this audit cannot tell you

Every number here treats **a recorded re-arrest within two years** as ground truth for
"committed another crime". Policing is not uniform. The base rates that drive the impossibility
result are themselves measured through a process that may be biased — and nothing in this
analysis, or any analysis of this dataset, can separate the two.

That is not a footnote. It is the limit of what the data can support, and it belongs in the
same breath as the result.

---

**Live demo:** [https://pranjal101shrivastava.github.io/Projects/#/p/fairness](https://pranjal101shrivastava.github.io/Projects/#/p/fairness)
**Code:** [github.com/Pranjal101Shrivastava/Projects](https://github.com/Pranjal101Shrivastava/Projects)
