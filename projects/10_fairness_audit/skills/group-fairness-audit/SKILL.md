---
name: group-fairness-audit
description: Audit a deployed scoring system against fairness criteria that cannot all hold
---

# Audit a deployed scoring system against fairness criteria that cannot all hold

*Derived from [`projects/10_fairness_audit`](../../), which applied this procedure to: COMPAS Recidivism Risk Scores (Broward County, 2013-2014).*

This is a **transferable procedure**, not instructions for rerunning the script in this
repository. It states what to do on a new dataset, where the judgement calls are, and which
failure modes this project actually hit.

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

---

## Reproduce the reference implementation

```bash
export PYTHONPATH=lib
python3 projects/10_fairness_audit/pipeline/build.py
python3 tools/audit.py
```

Results, with their run provenance, land in
[`projects/10_fairness_audit/artifacts/`](../../artifacts/).
