# Video 10 · Fairness Audit · COMPAS

**Estimated runtime:** ~7:10 at a measured pace (969 spoken words)
**Live page:** https://pranjal101shrivastava.github.io/Projects/#/p/fairness
**Project directory:** [`projects/10_fairness_audit/`](../../projects/10_fairness_audit/)

> **The one sentence this video has to land:** Both ProPublica and Northpointe were right — FPR 42.3% versus 19.4% alongside near-equal PPV — and the identity forcing that is verified on the real counts to 7e-05.

## Before you record

- Open **https://pranjal101shrivastava.github.io/Projects/#/p/fairness**.
- Switch the group selector on camera during the contingency-table section.
- This is the most sensitive subject in the portfolio — read the limitations beat slowly and do not cut it.

Everything in plain text below is meant to be **spoken**. Lines marked *On screen* and
*Note to self* are directions — do not read them aloud.

---

## [0:00] Cold open

> **On screen —** the per-group metrics table

In 2016 ProPublica published an investigation of COMPAS — a risk score used in American
courtrooms to inform bail and sentencing — and concluded it was biased against Black
defendants.

Northpointe, the company behind it, replied that the score was calibrated and therefore was
not biased.

That reads like a factual dispute. One of them must be wrong.

Neither of them was. And in this video I'm going to show you the arithmetic that makes both
statements true at the same time — computed on the real data, not quoted from a paper.

## [0:40] The data

> **On screen —** the Data provenance tab

This is ProPublica's own COMPAS release for Broward County, Florida.
7,214 records, reduced to 6,172 by their
published screening filters — which I've applied as they specified rather than inventing my
own.

I'm restricting comparisons to the 3 groups with at least
500 members. The smaller groups —
Other at 343, Asian at 31, Native American at 11
— are excluded, and reported as excluded, because error rates on a handful of people come
with intervals too wide to interpret.

I want to name that exclusion as a finding in itself: the smallest groups are the ones least
likely to be audited anywhere, and that's a structural gap, not housekeeping.

## [1:28] ProPublica's claim, measured

> **On screen —** the FPR column

Here's the first claim. Among defendants who did **not** go on to reoffend within two
years — people the tool got wrong — how often were they labelled high risk?

African-American defendants: 42.3%.

Hispanic defendants: 19.4%.

That's a ratio of 2.19 to one. A chi-squared test on the two largest groups
gives 129, with a p-value around 8e-30.

That is not sampling noise. ProPublica's claim is correct.

## [1:58] Northpointe's reply, also measured

> **On screen —** the PPV column

Now the other claim. Among the people the tool labelled high risk, how many actually
reoffended?

The range across groups is 56.0% to 65.0%. A gap of
0.089.

So a "high risk" label means close to the same thing whoever receives it. That's what
calibration means, and on this data it largely holds.

Northpointe's claim is also correct.

Both tables come from the same 6,172 rows. Nobody was
lying. Nobody made an arithmetic error.

## [2:33] Why both can be true

> **On screen —** the impossibility section

Here's the resolution, and it's algebra.

For any classifier at all, within any group, this identity holds:

False positive rate equals — p over one minus p — times — one minus PPV, over PPV — times
one minus the false negative rate.

Where p is that group's base rate: the share who actually did reoffend.

I didn't want to cite that. I wanted to check it. So I computed the right-hand side from the
observed counts and compared it to the observed false positive rate.

Maximum discrepancy across every group: **7.0e-05**.

That's floating-point rounding. The identity holds exactly.

Now read it again. If two groups have different base rates — and here they differ by
0.1518 — then holding PPV equal **forces** the false positive rates to
differ.

Not because of bad data. Not because of a biased vendor. Not because of a fixable modelling
choice. Because of arithmetic.

You can equalise error rates, or you can equalise predictive value. You cannot do both. That
result is Kleinberg and colleagues in 2016, and Chouldechova in 2017 — and what you're
looking at is it, verified on the actual counts.

## [3:55] The column to ignore

> **On screen —** the accuracy column

Quick aside, because it explains how this dispute stayed alive.

Look at accuracy. It varies by only about 2.3 percentage points across
groups.

So a vendor can report accuracy, truthfully, and sound fair. A single accuracy number
averages a false positive and a false negative into one figure — and here those two errors
land on *different people*. That's precisely the information accuracy destroys.

## [4:24] The obvious fix, tested

> **On screen —** the own-model comparison table

So the first thing everybody proposes: just don't give the model race.

I tested it. I trained a replacement model on 8 features — age,
prior convictions, charge degree, juvenile counts — with race **excluded entirely**. The
model never sees it.

COMPAS's false positive rate gap: 0.2296.

My model's gap, with race never seen:
**0.1425**.

About 38% of the gap closed. The rest
stayed.

Because prior arrest counts carry the information race would have carried. You cannot delete
a variable out of a correlated world. This has a name — fairness through unawareness — and
this is the demonstration that it doesn't work.

And I report my model against its own no-skill floor, same as everywhere else here: PR-AUC
0.6948 against a prevalence floor of
0.4552.

## [5:20] The table I nearly published

> **On screen —** the criteria ranking table

One more thing, about presentation, because I got this wrong first.

My initial criteria table had four rows and one column: VIOLATED, VIOLATED, VIOLATED,
VIOLATED. All four fairness criteria failed at a strict tolerance. Every word of it true.

And completely useless — because it erased the distinction the entire public argument was
about.

Ranked by distance from parity instead:

1. Predictive parity, or calibration — equal meaning of the label — gap 0.0892.
2. Predictive equality — equal false positive rates — gap 0.2296.
3. Equal opportunity — equal false negative rates — gap 0.2972.
4. Demographic parity — equal selection rates — gap 0.2991.

The furthest criterion is 3.35 times further from parity
than the closest one. *That ordering* is the argument. A pass/fail column deletes it.

## [6:16] Close

> **On screen —** the limitations section

And the limitation that has to be said in the same breath as every number above.

All of this treats a **recorded re-arrest within two years** as ground truth for "committed
another crime". Policing is not uniform across neighbourhoods or groups. So the base rates
that drive the impossibility result are themselves measured through a process that may be
biased.

Nothing in this analysis — or in any analysis of this dataset — can separate those two
things. A gap in measured base rates is not proof of a gap in underlying behaviour.

That's not a footnote. It's the boundary of what the data can support.

Next video: I build a DAG scheduler from scratch and then measure that my own pipeline
doesn't need one.
---

*Generated by [`tools/video_scripts.py`](../../tools/video_scripts.py) from committed
artifacts. Every figure spoken here is the figure on screen.*
