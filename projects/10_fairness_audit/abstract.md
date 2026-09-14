# Abstract — An Empirical Audit of COMPAS Against Four Fairness Criteria

**Objective.** Evaluate a deployed criminal-risk instrument against the major group-fairness
criteria simultaneously, and test the algebraic impossibility result against real data rather
than citing it.

**Data.** ProPublica's COMPAS release for Broward County, Florida: 7,214 records
reduced to 6,172 by the published screening filters.
Analysis is restricted to the 3 groups with at least
500 members (African-American, Caucasian, Hispanic). Overall two-year
recidivism rate 45.5%; overall high-risk rate
44.6%.

**Method.** Per-group contingency tables were computed at a decile-≥5 high-risk threshold, and
four criteria evaluated: demographic parity, equal opportunity (FNR parity), predictive
equality (FPR parity), and predictive parity (PPV). Criteria were ranked by absolute distance
from parity rather than reported as binary pass/fail. Chouldechova's identity
FPR = (p/(1−p))·((1−PPV)/PPV)·(1−FNR) was evaluated against observed counts. A replacement
gradient-boosted model excluding race was trained for comparison.

**Results.** False positive rates ranged 0.1938 (Hispanic) to
0.4234 (African-American), a 2.19× ratio (χ² = 128.7,
p = 8.0e-30). Positive predictive value ranged 0.5603–0.6495,
a gap of 0.0892. No criterion was satisfied at a
0.05 tolerance, but distances from parity differed by a factor of
3.35. The identity reproduced observed FPR to within
7.0e-05 for every group. The replacement model, trained without
race, retained an FPR gap of 0.1425 against COMPAS's 0.2296.

**Conclusion.** With base rates differing by 0.1518, equalised error rates
and equal predictive value are mutually exclusive as a matter of algebra; the observed
disagreement between ProPublica and Northpointe is therefore a disagreement about which
criterion to prioritise, not about the facts. Removing the protected attribute from the feature
set reduces but does not eliminate disparate error rates, because correlated features carry the
same information.

**Keywords.** algorithmic fairness, COMPAS, equalised odds, calibration, impossibility theorem,
recidivism prediction
