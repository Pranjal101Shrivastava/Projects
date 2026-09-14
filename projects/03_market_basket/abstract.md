# Abstract — Association Rule Mining with False Discovery Rate Control

**Objective.** Identify grocery item affinities strong enough and statistically robust
enough to justify merchandising changes, and quantify the computational trade-off between
the two canonical frequent-itemset algorithms.

**Data.** 9,835 real point-of-sale transactions over
169 item categories (arules Groceries), mean basket
4.409 items, 97.4% sparse.

**Method.** Apriori and FP-Growth were implemented from first principles and executed on
identical input at a support threshold of 0.1%. Rules were scored on six complementary
measures — support, confidence, lift, leverage, conviction and Zhang's metric — and each
subjected to a one-sided Fisher exact test with Benjamini-Hochberg correction at α =
0.05.

**Results.** Both algorithms returned identical output on all 13,106 frequent
itemsets, with FP-Growth 189× faster
(0.624s against 117.943s) using
2 database scans against
4, and generating no candidate itemsets where Apriori
generated 267,573 of which 72%
were pruned by downward closure. Of 52,285 candidate rules,
5,144 failed FDR control and were excluded. The strongest surviving
affinity reached lift 35.72. Negative associations were also detected, the
strongest reaching Zhang's metric -0.626.

**Conclusion.** Independent implementations agreeing exactly provides stronger correctness
evidence than either executing without error. Multiple-comparisons correction is rarely
applied in association mining despite tens of thousands of simultaneous tests; here roughly
one rule in ten failed it.

**Keywords.** association rules, Apriori, FP-Growth, false discovery rate, Fisher exact test
