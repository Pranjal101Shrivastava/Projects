# Abstract — Fraud Detection under Extreme Class Imbalance

**Objective.** Rank card transactions for manual review under a fixed analyst capacity, and
establish which evaluation metrics remain informative at a positive rate below 0.2%.

**Data.** 284,807 real transactions by European cardholders over
48 hours in September 2013 (Université Libre de Bruxelles), of
which 492 are fraudulent (0.1727%). Features V1–V28 are
principal components published in place of the original variables for confidentiality.

**Method.** Partitioning was chronological rather than random, because fraud is bursty and a
random split distributes a single compromised-card episode across both partitions. Three
detectors spanning two supervision regimes were compared: an Isolation Forest fitted only on
legitimate training transactions, a class-weighted logistic regression, and gradient boosting
under four distinct reweighting strategies. No synthetic minority oversampling was used.
Operating points were selected by expected cost under an assumed
500:10
false-negative to false-positive ratio.

**Results.** The best model reached PR-AUC 0.7657 against a no-skill floor of
0.00132 — the prevalence of the held-out window —
(580× lift). The unsupervised
detector attained ROC-AUC 0.9394 while achieving PR-AUC 0.0366 and
precision 6.2%, illustrating the divergence between the two metrics under
imbalance. A reweighting ablation found that setting `scale_pos_weight` to the
negative/positive ratio — standard practice — reduced gradient-boosting PR-AUC from
0.7359 to
0.0092.

**Conclusion.** ROC-AUC is unsuitable as a headline metric at this prevalence. Standard
reweighting guidance derived from linear models does not transfer to high-capacity boosted
ensembles with few positives, and should be verified by ablation rather than assumed.

**Keywords.** imbalanced classification, precision-recall AUC, cost-sensitive learning,
anomaly detection, temporal validation
