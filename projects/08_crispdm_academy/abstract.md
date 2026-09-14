# Abstract — Teaching Statistical Concepts from Real Rather Than Simulated Data

**Objective.** Construct interactive teaching material for six core statistical concepts in
which every figure derives from real observational data, and examine what changes when the
assumptions underlying textbook illustrations are not satisfied.

**Data.** Five real datasets: Titanic passenger records, ULB credit card transactions,
Melbourne daily temperatures, and derived series.

**Method.** Six modules were implemented — Bayes and conditional independence, threshold and
cost trade-offs, gradient descent, backpropagation, bias-variance decomposition, and the
central limit theorem — each computing its figures from observational data. Assumption
violations were quantified rather than assumed away: conditional dependence by Cramér's V,
gradient correctness by central finite differences, bias-variance components by bootstrap
resampling, and normality of sampling distributions by skewness and Shapiro-Wilk.

**Results.** 4 of 5 tested feature
pairs violated conditional independence on Titanic, yet Naive Bayes retained ranking
performance (ROC-AUC 0.837 against
0.844) while showing degraded
calibration. Analytic gradients matched finite differences to 1.4e-08.
The bias-variance curve did not exhibit the canonical U: variance accounted for only
3.6% of test error at maximum capacity, and training
error was non-monotone under bootstrap averaging. Convergence to normality was substantially
slower than the n = 30 heuristic implies, with sample-mean skewness still 2.00
at that size for a population of skewness 17.0.

**Conclusion.** Illustrations drawn from real data differ systematically from their textbook
forms. Those differences constitute teachable content rather than defects in the examples.

**Keywords.** statistical education, bias-variance decomposition, central limit theorem,
gradient checking, conditional independence
