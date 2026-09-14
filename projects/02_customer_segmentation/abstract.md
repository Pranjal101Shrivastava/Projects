# Abstract — Customer Segmentation with Bootstrap Stability Validation

**Objective.** Determine whether a bank's contacted customers fall into reproducible
segments, and whether those segments differ enough in subscription behaviour to justify
differentiated targeting.

**Data.** 41,188 real direct-marketing contact records from a Portuguese retail bank
(Moro, Cortez & Rita, 2014), comprising demographic, financial and campaign-history
attributes plus contemporaneous macroeconomic indicators.

**Method.** Subscription outcome and call duration were excluded from the feature space —
the former reserved for external validation, the latter because it is unknowable at scoring
time. Mixed-type features were encoded within a ColumnTransformer. Candidate k from 2 to 10
was evaluated against silhouette, Calinski-Harabasz and Davies-Bouldin indices, then
stress-tested over 30 bootstrap resamples using adjusted Rand index with a
reproducibility threshold of 0.75. Three algorithms with differing
geometric assumptions were compared at the selected k.

**Results.** All three validity indices unanimously favoured k = 2, while
only k ∈ {6, 7} satisfied the stability criterion; the reproducible
solution was adopted. At k = 6, conversion ranged from
4.15% to 63.83% across segments
(χ² = 4786, p < 0.001) despite the outcome being withheld from clustering. A
cohort of 1,592 customers averaging 12.9 contacts
converted at 15.4× below the
highest-converting segment. Cross-algorithm agreement was moderate
(ARI 0.4924 against a Gaussian mixture).

**Conclusion.** Internal validity indices and reproducibility can select different solutions,
and optimising geometry alone risks reporting a partition specific to the sample. Moderate
cross-algorithm agreement further indicates that a substantial share of apparent structure
reflects the algorithm's assumptions rather than the data.

**Keywords.** clustering, bootstrap stability, adjusted Rand index, external validation,
customer segmentation
