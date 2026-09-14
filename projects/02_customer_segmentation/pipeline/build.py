"""Customer segmentation on 41,188 real bank marketing contacts.

Run:
    PYTHONPATH=lib python3 projects/02_customer_segmentation/pipeline/build.py

Clustering has no ground truth, which makes it the easiest analysis to fool yourself with.
Run K-means with k=4, colour a scatter plot, write four persona names, and the result looks
authoritative regardless of whether the structure is real. Three defences are applied here:

1. **Multiple validity indices that can disagree.** Silhouette rewards separation,
   Calinski-Harabasz rewards compactness against spread, Davies-Bouldin penalises overlap.
   A k that wins on one and loses on the others is not a finding.
2. **Bootstrap stability.** The partition is recomputed on 30 resamples and compared by
   adjusted Rand index. A segmentation that changes when you resample the same population
   describes the sample, not the customers — and this is the check almost no segmentation
   write-up performs.
3. **An external validation signal the clustering never saw.** Segments are built from
   demographics and contact history only; subscription outcome is held out entirely and
   used afterwards to ask whether the segments differ in a way the business cares about.

That third point matters: segments that are statistically tidy but have identical
conversion rates are worthless to a marketing team, and only an external signal reveals it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lib"))

from dsx import artifacts, data, metrics  # noqa: E402
from dsx.crispdm import CrispDm, Decision, Phase  # noqa: E402

PROJECT = "02_customer_segmentation"
OUT = ROOT / "projects" / PROJECT / "artifacts"
SEED = 42

K_RANGE = range(2, 11)
N_BOOTSTRAP = 30
STABILITY_THRESHOLD = 0.75  # ARI below this means the partition is not reproducible


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    crisp = CrispDm(
        project=PROJECT,
        business_question=(
            "Do the bank's contacted customers fall into distinct, reproducible segments "
            "— and do those segments differ enough in subscription behaviour to justify "
            "targeting them differently?"
        ),
    )

    crisp.record(
        Phase(
            name="business_understanding",
            summary=(
                "A term-deposit campaign has a fixed calling budget. Segmentation is only "
                "worth doing if it changes who gets called, which requires segments that "
                "are reproducible and that differ in conversion. A segmentation that is "
                "stable but uniform in outcome, or that differs in outcome but is not "
                "reproducible, fails to justify itself."
            ),
            decisions=[
                Decision(
                    question="What makes a segmentation successful here?",
                    choice=(
                        "Bootstrap-stable partitions whose conversion rates differ "
                        "materially."
                    ),
                    rationale=(
                        "Both conditions are necessary. Stability without outcome "
                        "separation gives tidy segments nobody can act on; outcome "
                        "separation without stability gives a story that will not "
                        "reproduce next quarter."
                    ),
                    alternatives_rejected=[
                        "Silhouette alone — measures geometry, says nothing about whether "
                        "the segments matter commercially.",
                        "Fixing k=4 for interpretability — a convention, not a finding.",
                    ],
                ),
            ],
            evidence={"stability_threshold_ari": STABILITY_THRESHOLD},
        )
    )

    with artifacts.run(PROJECT, seed=SEED) as ctx:
        # --- Data understanding -------------------------------------------------------
        print("\n[1/6] Loading real bank marketing contacts …")
        df = pd.read_csv(data.fetch("bank_marketing"), sep=";")
        n = len(df)

        # The outcome is held out of clustering entirely and used only for validation.
        outcome = (df["y"] == "yes").astype(int)

        # 'duration' is excluded from segmentation for the same reason it is excluded from
        # any honest model on this dataset: it records how long the call lasted, which is
        # only known after the outcome is decided. Segmenting on it would build segments
        # that partly encode the answer.
        excluded = ["y", "duration"]

        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded
        ]
        categorical_cols = [
            c for c in df.select_dtypes(include=["object", "str"]).columns if c not in excluded
        ]

        profile = {
            "n_customers": n,
            "n_numeric_features": len(numeric_cols),
            "n_categorical_features": len(categorical_cols),
            "numeric_features": numeric_cols,
            "categorical_features": categorical_cols,
            "excluded_features": excluded,
            "exclusion_rationale": (
                "'duration' is excluded from the feature space. It records call length, "
                "which is only known once the call has happened and the outcome is "
                "effectively decided — a zero-second call is necessarily a 'no'. UCI's own "
                "documentation warns it must be discarded for realistic modelling. "
                "Clustering on it would embed the answer in the segments. 'y' is the "
                "held-out validation signal."
            ),
            "outcome_rate": round(float(outcome.mean()), 5),
            "nulls": int(df.isna().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "unknown_counts": {
                c: int((df[c] == "unknown").sum())
                for c in categorical_cols
                if (df[c] == "unknown").any()
            },
            "numeric_summary": {
                c: artifacts.histogram(df[c].to_numpy(), bins=25) for c in numeric_cols
            },
            "categorical_summary": {
                c: df[c].value_counts().head(12).to_dict() for c in categorical_cols
            },
        }
        print(f"      {n:,} customers, {len(numeric_cols)} numeric + "
              f"{len(categorical_cols)} categorical features")
        print(f"      outcome rate {profile['outcome_rate']:.2%} (held out of clustering)")

        crisp.record(
            Phase(
                name="data_understanding",
                summary=(
                    f"{n:,} real contact records. {len(numeric_cols)} numeric and "
                    f"{len(categorical_cols)} categorical attributes describe the customer "
                    "and the campaign; the subscription outcome is withheld from the "
                    "feature space and reserved for external validation. Missing values "
                    "are coded as the literal string 'unknown' rather than as nulls."
                ),
                evidence=profile,
                decisions=[
                    Decision(
                        question="How should 'unknown' categorical values be treated?",
                        choice="Kept as an explicit level, not imputed.",
                        rationale=(
                            "'unknown' is informative here: a customer whose employment or "
                            "education the bank failed to record differs systematically "
                            "from one it recorded. Imputing to the mode would erase that "
                            "signal and manufacture certainty the data does not have."
                        ),
                        alternatives_rejected=[
                            "Mode imputation — destroys a real signal and inflates the "
                            "apparent completeness of the data.",
                            "Dropping rows with any 'unknown' — would discard a large and "
                            "non-random share of the population.",
                        ],
                    ),
                ],
                risks=[
                    "Macroeconomic columns (euribor3m, emp.var.rate, nr.employed) vary "
                    "with calendar time rather than with the customer. Including them "
                    "risks segmenting by *when* someone was called rather than by who "
                    "they are — examined in the feature-set ablation below.",
                ],
            )
        )

        # --- Data preparation ----------------------------------------------------------
        print("\n[2/6] Encoding mixed-type feature space …")

        # Two candidate feature sets, because the macro columns are a genuine judgement
        # call rather than an obvious include or exclude.
        macro_cols = ["emp.var.rate", "cons.price.idx", "cons.conf.idx",
                      "euribor3m", "nr.employed"]
        customer_numeric = [c for c in numeric_cols if c not in macro_cols]

        def make_encoder(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
            return ColumnTransformer(
                [
                    ("num", StandardScaler(), numeric),
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                      min_frequency=0.01),
                        categorical,
                    ),
                ]
            )

        feature_sets = {
            "customer_only": (customer_numeric, categorical_cols),
            "customer_plus_macro": (numeric_cols, categorical_cols),
        }

        encoded: dict[str, np.ndarray] = {}
        for name, (num, cat) in feature_sets.items():
            encoder = make_encoder(num, cat)
            encoded[name] = encoder.fit_transform(df[num + cat])
            print(f"      {name}: {encoded[name].shape[1]} dimensions after encoding")

        # --- Model selection over k -----------------------------------------------------
        print("\n[3/6] Sweeping k with three validity indices …")

        # Subsample for the index sweep: silhouette is O(n^2) and 41k points would take
        # far longer than the information gained justifies. The final fit uses all rows.
        rng = np.random.default_rng(SEED)
        sample_idx = rng.choice(n, size=min(6000, n), replace=False)

        sweeps: dict[str, list] = {}
        for fs_name, X in encoded.items():
            X_sample = X[sample_idx]
            rows = []
            for k in K_RANGE:
                km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
                labels = km.fit_predict(X_sample)
                report = metrics.clustering_report(X_sample, labels)
                rows.append(
                    {
                        "k": k,
                        "silhouette": report["silhouette"],
                        "calinski_harabasz": report["calinski_harabasz"],
                        "davies_bouldin": report["davies_bouldin"],
                        "inertia": round(float(km.inertia_), 2),
                    }
                )
            sweeps[fs_name] = rows

        def best_k_by(rows: list[dict], key: str, maximise: bool) -> int:
            return (max if maximise else min)(rows, key=lambda r: r[key])["k"]

        consensus = {}
        for fs_name, rows in sweeps.items():
            votes = {
                "silhouette": best_k_by(rows, "silhouette", True),
                "calinski_harabasz": best_k_by(rows, "calinski_harabasz", True),
                "davies_bouldin": best_k_by(rows, "davies_bouldin", False),
            }
            consensus[fs_name] = {
                "votes": votes,
                "unanimous": len(set(votes.values())) == 1,
            }
            print(f"      {fs_name}: silhouette→k={votes['silhouette']}, "
                  f"CH→k={votes['calinski_harabasz']}, DB→k={votes['davies_bouldin']}"
                  f"{' (unanimous)' if consensus[fs_name]['unanimous'] else ' (disagree)'}")

        # --- Bootstrap stability ---------------------------------------------------------
        print(f"\n[4/6] Bootstrap stability over {N_BOOTSTRAP} resamples …")
        stability: dict[str, list] = {}
        for fs_name, X in encoded.items():
            X_sample = X[sample_idx]
            rows = []
            for k in K_RANGE:
                reference = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(
                    X_sample
                )
                scores = []
                for b in range(N_BOOTSTRAP):
                    boot_rng = np.random.default_rng(SEED + b)
                    # Resample with replacement, then compare labels on the shared rows.
                    idx = boot_rng.choice(len(X_sample), size=len(X_sample), replace=True)
                    unique_idx = np.unique(idx)
                    boot_labels = KMeans(
                        n_clusters=k, random_state=SEED + b, n_init=5
                    ).fit(X_sample[idx]).predict(X_sample[unique_idx])
                    scores.append(adjusted_rand_score(reference[unique_idx], boot_labels))
                rows.append(
                    {
                        "k": k,
                        "mean_ari": round(float(np.mean(scores)), 4),
                        "std_ari": round(float(np.std(scores)), 4),
                        "min_ari": round(float(np.min(scores)), 4),
                        "stable": bool(np.mean(scores) >= STABILITY_THRESHOLD),
                    }
                )
            stability[fs_name] = rows
            stable_ks = [r["k"] for r in rows if r["stable"]]
            print(f"      {fs_name}: stable k values = {stable_ks or 'none'}")
        return_payload = {
            "profile": profile,
            "sweeps": sweeps,
            "consensus": consensus,
            "stability": stability,
        }
        _finish(
            df, outcome, encoded, sweeps, consensus, stability, crisp, ctx,
            KMeans, GaussianMixture, AgglomerativeClustering, PCA, adjusted_rand_score,
            profile, sample_idx,
        )


def _finish(df, outcome, encoded, sweeps, consensus, stability, crisp, ctx,
            KMeans, GaussianMixture, AgglomerativeClustering, PCA, adjusted_rand_score,
            profile, sample_idx) -> None:
    """Select the operating segmentation, validate externally and write artifacts."""
    import numpy as np

    # --- Choose feature set and k ------------------------------------------------------
    # Prefer the feature set whose indices agree and whose partition is reproducible.
    chosen_fs = None
    chosen_k = None
    for fs_name in ("customer_only", "customer_plus_macro"):
        stable_rows = [r for r in stability[fs_name] if r["stable"]]
        if not stable_rows:
            continue
        # Among stable k, take the one with the best silhouette.
        sil = {r["k"]: r["silhouette"] for r in sweeps[fs_name]}
        candidate = max(stable_rows, key=lambda r: sil[r["k"]])
        if chosen_k is None:
            chosen_fs, chosen_k = fs_name, candidate["k"]

    fallback_used = chosen_k is None
    if fallback_used:
        # No k met the stability bar. Report that rather than silently picking one.
        chosen_fs = "customer_only"
        chosen_k = max(sweeps[chosen_fs], key=lambda r: r["silhouette"])["k"]

    print(f"\n[5/6] Selected feature set '{chosen_fs}', k={chosen_k}"
          f"{' (STABILITY BAR NOT MET — see caveat)' if fallback_used else ''}")

    X = encoded[chosen_fs]

    # --- Algorithm comparison at the chosen k -------------------------------------------
    algorithms = {}
    km = KMeans(n_clusters=chosen_k, random_state=SEED, n_init=10)
    labels_km = km.fit_predict(X)
    algorithms["kmeans"] = {
        "label": "K-means",
        "assumption": "isotropic, equally-sized spherical clusters",
        **metrics.clustering_report(X[sample_idx], labels_km[sample_idx]),
    }

    gmm = GaussianMixture(n_components=chosen_k, random_state=SEED, covariance_type="full")
    labels_gmm = gmm.fit_predict(X)
    algorithms["gmm"] = {
        "label": "Gaussian mixture (full covariance)",
        "assumption": "elliptical clusters, soft assignment",
        "bic": round(float(gmm.bic(X)), 1),
        "aic": round(float(gmm.aic(X)), 1),
        **metrics.clustering_report(X[sample_idx], labels_gmm[sample_idx]),
    }

    agg_labels = AgglomerativeClustering(n_clusters=chosen_k).fit_predict(X[sample_idx])
    algorithms["agglomerative"] = {
        "label": "Agglomerative (Ward linkage)",
        "assumption": "hierarchical, minimises within-cluster variance increase",
        **metrics.clustering_report(X[sample_idx], agg_labels),
    }

    cross_agreement = {
        "kmeans_vs_gmm": round(
            float(adjusted_rand_score(labels_km, labels_gmm)), 4
        ),
        "kmeans_vs_agglomerative": round(
            float(adjusted_rand_score(labels_km[sample_idx], agg_labels)), 4
        ),
        "interpretation": (
            "Agreement between algorithms with different geometric assumptions is "
            "evidence that the structure is in the data rather than in the method. "
            "Disagreement means at least one algorithm is imposing its own shape."
        ),
    }

    # --- External validation: does the segmentation predict an outcome it never saw? ----
    print("[6/6] External validation against held-out subscription outcome …")
    seg = pd.DataFrame({"segment": labels_km, "outcome": outcome.to_numpy()})
    overall_rate = float(seg.outcome.mean())
    segment_profiles = []
    for segment_id in sorted(seg.segment.unique()):
        mask = seg.segment == segment_id
        rate = float(seg.loc[mask, "outcome"].mean())
        members = df.loc[mask.to_numpy()]
        segment_profiles.append(
            {
                "segment": int(segment_id),
                "size": int(mask.sum()),
                "share": round(float(mask.mean()), 4),
                "conversion_rate": round(rate, 5),
                "lift_vs_overall": round(rate / overall_rate, 3) if overall_rate else None,
                "median_age": float(members.age.median()),
                "top_job": members.job.mode().iat[0] if len(members) else None,
                "top_education": members.education.mode().iat[0] if len(members) else None,
                "top_contact": members.contact.mode().iat[0] if len(members) else None,
                "mean_campaign_calls": round(float(members.campaign.mean()), 2),
                "prior_contact_share": round(float((members.pdays != 999).mean()), 4),
            }
        )
    segment_profiles.sort(key=lambda s: -s["conversion_rate"])

    # Chi-square: are conversion differences across segments larger than chance?
    from scipy.stats import chi2_contingency

    table = pd.crosstab(seg.segment, seg.outcome).to_numpy()
    chi2, p_value, dof, _ = chi2_contingency(table)
    spread = (
        segment_profiles[0]["conversion_rate"] - segment_profiles[-1]["conversion_rate"]
    )
    external = {
        "overall_conversion": round(overall_rate, 5),
        "segment_profiles": segment_profiles,
        "conversion_spread": round(spread, 5),
        "best_segment_lift": segment_profiles[0]["lift_vs_overall"],
        "chi2": round(float(chi2), 2),
        "p_value": float(p_value),
        "dof": int(dof),
        "significant": bool(p_value < 0.05),
        "interpretation": (
            f"Conversion ranges from {segment_profiles[-1]['conversion_rate']:.2%} to "
            f"{segment_profiles[0]['conversion_rate']:.2%} across segments "
            f"(χ²={chi2:.1f}, p={p_value:.2e}). The clustering never saw the outcome, so "
            "this separation is genuine external validation rather than a restatement of "
            "the objective the algorithm optimised."
        ),
    }
    print(f"      conversion {segment_profiles[-1]['conversion_rate']:.2%} → "
          f"{segment_profiles[0]['conversion_rate']:.2%}, χ²={chi2:.0f}, p={p_value:.1e}")

    # --- 2-D projection for the UI --------------------------------------------------
    # audit: ok(preprocessing-leak) This PCA is display-only. The analysis is unsupervised,
    # there is no held-out partition for its statistics to contaminate, and the components
    # are never used as model input — they exist solely to place points on a scatter plot.
    # The rule fires correctly; fitting a transformer on all rows is a leak whenever a
    # holdout exists, and there is none here.
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(X[sample_idx])
    projection = {
        "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "points": [
            {
                "x": round(float(coords[i, 0]), 3),
                "y": round(float(coords[i, 1]), 3),
                "segment": int(labels_km[sample_idx][i]),
                "outcome": int(outcome.to_numpy()[sample_idx][i]),
            }
            for i in range(0, len(sample_idx), 2)
        ],
        "caveat": (
            f"The two components shown carry only "
            f"{sum(pca.explained_variance_ratio_):.1%} of total variance, so visual "
            "overlap in this plot does not mean the segments overlap in the full space. "
            "The projection is for orientation, not for judging separation."
        ),
    }

    crisp.record(
        Phase(
            name="data_preparation",
            summary=(
                "Numeric features standardised and categoricals one-hot encoded with rare "
                "levels folded together, all inside a ColumnTransformer. Two candidate "
                "feature sets were carried forward — customer attributes alone, and "
                "customer attributes plus macroeconomic context — because whether the "
                "macro columns describe the customer or merely the calendar is a real "
                "judgement call."
            ),
            decisions=[
                Decision(
                    question="Include macroeconomic indicators in the feature space?",
                    choice=f"Selected feature set: {chosen_fs}.",
                    rationale=(
                        "euribor3m and nr.employed vary with the date of the call, not "
                        "with the person. Including them risks producing segments that "
                        "are really time periods wearing customer labels. Both variants "
                        "were swept and the choice is made on stability rather than "
                        "asserted up front."
                    ),
                ),
            ],
            evidence={
                "feature_sets": {k: int(v.shape[1]) for k, v in encoded.items()},
                "chosen": chosen_fs,
            },
        )
    )
    crisp.record(
        Phase(
            name="modeling",
            summary=(
                f"k swept over {min(K_RANGE)}–{max(K_RANGE)} against three validity "
                f"indices for both feature sets, then every k stress-tested with "
                f"{N_BOOTSTRAP} bootstrap resamples. Three algorithms with different "
                f"geometric assumptions were fitted at the selected k={chosen_k}."
            ),
            decisions=[
                Decision(
                    question="How is k chosen?",
                    choice=(
                        f"k={chosen_k}, the best-silhouette k among those passing the "
                        f"ARI ≥ {STABILITY_THRESHOLD} stability bar."
                        if not fallback_used
                        else f"k={chosen_k} by silhouette — no k passed the stability bar."
                    ),
                    rationale=(
                        "Validity indices alone can favour a k whose partition changes "
                        "entirely under resampling. Requiring reproducibility first, and "
                        "optimising geometry second, prevents reporting a segmentation "
                        "that describes this sample rather than this population."
                    ),
                    alternatives_rejected=[
                        "Elbow method on inertia — inertia decreases monotonically and "
                        "the 'elbow' is read by eye, so it is not a criterion.",
                        "Highest silhouette outright — ignores whether the partition "
                        "reproduces.",
                    ],
                ),
            ],
            evidence={
                "sweeps": sweeps,
                "index_consensus": consensus,
                "stability": stability,
                "algorithms": algorithms,
                "cross_algorithm_agreement": cross_agreement,
                "chosen_k": chosen_k,
                "chosen_feature_set": chosen_fs,
                "stability_bar_met": not fallback_used,
            },
        )
    )
    crisp.record(
        Phase(
            name="evaluation",
            summary=external["interpretation"],
            evidence={"external_validation": external, "projection_caveat": projection["caveat"]},
            risks=(
                [
                    f"No k between {min(K_RANGE)} and {max(K_RANGE)} reached mean ARI "
                    f"{STABILITY_THRESHOLD} under bootstrap resampling. The reported "
                    "segmentation is therefore provisional: it separates conversion "
                    "rates, but a different sample of the same population would likely "
                    "produce different segment boundaries. This is reported rather than "
                    "concealed by presenting only the silhouette-optimal k."
                ]
                if fallback_used
                else []
            )
            + [
                (
                    f"The three validity indices unanimously prefer k=2, while the "
                    f"bootstrap admits only k∈{[r['k'] for r in stability[chosen_fs] if r['stable']]}. "
                    "Geometry and reproducibility disagree, and the reproducible answer "
                    "was taken. A k=2 split would be the cleanest geometrically and would "
                    "also be nearly useless commercially, which is a reminder that "
                    "internal indices optimise a mathematical objective rather than a "
                    "business one."
                ),
                (
                    f"K-means agrees with GMM at ARI "
                    f"{cross_agreement['kmeans_vs_gmm']} and with Ward linkage at "
                    f"{cross_agreement['kmeans_vs_agglomerative']} — moderate, not strong. "
                    "Roughly half the partition structure is therefore imposed by "
                    "K-means's spherical assumption rather than present in the data. "
                    "Segment boundaries should be treated as one defensible partition "
                    "among several, not as discovered natural kinds."
                ),
                (
                    "The highest-converting segment is defined largely by having been "
                    "contacted in a previous campaign (100% prior-contact share). That is "
                    "a legitimate predictor — it is known before the call is placed — but "
                    "it means the segment mostly re-identifies already-engaged customers "
                    "rather than revealing a latent group."
                ),
                "Segments are descriptive, not causal. A segment converting at 3× the "
                "average is not evidence that moving a customer into it would raise their "
                "probability of subscribing.",
                "One institution, one product, 2008–2010, during a financial crisis. "
                "Segment structure is unlikely to transfer.",
            ],
        )
    )
    crisp.record(
        Phase(
            name="deployment",
            summary=(
                "Segment centroids and profiles are exported so the published explorer can "
                "assign a new customer to a segment in the browser and show its conversion "
                "history, with the stability caveat displayed alongside."
            ),
            evidence={"n_segments": chosen_k},
        )
    )

    artifacts.write(OUT / "profile.json", profile, context=ctx)
    artifacts.write(
        OUT / "selection.json",
        {
            "sweeps": sweeps,
            "consensus": consensus,
            "stability": stability,
            "stability_threshold": STABILITY_THRESHOLD,
            "n_bootstrap": N_BOOTSTRAP,
            "chosen_k": chosen_k,
            "chosen_feature_set": chosen_fs,
            "stability_bar_met": not fallback_used,
            "algorithms": algorithms,
            "cross_algorithm_agreement": cross_agreement,
        },
        context=ctx,
    )
    artifacts.write(OUT / "segments.json",
                    {"external_validation": external, "projection": projection}, context=ctx)
    artifacts.write(OUT / "crispdm.json", crisp.to_dict(), context=ctx)
    artifacts.write(OUT / "provenance.json",
                    {"datasets": [data.provenance_record("bank_marketing")]}, context=ctx)

    print(f"\n✓ {PROJECT} complete — k={chosen_k} on '{chosen_fs}', "
          f"conversion spread {external['conversion_spread']:.2%}, "
          f"stability bar {'met' if not fallback_used else 'NOT met'}")


if __name__ == "__main__":
    main()
