"""Sub-linear similarity search: MinHash and LSH, with the approximation error measured.

Run:
    PYTHONPATH=lib python3 projects/09_similarity_search/pipeline/build.py

The task is entity resolution on real data: 1,534 company names, entered free-form by US
consumers filing complaints, in which the same institution appears under many spellings.
Finding those variants is a nearest-neighbour problem, and doing it exactly costs O(n²)
pair comparisons.

MinHash and LSH make it sub-linear. The entire point of both is that they are
**approximations** — so the only honest way to present them is to compute the exact answer
too and measure precisely what the approximation loses.

That is what this project is: not "LSH is fast", but "LSH is this much faster and misses
this many true pairs, and here is the recall/speed curve you are choosing a point on".

Both algorithms are implemented from first principles. MinHash is four lines of insight
wrapped in one loop, and a library call would hide the part worth understanding: that the
probability two sets produce the same minimum hash is exactly their Jaccard similarity.
"""

from __future__ import annotations

import hashlib
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lib"))

from dsx import artifacts, data  # noqa: E402
from dsx.crispdm import CrispDm, Decision, Phase  # noqa: E402

PROJECT = "09_similarity_search"
OUT = ROOT / "projects" / PROJECT / "artifacts"
SEED = 42

SHINGLE_SIZE = 3          # character 3-grams
N_PERMUTATIONS = 128      # MinHash signature length
SIMILARITY_THRESHOLD = 0.5
MERSENNE_PRIME = (1 << 61) - 1


# ======================================================================================
# Shingling
# ======================================================================================
def normalise(name: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace.

    Deliberately conservative: it does NOT strip corporate suffixes like 'LLC' or 'Inc'.
    Removing them would make the matching task artificially easy and would hide the
    question the project is actually about — whether the approximation finds pairs that
    exact search finds.
    """
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def shingles(text: str, k: int = SHINGLE_SIZE) -> set[int]:
    """Hash the character k-grams of a string into a set of integers.

    Character shingles rather than word tokens, because the variation being detected is
    largely orthographic — 'Bank of America' vs 'Bank of Amerca' vs 'BankofAmerica'. Word
    tokens treat a single typo as a completely different token; character 3-grams degrade
    gracefully, losing only the grams that span the error.
    """
    padded = normalise(text)
    if len(padded) < k:
        padded = padded.ljust(k)
    return {
        int.from_bytes(hashlib.blake2b(padded[i : i + k].encode(), digest_size=8).digest(), "big")
        for i in range(len(padded) - k + 1)
    }


def jaccard(a: set[int], b: set[int]) -> float:
    """Exact Jaccard similarity |A ∩ B| / |A ∪ B|."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


# ======================================================================================
# MinHash
# ======================================================================================
def build_minhash_signatures(sets: list[set[int]], n_perm: int, seed: int) -> np.ndarray:
    """Compute MinHash signatures for every set.

    The idea in one sentence: under a random permutation of the universe of shingles, the
    probability that two sets share the same minimum element is *exactly* their Jaccard
    similarity. Repeat with ``n_perm`` independent permutations and the fraction of
    positions where two signatures agree is an unbiased estimator of Jaccard.

    Its standard deviation is ``sqrt(s(1-s)/k)``, not a constant: the agreement count is
    Binomial(k, s). That means precision is best exactly where it matters least — near
    s=0, where the sd is ~0 — and worst at s=0.5, where it peaks at ``0.5/sqrt(k)``. The
    commonly quoted ``1/sqrt(k)`` is twice that peak, and treating it as the typical error
    misreads the estimator in both directions. The evaluation below reports error
    stratified by true similarity for this reason.

    True random permutations of a 2^64 universe are not storable, so we use the standard
    substitute: a family of universal hash functions h(x) = (a·x + b) mod p, with p a
    Mersenne prime larger than any shingle hash. Each (a, b) pair behaves like an
    independent permutation for this purpose.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(1, MERSENNE_PRIME, size=n_perm, dtype=np.uint64)
    b = rng.integers(0, MERSENNE_PRIME, size=n_perm, dtype=np.uint64)

    signatures = np.full((len(sets), n_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, shingle_set in enumerate(sets):
        if not shingle_set:
            continue
        values = np.fromiter(shingle_set, dtype=np.uint64, count=len(shingle_set))
        # Outer product of hashes against permutations, then min down the set axis.
        hashed = (np.outer(values, a) + b) % MERSENNE_PRIME
        signatures[i] = hashed.min(axis=0)
    return signatures


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Fraction of signature positions that agree — the MinHash estimate of Jaccard."""
    return float(np.mean(sig_a == sig_b))


# ======================================================================================
# LSH banding
# ======================================================================================
def lsh_candidates(signatures: np.ndarray, bands: int, rows: int) -> set[tuple[int, int]]:
    """Find candidate pairs by banded locality-sensitive hashing.

    The signature is split into ``bands`` bands of ``rows`` rows each. Two items become
    candidates if they agree *exactly* on at least one entire band.

    The probability of that happening for a pair with Jaccard s is::

        P(candidate) = 1 - (1 - s^rows)^bands

    which is an S-curve with its steep region near s ≈ (1/bands)^(1/rows). Choosing bands
    and rows is choosing where that threshold sits — and therefore choosing the
    recall/precision trade-off explicitly rather than by accident.
    """
    candidates: set[tuple[int, int]] = set()
    for band in range(bands):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        start = band * rows
        chunk = signatures[:, start : start + rows]
        for index in range(len(signatures)):
            buckets[chunk[index].tobytes()].append(index)
        for members in buckets.values():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    candidates.add((members[i], members[j]))
    return candidates


def s_curve(bands: int, rows: int, points: int = 50) -> list[dict]:
    """Theoretical probability that a pair becomes a candidate, as a function of Jaccard."""
    return [
        {
            "similarity": round(float(s), 3),
            "p_candidate": round(float(1 - (1 - s**rows) ** bands), 5),
        }
        for s in np.linspace(0, 1, points)
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    crisp = CrispDm(
        project=PROJECT,
        business_question=(
            "The same company appears under many spellings in free-text complaint records. "
            "Can near-duplicate names be found without comparing every pair — and what "
            "exactly does the faster method miss?"
        ),
    )
    crisp.record(
        Phase(
            name="business_understanding",
            summary=(
                "Counting complaints per institution requires knowing which name strings "
                "refer to the same institution. Exact pairwise comparison is O(n²), which "
                "is fine at 1,534 names and impossible at a million. The question is what "
                "the sub-linear alternative costs in accuracy, which can only be answered "
                "by computing the exact answer as well and comparing."
            ),
            decisions=[
                Decision(
                    question="What makes this project's result trustworthy?",
                    choice=(
                        "Computing the exact O(n²) answer and measuring the approximation "
                        "against it."
                    ),
                    rationale=(
                        "LSH is an approximation. A write-up reporting only its speed is "
                        "reporting half the trade. At this scale the exact answer is "
                        "computable — 1.18M pairs — so the recall and precision of the "
                        "approximation are measurable rather than assumed."
                    ),
                    alternatives_rejected=[
                        "Report the speedup alone — the usual presentation, and it omits "
                        "the cost being paid for it.",
                        "Use a library implementation — hides the banding mechanism, which "
                        "is the part that determines the trade-off.",
                    ],
                ),
            ],
            evidence={"n_permutations": N_PERMUTATIONS, "shingle_size": SHINGLE_SIZE},
        )
    )

    with artifacts.run(PROJECT, seed=SEED) as ctx:
        # --- Data understanding -------------------------------------------------------
        print("\n[1/6] Loading real consumer complaint records …")
        df = data.load_csv("consumer_complaints")
        companies = sorted(df["Company"].dropna().unique().tolist())
        n = len(companies)

        complaint_counts = df["Company"].value_counts()
        profile = {
            "n_complaints": int(len(df)),
            "n_distinct_company_strings": n,
            "n_products": int(df["Product"].nunique()),
            "n_issues": int(df["Issue"].nunique()),
            "exact_pairs": n * (n - 1) // 2,
            "top_companies": [
                {"company": c, "complaints": int(v)}
                for c, v in complaint_counts.head(10).items()
            ],
            "name_length": artifacts.histogram(
                np.array([len(c) for c in companies], dtype=float), bins=25
            ),
        }
        print(f"      {len(df):,} complaints, {n:,} distinct company strings")
        print(f"      exact comparison would need {profile['exact_pairs']:,} pairs")

        # --- Data preparation ----------------------------------------------------------
        print(f"\n[2/6] Shingling into character {SHINGLE_SIZE}-grams …")
        shingle_sets = [shingles(c) for c in companies]
        sizes = np.array([len(s) for s in shingle_sets], dtype=float)
        prep = {
            "shingle_size": SHINGLE_SIZE,
            "mean_shingles_per_name": round(float(sizes.mean()), 2),
            "min_shingles": int(sizes.min()),
            "max_shingles": int(sizes.max()),
            "normalisation": (
                "Lowercased, punctuation replaced by spaces, whitespace collapsed. "
                "Corporate suffixes (LLC, Inc, Corporation) are deliberately NOT stripped: "
                "removing them would make matching artificially easy and obscure the "
                "question of whether the approximation finds what exact search finds."
            ),
        }
        print(f"      {prep['mean_shingles_per_name']:.1f} shingles per name on average")

        # --- Exact ground truth ---------------------------------------------------------
        print(f"\n[3/6] Exact O(n²) search over {profile['exact_pairs']:,} pairs …")
        started = time.perf_counter()
        exact_pairs: dict[tuple[int, int], float] = {}
        for i in range(n):
            si = shingle_sets[i]
            for j in range(i + 1, n):
                similarity = jaccard(si, shingle_sets[j])
                if similarity >= SIMILARITY_THRESHOLD:
                    exact_pairs[(i, j)] = similarity
        exact_seconds = time.perf_counter() - started
        print(f"      {len(exact_pairs):,} pairs above Jaccard {SIMILARITY_THRESHOLD} "
              f"in {exact_seconds:.1f}s")

        # --- MinHash --------------------------------------------------------------------
        print(f"\n[4/6] Building {N_PERMUTATIONS}-permutation MinHash signatures …")
        started = time.perf_counter()
        signatures = build_minhash_signatures(shingle_sets, N_PERMUTATIONS, SEED)
        minhash_seconds = time.perf_counter() - started

        # How good is the MinHash estimate itself, before LSH enters the picture?
        rng = np.random.default_rng(SEED)
        sample_pairs = [
            (int(rng.integers(0, n)), int(rng.integers(0, n))) for _ in range(4000)
        ]
        sample_pairs = [(i, j) for i, j in sample_pairs if i != j]
        estimate_errors = []
        estimate_scatter = []
        for i, j in sample_pairs:
            true_similarity = jaccard(shingle_sets[i], shingle_sets[j])
            estimated = estimate_jaccard(signatures[i], signatures[j])
            estimate_errors.append(estimated - true_similarity)
            if len(estimate_scatter) < 600:
                estimate_scatter.append(
                    {"true": round(true_similarity, 4), "estimated": round(estimated, 4)}
                )

        errors = np.array(estimate_errors)
        true_similarities = np.array(
            [jaccard(shingle_sets[i], shingle_sets[j]) for i, j in sample_pairs]
        )

        # The MinHash signature agreement count is Binomial(k, s), so the estimator's
        # standard deviation is sqrt(s(1-s)/k) — a function of the similarity being
        # estimated, NOT a constant. It is zero at s=0 and maximal at s=0.5, where it
        # equals 0.5/sqrt(k).
        #
        # This matters here because random pairs of company names are overwhelmingly
        # dissimilar, so an aggregate standard deviation over all sampled pairs is small
        # for a trivial reason and would look — wrongly — like the estimator beating its
        # own theoretical variance. Reporting it stratified by true similarity is the only
        # way the comparison against theory means anything.
        bands_of_s = [(0.0, 0.01), (0.01, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.01)]
        stratified = []
        for lo, hi in bands_of_s:
            mask = (true_similarities >= lo) & (true_similarities < hi)
            if mask.sum() < 5:
                continue
            midpoint = float(true_similarities[mask].mean())
            stratified.append(
                {
                    "similarity_band": f"[{lo:.2f}, {hi:.2f})",
                    "n_pairs": int(mask.sum()),
                    "mean_true_similarity": round(midpoint, 4),
                    "observed_std": round(float(errors[mask].std()), 5),
                    "theoretical_std": round(
                        float(np.sqrt(midpoint * (1 - midpoint) / N_PERMUTATIONS)), 5
                    ),
                    "mean_error": round(float(errors[mask].mean()), 5),
                }
            )

        max_theoretical_std = 0.5 / np.sqrt(N_PERMUTATIONS)
        minhash_accuracy = {
            "n_pairs_sampled": len(sample_pairs),
            "mean_error": round(float(errors.mean()), 5),
            "mean_absolute_error": round(float(np.abs(errors).mean()), 5),
            "std_error_aggregate": round(float(errors.std()), 5),
            "max_theoretical_std": round(float(max_theoretical_std), 5),
            "stratified_by_similarity": stratified,
            "share_of_pairs_below_0_1": round(
                float((true_similarities < 0.1).mean()), 4
            ),
            "scatter": estimate_scatter,
            "interpretation": (
                f"MinHash estimates Jaccard with mean error {errors.mean():+.5f} — "
                "essentially unbiased, as the theory requires. The aggregate standard "
                f"deviation ({errors.std():.4f}) looks impressively small, but that figure "
                "is misleading on its own and is reported here only so it can be "
                f"corrected: {(true_similarities < 0.1).mean():.0%} of randomly drawn pairs "
                "have similarity below 0.1, and the estimator's standard deviation is "
                "sqrt(s(1-s)/k) — near zero for dissimilar pairs. It is maximal at s=0.5, "
                f"where it equals 0.5/sqrt(k) = {max_theoretical_std:.4f}. The stratified "
                "table is the meaningful comparison: within each similarity band the "
                "observed spread tracks the theoretical value closely. Note also that the "
                "commonly quoted rule of thumb 1/sqrt(k) = "
                f"{1 / np.sqrt(N_PERMUTATIONS):.4f} is twice the true maximum."
            ),
        }
        print(f"      estimate bias {errors.mean():+.5f} (unbiased as required)")
        for row in stratified:
            print(f"        s in {row['similarity_band']:<14} n={row['n_pairs']:>4}  "
                  f"observed sd {row['observed_std']:.4f}  "
                  f"theory {row['theoretical_std']:.4f}")

        # --- LSH sweep ------------------------------------------------------------------
        print("\n[5/6] Sweeping LSH band configurations …")
        exact_set = set(exact_pairs)
        configurations = []
        for bands, rows in [(8, 16), (16, 8), (26, 4), (32, 4), (43, 3), (64, 2)]:
            if bands * rows > N_PERMUTATIONS:
                continue
            # Time the complete two-stage pipeline, not just candidate generation.
            # Verification is not optional — LSH output contains false positives by
            # construction — so timing only the banding step would overstate the speedup
            # by excluding work the method requires.
            started = time.perf_counter()
            candidates = lsh_candidates(signatures, bands, rows)
            banding_seconds = time.perf_counter() - started

            started = time.perf_counter()
            verified = {
                pair for pair in candidates
                if jaccard(shingle_sets[pair[0]], shingle_sets[pair[1]]) >= SIMILARITY_THRESHOLD
            }
            verify_seconds = time.perf_counter() - started
            lsh_seconds = banding_seconds + verify_seconds
            found = verified & exact_set
            recall = len(found) / max(1, len(exact_set))
            precision = len(found) / max(1, len(verified))
            threshold_estimate = (1 / bands) ** (1 / rows)

            configurations.append(
                {
                    "bands": bands,
                    "rows": rows,
                    "threshold_estimate": round(float(threshold_estimate), 4),
                    "candidate_pairs": len(candidates),
                    "candidate_fraction_of_exact": round(
                        len(candidates) / max(1, profile["exact_pairs"]), 6
                    ),
                    "verified_pairs": len(verified),
                    "true_pairs_found": len(found),
                    "true_pairs_missed": len(exact_set) - len(found),
                    "recall": round(recall, 4),
                    "precision": round(precision, 4),
                    "f1": round(
                        2 * recall * precision / max(1e-9, recall + precision), 4
                    ),
                    "banding_seconds": round(banding_seconds, 4),
                    "verify_seconds": round(verify_seconds, 4),
                    "total_seconds": round(lsh_seconds, 4),
                    "speedup_vs_exact": round(exact_seconds / max(1e-9, lsh_seconds), 1),
                    "s_curve": s_curve(bands, rows, points=40),
                }
            )
            print(f"      b={bands:>3} r={rows:<3} recall {recall:.3f}  "
                  f"precision {precision:.3f}  {len(candidates):>7,} candidates  "
                  f"{exact_seconds / max(1e-9, lsh_seconds):>6.1f}× faster")

        best = max(configurations, key=lambda c: c["f1"])
        highest_recall = max(configurations, key=lambda c: c["recall"])

        # --- Example matches -------------------------------------------------------------
        top_matches = sorted(exact_pairs.items(), key=lambda kv: -kv[1])[:25]
        examples = [
            {
                "a": companies[i],
                "b": companies[j],
                "jaccard": round(similarity, 4),
                "minhash_estimate": round(estimate_jaccard(signatures[i], signatures[j]), 4),
                "complaints_a": int(complaint_counts.get(companies[i], 0)),
                "complaints_b": int(complaint_counts.get(companies[j], 0)),
            }
            for (i, j), similarity in top_matches
        ]

        crisp.record(
            Phase(
                name="data_understanding",
                summary=(
                    f"{len(df):,} real complaints naming {n:,} distinct company strings. "
                    f"Exhaustive comparison requires {profile['exact_pairs']:,} pairs, which "
                    "is tractable here and is used as ground truth for everything below."
                ),
                evidence=profile,
                risks=[
                    "Ground truth is 'Jaccard above a threshold', not 'genuinely the same "
                    "company'. String similarity is a proxy: two distinct subsidiaries of "
                    "one group can score highly, and a company that rebranded entirely "
                    "will score low. The recall figures measure agreement with exact "
                    "string search, not with reality.",
                ],
            )
        )
        crisp.record(
            Phase(
                name="data_preparation",
                summary=(
                    f"Names normalised and hashed into character {SHINGLE_SIZE}-gram sets, "
                    f"averaging {prep['mean_shingles_per_name']:.1f} shingles each."
                ),
                decisions=[
                    Decision(
                        question="Character shingles or word tokens?",
                        choice=f"Character {SHINGLE_SIZE}-grams.",
                        rationale=(
                            "The variation here is largely orthographic — spacing, "
                            "abbreviation, typos. Word tokenisation treats a single "
                            "misspelling as an entirely different token and the similarity "
                            "collapses. Character n-grams degrade gracefully, losing only "
                            "the few grams spanning the error."
                        ),
                        alternatives_rejected=[
                            "Word tokens — brittle to typos, which are the dominant source "
                            "of variation in free-text entry.",
                            "Stripping corporate suffixes — would inflate every similarity "
                            "and make the comparison against exact search less informative.",
                        ],
                    ),
                ],
                evidence=prep,
            )
        )
        crisp.record(
            Phase(
                name="modeling",
                summary=(
                    f"{N_PERMUTATIONS}-permutation MinHash signatures built in "
                    f"{minhash_seconds:.2f}s, then {len(configurations)} LSH band "
                    "configurations swept, each verified exactly and scored against the "
                    "brute-force ground truth."
                ),
                decisions=[
                    Decision(
                        question="How are bands and rows chosen?",
                        choice="Swept, and the whole trade-off curve published.",
                        rationale=(
                            "bands × rows fixes the S-curve threshold at approximately "
                            "(1/b)^(1/r). There is no universally correct point on it — "
                            "more bands means higher recall and more candidates to verify. "
                            "Publishing the sweep lets a reader pick the point their own "
                            "recall requirement implies."
                        ),
                        alternatives_rejected=[
                            "Pick one configuration and report its numbers — presents a "
                            "chosen point on a trade-off as though it were the method's "
                            "performance.",
                        ],
                    ),
                    Decision(
                        question="Are LSH candidates used directly?",
                        choice="No — every candidate is verified exactly.",
                        rationale=(
                            "LSH is a filter, not an answer. Its output contains false "
                            "positives by construction. The standard two-stage pattern — "
                            "LSH proposes, exact comparison disposes — keeps precision at "
                            "1.0 against the threshold while retaining the speedup, because "
                            "verification runs over candidates rather than over all pairs."
                        ),
                    ),
                ],
                evidence={
                    "minhash_seconds": round(minhash_seconds, 3),
                    "exact_seconds": round(exact_seconds, 3),
                    "minhash_accuracy": minhash_accuracy,
                    "configurations": configurations,
                },
            )
        )
        crisp.record(
            Phase(
                name="evaluation",
                summary=(
                    f"Best configuration by F1 is b={best['bands']}, r={best['rows']}: "
                    f"recall {best['recall']:.1%} at {best['speedup_vs_exact']:.0f}× the "
                    f"speed of exact search, examining only "
                    f"{best['candidate_fraction_of_exact']:.4%} of all pairs. Highest "
                    f"recall achieved is {highest_recall['recall']:.1%} "
                    f"(b={highest_recall['bands']}, r={highest_recall['rows']})."
                ),
                evidence={
                    "best_by_f1": best,
                    "highest_recall": highest_recall,
                    "exact_pairs_found": len(exact_set),
                    "examples": examples,
                },
                risks=[
                    f"No configuration reaches 100% recall. The best here misses "
                    f"{best['true_pairs_missed']} of {len(exact_set)} true pairs. That is "
                    "not a defect to be tuned away — it is the guarantee LSH offers: "
                    "probabilistic, not exhaustive. Any application where a missed pair is "
                    "unacceptable needs exact search or a different method.",
                    "Speedup is measured at n=1,534, where exact search is already fast. "
                    "The asymptotic argument is what matters: exact search grows as O(n²) "
                    "while LSH grows roughly linearly, so the advantage widens with scale "
                    "and these figures understate it.",
                    "Both stages run in one process on one core. A distributed "
                    "implementation would change the constants substantially.",
                ],
            )
        )
        crisp.record(
            Phase(
                name="deployment",
                summary=(
                    "Signatures, the band sweep, the S-curves and the top exact matches are "
                    "exported so the published page can let a reader move the threshold and "
                    "watch recall and candidate volume trade against each other."
                ),
                evidence={"configurations": len(configurations), "examples": len(examples)},
            )
        )

        print("\n[6/6] Writing artifacts …")
        artifacts.write(OUT / "profile.json", {"profile": profile, "preparation": prep},
                        context=ctx)
        artifacts.write(
            OUT / "results.json",
            {
                "exact": {
                    "pairs_compared": profile["exact_pairs"],
                    "pairs_found": len(exact_set),
                    "seconds": round(exact_seconds, 3),
                    "threshold": SIMILARITY_THRESHOLD,
                },
                "minhash": {
                    "n_permutations": N_PERMUTATIONS,
                    "seconds": round(minhash_seconds, 3),
                    "accuracy": minhash_accuracy,
                },
                "lsh_configurations": configurations,
                "best_by_f1": best,
                "examples": examples,
            },
            context=ctx,
        )
        artifacts.write(OUT / "crispdm.json", crisp.to_dict(), context=ctx)
        artifacts.write(OUT / "provenance.json",
                        {"datasets": [data.provenance_record("consumer_complaints")]},
                        context=ctx)

    print(f"\n✓ {PROJECT} complete — best recall {best['recall']:.1%} at "
          f"{best['speedup_vs_exact']:.0f}× speedup")


if __name__ == "__main__":
    main()
