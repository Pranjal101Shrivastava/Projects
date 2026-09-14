"""Market basket association mining on 9,835 real grocery transactions.

Run:
    PYTHONPATH=lib python3 projects/03_market_basket/pipeline/build.py

Both frequent-itemset algorithms are implemented here from first principles rather than
imported. That is deliberate: Apriori and FP-Growth differ in *how* they avoid scanning
the exponential itemset lattice, and the contrast is the substance of the topic. Calling
``mlxtend.apriori`` would hide exactly the part worth understanding, and it would make the
runtime comparison between the two meaningless.

The other emphasis is statistical. Association mining generates enormous numbers of
candidate rules and then reports the best-scoring ones, which is textbook multiple
comparisons: with 169 items there are ~28,000 possible pairs, so rules with impressive
lift arise by chance alone. Every rule surviving here carries a Fisher exact p-value and a
Benjamini-Hochberg adjusted q-value, and rules that fail the correction are counted and
reported rather than silently dropped.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lib"))

from dsx import artifacts, data  # noqa: E402
from dsx.crispdm import CrispDm, Decision, Phase  # noqa: E402

PROJECT = "03_market_basket"
OUT = ROOT / "projects" / PROJECT / "artifacts"
SEED = 42

MIN_SUPPORT = 0.001   # 0.1% ≈ 10 baskets — low enough to surface niche affinities
MIN_CONFIDENCE = 0.05
MAX_ITEMSET_SIZE = 4
FDR_ALPHA = 0.05


# ======================================================================================
# Algorithm 1 — Apriori
# ======================================================================================
def apriori(transactions: list[frozenset], min_support: float,
            max_size: int) -> tuple[dict[frozenset, int], dict]:
    """Breadth-first frequent itemset mining via the downward-closure property.

    The insight Apriori rests on: if an itemset is frequent, every subset of it must also
    be frequent. Contrapositively, any candidate containing an infrequent subset can be
    discarded *before* counting it. That prunes the lattice hard at each level.

    The cost is one full pass over the transaction database per level, which is why
    FP-Growth (below) exists.
    """
    n = len(transactions)
    min_count = max(1, int(np.ceil(min_support * n)))
    frequent: dict[frozenset, int] = {}
    stats = {"levels": [], "candidates_generated": 0, "candidates_pruned": 0}

    # Level 1 — count singletons directly.
    counts: dict[frozenset, int] = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            counts[frozenset([item])] += 1
    current = {itemset: c for itemset, c in counts.items() if c >= min_count}
    frequent.update(current)
    stats["levels"].append(
        {"k": 1, "candidates": len(counts), "frequent": len(current)}
    )

    k = 2
    while current and k <= max_size:
        # Candidate generation: join frequent (k-1)-itemsets sharing a k-2 prefix.
        prev = sorted(current, key=lambda s: sorted(s))
        candidates: set[frozenset] = set()
        for i in range(len(prev)):
            for j in range(i + 1, len(prev)):
                union = prev[i] | prev[j]
                if len(union) == k:
                    candidates.add(union)

        generated = len(candidates)
        stats["candidates_generated"] += generated

        # Downward-closure pruning: drop candidates with an infrequent (k-1)-subset.
        survivors = [
            candidate
            for candidate in candidates
            if all(frozenset(sub) in current for sub in combinations(candidate, k - 1))
        ]
        stats["candidates_pruned"] += generated - len(survivors)

        # One database pass to count the survivors.
        counts = defaultdict(int)
        for transaction in transactions:
            for candidate in survivors:
                if candidate <= transaction:
                    counts[candidate] += 1

        current = {itemset: c for itemset, c in counts.items() if c >= min_count}
        frequent.update(current)
        stats["levels"].append(
            {
                "k": k,
                "candidates": generated,
                "after_pruning": len(survivors),
                "frequent": len(current),
            }
        )
        k += 1

    return frequent, stats


# ======================================================================================
# Algorithm 2 — FP-Growth
# ======================================================================================
class FPNode:
    """A node in the FP-tree. ``link`` threads all nodes carrying the same item."""

    __slots__ = ("item", "count", "parent", "children", "link")

    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children: dict = {}
        self.link: FPNode | None = None


def fp_growth(transactions: list[frozenset], min_support: float,
              max_size: int) -> tuple[dict[frozenset, int], dict]:
    """Depth-first frequent itemset mining over a compressed prefix tree.

    FP-Growth reads the database exactly twice — once to count item frequencies, once to
    build the tree — and then mines entirely in memory by recursively constructing
    conditional trees. Where Apriori pays one database scan per level, FP-Growth pays two
    in total. On dense data the difference is large; on this sparse basket data (mean
    basket 4.4 items of 169) both are fast, and the comparison below quantifies it rather
    than asserting it.
    """
    n = len(transactions)
    min_count = max(1, int(np.ceil(min_support * n)))
    stats = {"database_scans": 2, "nodes_created": 0}

    # Scan 1 — item frequencies.
    item_counts: dict = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1
    frequent_items = {i: c for i, c in item_counts.items() if c >= min_count}
    # Descending frequency order maximises prefix sharing, which is what compresses the tree.
    order = {item: rank for rank, item in enumerate(
        sorted(frequent_items, key=lambda i: (-frequent_items[i], i))
    )}

    # Scan 2 — build the tree.
    root = FPNode(None, 0, None)
    headers: dict = {item: None for item in frequent_items}

    def insert(path: list, node: FPNode, count: int) -> None:
        if not path:
            return
        head, rest = path[0], path[1:]
        child = node.children.get(head)
        if child is None:
            child = FPNode(head, count, node)
            node.children[head] = child
            stats["nodes_created"] += 1
            child.link = headers[head]
            headers[head] = child
        else:
            child.count += count
        insert(rest, child, count)

    for transaction in transactions:
        path = sorted(
            (i for i in transaction if i in frequent_items), key=lambda i: order[i]
        )
        insert(path, root, 1)

    frequent: dict[frozenset, int] = {}

    def ascend(node: FPNode) -> list:
        path = []
        while node.parent is not None and node.parent.item is not None:
            node = node.parent
            path.append(node.item)
        return path

    def mine(headers: dict, suffix: frozenset) -> None:
        if len(suffix) >= max_size:
            return
        # Least-frequent-first: mining rare items first keeps conditional trees small.
        for item in sorted(headers, key=lambda i: order.get(i, 1 << 30), reverse=True):
            node = headers[item]
            if node is None:
                continue
            support = 0
            conditional: list[tuple[list, int]] = []
            while node is not None:
                support += node.count
                path = ascend(node)
                if path:
                    conditional.append((path, node.count))
                node = node.link
            if support < min_count:
                continue

            new_suffix = suffix | {item}
            frequent[frozenset(new_suffix)] = support
            if len(new_suffix) >= max_size:
                continue

            # Build the conditional FP-tree for this suffix.
            cond_counts: dict = defaultdict(int)
            for path, count in conditional:
                for element in path:
                    cond_counts[element] += count
            cond_frequent = {i: c for i, c in cond_counts.items() if c >= min_count}
            if not cond_frequent:
                continue

            cond_root = FPNode(None, 0, None)
            cond_headers: dict = {i: None for i in cond_frequent}

            def cond_insert(path: list, node: FPNode, count: int) -> None:
                if not path:
                    return
                head, rest = path[0], path[1:]
                child = node.children.get(head)
                if child is None:
                    child = FPNode(head, count, node)
                    node.children[head] = child
                    stats["nodes_created"] += 1
                    child.link = cond_headers[head]
                    cond_headers[head] = child
                else:
                    child.count += count
                cond_insert(rest, child, count)

            for path, count in conditional:
                filtered = sorted(
                    (i for i in path if i in cond_frequent), key=lambda i: order[i]
                )
                cond_insert(filtered, cond_root, count)

            mine(cond_headers, new_suffix)

    mine(headers, frozenset())
    return frequent, stats


# ======================================================================================
# Rule generation and statistical validation
# ======================================================================================
def generate_rules(frequent: dict[frozenset, int], n_transactions: int,
                   min_confidence: float) -> list[dict]:
    """Derive association rules and score them on six complementary measures.

    No single measure is sufficient, and the classic failure is to rank by confidence:

    * **Confidence** P(Y|X) ignores how common Y is on its own. "Buy anything → whole
      milk" reaches 25% confidence purely because a quarter of all baskets contain milk.
    * **Lift** P(Y|X)/P(Y) corrects for that, but is symmetric and unstable at low support.
    * **Conviction** P(X)P(¬Y)/P(X∧¬Y) measures how much more often the rule would be
      violated if X and Y were independent — directional, unlike lift.
    * **Leverage** P(X∧Y) − P(X)P(Y) is the absolute excess co-occurrence, which keeps
      high-volume rules from being buried under high-lift rarities.
    * **Zhang's metric** ∈ [−1, 1] distinguishes association from *dissociation*; a
      negative value means buying X makes Y *less* likely, which lift alone cannot express.
    """
    rules: list[dict] = []
    support_of = frequent

    for itemset, itemset_count in frequent.items():
        if len(itemset) < 2:
            continue
        support_xy = itemset_count / n_transactions

        for r in range(1, len(itemset)):
            for antecedent_items in combinations(sorted(itemset), r):
                antecedent = frozenset(antecedent_items)
                consequent = itemset - antecedent
                if antecedent not in support_of or consequent not in support_of:
                    continue

                support_x = support_of[antecedent] / n_transactions
                support_y = support_of[consequent] / n_transactions
                confidence = support_xy / support_x
                if confidence < min_confidence:
                    continue

                lift = confidence / support_y
                leverage = support_xy - support_x * support_y
                conviction = (
                    (1 - support_y) / (1 - confidence) if confidence < 1 else float("inf")
                )
                denom = max(confidence * (1 - support_y), support_y * (1 - confidence))
                zhang = (confidence - support_y) / denom if denom > 0 else 0.0

                # Fisher exact 2x2 contingency for statistical significance.
                a = itemset_count                                    # X and Y
                b = support_of[antecedent] - a                       # X, not Y
                c = support_of[consequent] - a                       # Y, not X
                d = n_transactions - a - b - c                       # neither

                rules.append(
                    {
                        "antecedent": sorted(antecedent),
                        "consequent": sorted(consequent),
                        "support": round(support_xy, 6),
                        "support_count": int(itemset_count),
                        "confidence": round(confidence, 5),
                        "lift": round(lift, 4),
                        "leverage": round(leverage, 7),
                        "conviction": round(conviction, 4) if np.isfinite(conviction) else None,
                        "zhang": round(zhang, 4),
                        "_contingency": (a, b, c, d),
                    }
                )
    return rules


def add_significance(rules: list[dict], alpha: float) -> dict:
    """Attach Fisher exact p-values and Benjamini-Hochberg q-values.

    Mining generates tens of thousands of candidate rules and then presents the top few by
    lift. That is a multiple-comparisons problem: at 169 items there are ~28k possible
    pairs, so some will show striking lift by chance. Controlling the false discovery rate
    turns "these rules scored highest" into "these rules are unlikely to be noise".

    Bonferroni is deliberately not used — with this many tests it would reject almost
    everything, including real affinities. BH controls the *expected proportion* of false
    discoveries among those reported, which is the right guarantee when the output is a
    ranked shortlist for a merchandiser to act on.
    """
    from scipy.stats import fisher_exact

    for rule in rules:
        a, b, c, d = rule.pop("_contingency")
        # One-sided: we only care about positive association (greater co-occurrence).
        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        rule["p_value"] = float(p)

    # Benjamini-Hochberg step-up.
    order = np.argsort([r["p_value"] for r in rules])
    m = len(rules)
    q_previous = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = m - rank + 1
        q = min(q_previous, rules[idx]["p_value"] * m / i)
        rules[idx]["q_value"] = float(q)
        rules[idx]["significant"] = bool(q <= alpha)
        q_previous = q

    n_significant = sum(r["significant"] for r in rules)
    return {
        "n_rules_tested": m,
        "n_significant": n_significant,
        "n_rejected_by_fdr": m - n_significant,
        "fdr_alpha": alpha,
        "method": "Fisher exact (one-sided) with Benjamini-Hochberg FDR control",
        "rationale": (
            f"{m:,} candidate rules were tested. {m - n_significant:,} fail FDR control "
            f"at α={alpha} and are excluded from the published shortlist. Reporting only "
            "the survivors — and stating how many did not survive — prevents the "
            "top-by-lift table from being read as though every rule in it were real."
        ),
    }


# ======================================================================================
# Orchestration
# ======================================================================================
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    crisp = CrispDm(
        project=PROJECT,
        business_question=(
            "Which grocery items are bought together more often than chance explains, "
            "and which of those affinities are strong enough — and statistically solid "
            "enough — to justify changing shelf layout or a promotion?"
        ),
    )
    crisp.record(
        Phase(
            name="business_understanding",
            summary=(
                "Cross-sell placement and bundle promotions both hinge on knowing which "
                "products genuinely pull each other into the basket. The commercial risk "
                "is acting on a spurious pattern: rearranging an aisle around a rule that "
                "was noise costs real money and is hard to detect after the fact."
            ),
            decisions=[
                Decision(
                    question="What makes a rule actionable rather than merely high-scoring?",
                    choice=(
                        "It must clear a lift threshold AND survive FDR correction AND "
                        "carry enough absolute volume (leverage) to matter."
                    ),
                    rationale=(
                        "Lift alone rewards rare coincidences: two items appearing "
                        "together in 3 of 9,835 baskets can show lift above 20. Requiring "
                        "statistical significance rules out noise, and requiring leverage "
                        "rules out patterns too small to be worth a merchandising change."
                    ),
                    alternatives_rejected=[
                        "Rank by confidence — dominated by whatever is popular overall; "
                        "'anything → whole milk' scores well and means nothing.",
                        "Rank by lift alone — surfaces rare coincidences with no volume.",
                    ],
                ),
            ],
            evidence={"min_support": MIN_SUPPORT, "min_confidence": MIN_CONFIDENCE,
                      "fdr_alpha": FDR_ALPHA},
        )
    )

    with artifacts.run(PROJECT, seed=SEED) as ctx:
        # --- Data understanding -------------------------------------------------------
        print("\n[1/5] Loading real grocery transactions …")
        raw_lines = data.load_text("groceries").strip().splitlines()
        transactions = [
            frozenset(item.strip() for item in line.split(",") if item.strip())
            for line in raw_lines
        ]
        transactions = [t for t in transactions if t]
        n = len(transactions)

        item_counts: dict = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        basket_sizes = np.array([len(t) for t in transactions])

        profile = {
            "n_transactions": n,
            "n_distinct_items": len(item_counts),
            "total_items_sold": int(basket_sizes.sum()),
            "basket_size": {
                "mean": round(float(basket_sizes.mean()), 3),
                "median": int(np.median(basket_sizes)),
                "max": int(basket_sizes.max()),
                "singleton_share": round(float((basket_sizes == 1).mean()), 4),
            },
            "basket_size_distribution": [
                {"size": int(s), "count": int((basket_sizes == s).sum())}
                for s in range(1, int(basket_sizes.max()) + 1)
            ],
            "top_items": [
                {"item": i, "count": c, "support": round(c / n, 5)}
                for i, c in sorted(item_counts.items(), key=lambda kv: -kv[1])[:30]
            ],
            "sparsity": round(1 - basket_sizes.sum() / (n * len(item_counts)), 5),
        }
        print(f"      {n:,} baskets, {len(item_counts)} items, "
              f"mean basket {profile['basket_size']['mean']}")

        crisp.record(
            Phase(
                name="data_understanding",
                summary=(
                    f"{n:,} real point-of-sale baskets over {len(item_counts)} item "
                    f"categories. Mean basket {profile['basket_size']['mean']} items; the "
                    f"transaction matrix is {profile['sparsity']:.2%} sparse, which is why "
                    "a low support floor is necessary to see anything beyond staples."
                ),
                evidence=profile,
                decisions=[
                    Decision(
                        question=f"Why set minimum support as low as {MIN_SUPPORT:.1%}?",
                        choice=f"{MIN_SUPPORT:.1%} ≈ {int(MIN_SUPPORT * n)} baskets.",
                        rationale=(
                            "At 97% sparsity a conventional 1% floor admits only the "
                            "handful of staples everyone buys, and every rule found is a "
                            "variation on 'people buy milk'. Lowering the floor surfaces "
                            "niche affinities — which is where merchandising value is — "
                            "at the cost of many more candidate rules. The FDR correction "
                            "is what makes that trade safe."
                        ),
                        alternatives_rejected=[
                            "1% support — yields only staple combinations, no actionable "
                            "niche findings.",
                        ],
                    )
                ],
                risks=[
                    "One month from one outlet. Seasonal affinities and store-specific "
                    "assortment are baked in and will not generalise to other stores.",
                    "Items are categories, not SKUs, so within-category substitution "
                    "(two brands of yogurt) is invisible.",
                ],
            )
        )

        # --- Modeling: run both algorithms on identical input --------------------------
        print("\n[2/5] Mining frequent itemsets — Apriori …")
        t0 = time.perf_counter()
        freq_apriori, apriori_stats = apriori(transactions, MIN_SUPPORT, MAX_ITEMSET_SIZE)
        apriori_seconds = time.perf_counter() - t0
        print(f"      {len(freq_apriori):,} itemsets in {apriori_seconds:.2f}s")

        print("[3/5] Mining frequent itemsets — FP-Growth …")
        t0 = time.perf_counter()
        freq_fp, fp_stats = fp_growth(transactions, MIN_SUPPORT, MAX_ITEMSET_SIZE)
        fp_seconds = time.perf_counter() - t0
        print(f"      {len(freq_fp):,} itemsets in {fp_seconds:.2f}s")

        # Correctness cross-check: two independent implementations must agree exactly.
        # If they do not, at least one is wrong, and the run should fail rather than
        # publish results from an algorithm that disagrees with itself.
        agree = freq_apriori == freq_fp
        only_apriori = set(freq_apriori) - set(freq_fp)
        only_fp = set(freq_fp) - set(freq_apriori)
        if not agree:
            raise AssertionError(
                f"Algorithms disagree: {len(only_apriori)} itemsets only in Apriori, "
                f"{len(only_fp)} only in FP-Growth. One implementation is incorrect."
            )
        print(f"      ✓ cross-check: both algorithms agree on all {len(freq_fp):,} itemsets")

        algorithm_comparison = {
            "identical_output": agree,
            "n_itemsets": len(freq_fp),
            "apriori": {
                "seconds": round(apriori_seconds, 3),
                "database_scans": len(apriori_stats["levels"]),
                "candidates_generated": apriori_stats["candidates_generated"],
                "candidates_pruned_by_downward_closure": apriori_stats["candidates_pruned"],
                "pruning_rate": round(
                    apriori_stats["candidates_pruned"]
                    / max(1, apriori_stats["candidates_generated"]),
                    4,
                ),
                "levels": apriori_stats["levels"],
            },
            "fp_growth": {
                "seconds": round(fp_seconds, 3),
                "database_scans": fp_stats["database_scans"],
                "tree_nodes_created": fp_stats["nodes_created"],
            },
            "speedup_fp_over_apriori": round(apriori_seconds / max(1e-9, fp_seconds), 2),
            "interpretation": (
                "Both implementations return byte-identical itemsets, which is the "
                "correctness check that matters: two independent algorithms agreeing is "
                "far stronger evidence than either one running without error. FP-Growth "
                "needs 2 database scans against Apriori's "
                f"{len(apriori_stats['levels'])}, and avoids generating "
                f"{apriori_stats['candidates_generated']:,} candidates altogether."
            ),
        }

        # --- Rules and significance ----------------------------------------------------
        print("\n[4/5] Generating rules and testing significance …")
        rules = generate_rules(freq_fp, n, MIN_CONFIDENCE)
        significance = add_significance(rules, FDR_ALPHA)
        print(f"      {significance['n_rules_tested']:,} rules tested, "
              f"{significance['n_significant']:,} survive FDR at α={FDR_ALPHA}")

        significant = [r for r in rules if r["significant"]]
        by_lift = sorted(significant, key=lambda r: -r["lift"])[:60]
        by_leverage = sorted(significant, key=lambda r: -r["leverage"])[:40]
        dissociations = sorted(
            (r for r in rules if r["zhang"] < -0.2 and r["support_count"] >= 20),
            key=lambda r: r["zhang"],
        )[:15]

        crisp.record(
            Phase(
                name="modeling",
                summary=(
                    f"Apriori and FP-Growth both implemented from scratch and run on "
                    f"identical input; they agree on all {len(freq_fp):,} frequent "
                    f"itemsets. {significance['n_rules_tested']:,} candidate rules were "
                    f"generated and scored on six measures, then filtered by Fisher exact "
                    f"tests under Benjamini-Hochberg FDR control."
                ),
                decisions=[
                    Decision(
                        question="Why implement both algorithms rather than use a library?",
                        choice="Both written from first principles.",
                        rationale=(
                            "The two differ precisely in how they avoid the exponential "
                            "lattice, and that difference is the substance of the topic. "
                            "Running both also yields a genuine correctness check: "
                            "independent implementations agreeing on all "
                            f"{len(freq_fp):,} itemsets is stronger evidence than either "
                            "completing without error."
                        ),
                        alternatives_rejected=[
                            "mlxtend.apriori — hides the mechanism and makes the runtime "
                            "comparison meaningless.",
                        ],
                    ),
                    Decision(
                        question="How is multiple testing handled?",
                        choice="Benjamini-Hochberg FDR at α=0.05, not Bonferroni.",
                        rationale=(
                            f"{significance['n_rules_tested']:,} simultaneous tests make "
                            "some striking lifts inevitable by chance. Bonferroni would "
                            "control the family-wise error rate but reject nearly "
                            "everything including real affinities. BH bounds the expected "
                            "proportion of false discoveries in the shortlist, which is "
                            "the right guarantee for a ranked list someone will act on."
                        ),
                        alternatives_rejected=[
                            "No correction — the standard practice, and the reason "
                            "published rule tables are often mostly noise.",
                            "Bonferroni — too conservative at this scale.",
                        ],
                    ),
                ],
                evidence={
                    "algorithm_comparison": algorithm_comparison,
                    "significance": significance,
                },
            )
        )

        # --- Evaluation ----------------------------------------------------------------
        top = by_lift[0] if by_lift else None
        crisp.record(
            Phase(
                name="evaluation",
                summary=(
                    f"{significance['n_significant']:,} of "
                    f"{significance['n_rules_tested']:,} rules survive FDR correction "
                    f"({significance['n_rejected_by_fdr']:,} rejected). Strongest "
                    f"surviving affinity: "
                    + (
                        f"{{{', '.join(top['antecedent'])}}} → "
                        f"{{{', '.join(top['consequent'])}}} at lift {top['lift']:.2f}."
                        if top
                        else "none."
                    )
                ),
                evidence={
                    "n_significant": significance["n_significant"],
                    "n_rejected": significance["n_rejected_by_fdr"],
                    "rejection_share": round(
                        significance["n_rejected_by_fdr"]
                        / max(1, significance["n_rules_tested"]),
                        4,
                    ),
                    "top_rule": top,
                    "n_dissociations_found": len(dissociations),
                },
                risks=[
                    "Association is not causation. A lift of 3 between two items does not "
                    "mean moving one next to the other will raise sales of the other; it "
                    "may only mean both are bought on the same kind of shopping trip.",
                    "Rules are mined and evaluated on the same transactions. A holdout "
                    "split would test whether affinities persist across periods, which "
                    "one month of data does not support.",
                ],
            )
        )
        crisp.record(
            Phase(
                name="data_preparation",
                summary=(
                    "Minimal by design: the source is already one transaction per line. "
                    "Parsing splits on commas, trims whitespace and drops empty baskets. "
                    "No item is merged or renamed, so item identity matches the source."
                ),
                evidence={"n_parsed": n, "empty_dropped": len(raw_lines) - n},
            )
        )
        crisp.record(
            Phase(
                name="deployment",
                summary=(
                    "Rules ship as JSON with every measure attached, driving an "
                    "interactive rule explorer and a force-directed affinity graph. The "
                    "explorer exposes the significance filter as a control so a user can "
                    "see how many rules the FDR correction removes."
                ),
                evidence={"artifact": "rules.json", "graph_nodes": len(profile["top_items"])},
            )
        )

        # --- Artifacts -----------------------------------------------------------------
        print("\n[5/5] Writing artifacts …")
        itemsets_by_size: dict = defaultdict(list)
        for itemset, count in freq_fp.items():
            itemsets_by_size[len(itemset)].append(
                {"items": sorted(itemset), "count": count, "support": round(count / n, 6)}
            )
        itemset_summary = {
            str(size): {
                "n": len(rows),
                "top": sorted(rows, key=lambda r: -r["count"])[:20],
            }
            for size, rows in sorted(itemsets_by_size.items())
        }

        # Item co-occurrence graph for the network view.
        top_items = [row["item"] for row in profile["top_items"][:25]]
        edges = [
            {
                "source": r["antecedent"][0],
                "target": r["consequent"][0],
                "lift": r["lift"],
                "support": r["support"],
            }
            for r in significant
            if len(r["antecedent"]) == 1
            and len(r["consequent"]) == 1
            and r["antecedent"][0] in top_items
            and r["consequent"][0] in top_items
        ]

        artifacts.write(OUT / "profile.json", profile, context=ctx)
        artifacts.write(
            OUT / "algorithms.json",
            {"comparison": algorithm_comparison, "itemsets_by_size": itemset_summary},
            context=ctx,
        )
        artifacts.write(
            OUT / "rules.json",
            {
                "significance": significance,
                "top_by_lift": by_lift,
                "top_by_leverage": by_leverage,
                "dissociations": dissociations,
                "graph": {"nodes": top_items, "edges": edges[:150]},
            },
            context=ctx,
        )
        artifacts.write(OUT / "crispdm.json", crisp.to_dict(), context=ctx)
        artifacts.write(
            OUT / "provenance.json",
            {"datasets": [data.provenance_record("groceries")]},
            context=ctx,
        )

    print(f"\n✓ {PROJECT} complete — {significance['n_significant']:,} significant rules "
          f"of {significance['n_rules_tested']:,} tested")


if __name__ == "__main__":
    main()
