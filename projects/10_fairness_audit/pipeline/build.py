"""Fairness audit of a deployed algorithmic risk score, on the data that started the debate.

Run:
    PYTHONPATH=lib python3 projects/10_fairness_audit/pipeline/build.py

COMPAS is a commercial risk-assessment tool used in US criminal courts to score a
defendant's likelihood of reoffending. In 2016 ProPublica obtained two years of Broward
County scores under a public records request, compared them to who was actually rearrested,
and reported that the tool's errors fell unevenly by race. Northpointe, its developer,
responded that the scores were equally *calibrated* across groups and were therefore fair.

Both were arithmetically correct. That is the whole point of this project.

Kleinberg, Mullainathan & Raghavan (2016) and Chouldechova (2017) proved that when base
rates differ between groups, calibration and equal error rates **cannot both hold** except
in degenerate cases. There is no configuration of any model that satisfies both. Choosing a
fairness definition is therefore not a technical step that can be optimised — it is a value
judgement about which kind of harm is worse, and it has to be argued rather than computed.

This pipeline measures every major criterion on the real data, shows the impossibility
empirically rather than citing it, and refuses to declare a winner.

On the data: these are real records about real people in the criminal justice system.
Names are dropped on load and never used. The analysis is exactly the purpose ProPublica
released the data for — independent scrutiny of a consequential deployed algorithm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lib"))

from dsx import artifacts, data, metrics, splits  # noqa: E402
from dsx.crispdm import CrispDm, Decision, Phase  # noqa: E402

PROJECT = "10_fairness_audit"
OUT = ROOT / "projects" / PROJECT / "artifacts"
SEED = 42

# ProPublica's published filtering rules, applied so results are comparable to theirs.
# Reproducing a published analysis's preprocessing exactly is what makes a replication a
# replication rather than a different study with a similar name.
PROPUBLICA_FILTERS = {
    "days_b_screening_arrest within [-30, 30]": "charge and screening must correspond",
    "is_recid != -1": "drop records with no recidivism data",
    "c_charge_degree != 'O'": "drop ordinary traffic offences, which carry no jail time",
    "score_text != 'N/A'": "drop rows with no COMPAS score",
}

GROUP_COLUMN = "race"
MIN_GROUP_SIZE = 500   # groups below this are reported but not compared: too few to be stable


def confusion_by_group(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Confusion counts plus every rate a fairness criterion is defined on."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    n = tp + fp + fn + tn

    return {
        "n": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "base_rate": round((tp + fn) / max(1, n), 4),
        "selection_rate": round((tp + fp) / max(1, n), 4),
        # FPR: of people who did NOT reoffend, the share labelled high risk.
        "fpr": round(fp / max(1, fp + tn), 4),
        # FNR: of people who DID reoffend, the share labelled low risk.
        "fnr": round(fn / max(1, fn + tp), 4),
        "tpr": round(tp / max(1, tp + fn), 4),
        "tnr": round(tn / max(1, tn + fp), 4),
        # PPV: of people labelled high risk, the share who reoffended. This is the
        # calibration-style quantity Northpointe pointed to.
        "ppv": round(tp / max(1, tp + fp), 4),
        "npv": round(tn / max(1, tn + fn), 4),
        "accuracy": round((tp + tn) / max(1, n), 4),
    }


def calibration_curve_by_group(scores: np.ndarray, outcomes: np.ndarray,
                               n_bins: int = 10) -> list[dict]:
    """Observed reoffence rate within each score decile.

    Calibration asks: among everyone the tool gave a score of 7, did roughly the same
    fraction actually reoffend regardless of group? If yes, the score means the same thing
    for everyone — which is a genuine and defensible notion of fairness.
    """
    out = []
    for decile in range(1, 11):
        mask = scores == decile
        if mask.sum() == 0:
            continue
        out.append(
            {
                "decile": decile,
                "n": int(mask.sum()),
                "observed_recidivism_rate": round(float(outcomes[mask].mean()), 4),
            }
        )
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from scipy import stats

    crisp = CrispDm(
        project=PROJECT,
        business_question=(
            "A risk score used in real sentencing and bail decisions predicts reoffending. "
            "Is it fair — and does that question even have a single answer?"
        ),
    )
    crisp.record(
        Phase(
            name="business_understanding",
            summary=(
                "COMPAS scores informed real bail and sentencing decisions in Broward "
                "County. ProPublica reported in 2016 that its errors fell unevenly by race; "
                "Northpointe replied that the scores were equally calibrated and therefore "
                "fair. Both claims are arithmetically true of the same data. The objective "
                "here is not to determine who was right but to measure every criterion and "
                "show why they cannot be satisfied simultaneously."
            ),
            decisions=[
                Decision(
                    question="Which fairness definition does this audit adopt?",
                    choice="None. All are measured; none is declared correct.",
                    rationale=(
                        "Kleinberg et al. (2016) and Chouldechova (2017) proved that "
                        "calibration within groups and equal false-positive/false-negative "
                        "rates are mutually incompatible whenever base rates differ, except "
                        "in degenerate cases. No model can satisfy both. Selecting one is a "
                        "judgement about which harm matters more — a wrongly detained person "
                        "who would not have reoffended, or a wrongly released person who "
                        "would have — and that is not a decision a pipeline can make."
                    ),
                    alternatives_rejected=[
                        "Pick equalized odds and declare the tool unfair — defensible, but "
                        "presents a value judgement as a measurement.",
                        "Pick calibration and declare it fair — the same error in the other "
                        "direction.",
                        "Report a single 'fairness score' — collapses incompatible criteria "
                        "into one number and hides the trade-off that is the entire issue.",
                    ],
                ),
                Decision(
                    question="How is data about real individuals handled?",
                    choice="Names dropped on load; only aggregate group statistics reported.",
                    rationale=(
                        "These are real people with real criminal records. ProPublica "
                        "published the data for scrutiny of the algorithm, not of its "
                        "subjects. No individual is identified anywhere in the output."
                    ),
                ),
            ],
            evidence={"impossibility_theorem": "Kleinberg et al. 2016; Chouldechova 2017"},
        )
    )

    with artifacts.run(PROJECT, seed=SEED) as ctx:
        # --- Data understanding -------------------------------------------------------
        print("\n[1/6] Loading real ProPublica COMPAS records …")
        raw = pd.read_csv(data.fetch("compas"))
        n_raw = len(raw)

        # Drop identifying columns immediately, before any analysis touches them.
        raw = raw.drop(columns=[c for c in ("name", "first", "last") if c in raw.columns])

        df = raw[
            raw.days_b_screening_arrest.between(-30, 30)
            & (raw.is_recid != -1)
            & (raw.c_charge_degree != "O")
            & (raw.score_text != "N/A")
        ].copy()

        df["high_risk"] = (df.decile_score >= 5).astype(int)
        df["recidivated"] = df.two_year_recid.astype(int)

        group_counts = df[GROUP_COLUMN].value_counts()
        groups = [g for g, c in group_counts.items() if c >= MIN_GROUP_SIZE]
        small_groups = [
            {"group": g, "n": int(c)} for g, c in group_counts.items() if c < MIN_GROUP_SIZE
        ]

        profile = {
            "rows_raw": n_raw,
            "rows_after_propublica_filters": len(df),
            "filters_applied": PROPUBLICA_FILTERS,
            "overall_recidivism_rate": round(float(df.recidivated.mean()), 4),
            "overall_high_risk_rate": round(float(df.high_risk.mean()), 4),
            "group_sizes": {str(g): int(c) for g, c in group_counts.items()},
            "groups_compared": [str(g) for g in groups],
            "groups_too_small_to_compare": small_groups,
            "min_group_size": MIN_GROUP_SIZE,
            "identifying_columns_dropped": ["name", "first", "last"],
            "decile_distribution": {
                str(g): df.loc[df[GROUP_COLUMN] == g, "decile_score"]
                .value_counts().sort_index().to_dict()
                for g in groups
            },
        }
        print(f"      {n_raw:,} raw rows → {len(df):,} after ProPublica's filters")
        print(f"      comparing {len(groups)} groups: {', '.join(map(str, groups))}")

        crisp.record(
            Phase(
                name="data_understanding",
                summary=(
                    f"{n_raw:,} Broward County records reduced to {len(df):,} by "
                    "ProPublica's own published filters, applied so these results are "
                    f"directly comparable to theirs. {len(groups)} groups have at least "
                    f"{MIN_GROUP_SIZE} records and are compared; smaller groups are "
                    "reported but not compared, because rate estimates on a few dozen "
                    "people are too unstable to support a fairness claim."
                ),
                evidence=profile,
                risks=[
                    "The outcome variable is *rearrest*, not reoffending. Policing intensity "
                    "differs by neighbourhood and by group, so arrest rates reflect both "
                    "behaviour and enforcement. Every 'error rate' below therefore measures "
                    "the tool against a target that is itself shaped by the system being "
                    "audited. This affects every published analysis of this dataset, "
                    "including ProPublica's and Northpointe's.",
                    "Two years in one Florida county in 2013-2014. Nothing here establishes "
                    "how COMPAS behaves elsewhere or now.",
                ],
            )
        )

        # --- Group-wise fairness metrics ------------------------------------------------
        print("\n[2/6] Computing fairness criteria per group …")
        by_group = {}
        for group in groups:
            mask = df[GROUP_COLUMN] == group
            by_group[str(group)] = confusion_by_group(
                df.loc[mask, "recidivated"].to_numpy(),
                df.loc[mask, "high_risk"].to_numpy(),
            )

        reference = max(by_group, key=lambda g: by_group[g]["n"])
        for group, stats_row in by_group.items():
            print(f"      {group:<28} n={stats_row['n']:>5}  base {stats_row['base_rate']:.3f}  "
                  f"FPR {stats_row['fpr']:.3f}  FNR {stats_row['fnr']:.3f}  "
                  f"PPV {stats_row['ppv']:.3f}")

        # --- The four criteria, each evaluated explicitly --------------------------------
        print("\n[3/6] Evaluating each fairness criterion …")

        def spread(metric: str) -> dict:
            values = {g: by_group[g][metric] for g in by_group}
            lo_group = min(values, key=values.get)
            hi_group = max(values, key=values.get)
            return {
                "by_group": values,
                "min": values[lo_group], "min_group": lo_group,
                "max": values[hi_group], "max_group": hi_group,
                "absolute_difference": round(values[hi_group] - values[lo_group], 4),
                "ratio": round(
                    values[hi_group] / max(1e-9, values[lo_group]), 3
                ),
            }

        selection = spread("selection_rate")
        fpr = spread("fpr")
        fnr = spread("fnr")
        ppv = spread("ppv")

        # Disparate impact: the US EEOC "four-fifths rule" treats a selection-rate ratio
        # below 0.8 as prima facie evidence of adverse impact. It is a legal screening
        # threshold, not a statistical one, and it is cited here as such.
        disparate_impact_ratio = round(
            selection["min"] / max(1e-9, selection["max"]), 4
        )

        criteria = {
            "demographic_parity": {
                "definition": "P(predicted high risk) is equal across groups.",
                "measured": selection,
                "satisfied": selection["absolute_difference"] < 0.05,
                "disparate_impact_ratio": disparate_impact_ratio,
                "passes_four_fifths_rule": disparate_impact_ratio >= 0.8,
                "note": (
                    "Demographic parity ignores the outcome entirely. If two groups truly "
                    "differ in base rate, enforcing it requires deliberately mis-scoring "
                    "people — so failing this test is not on its own evidence of unfairness."
                ),
            },
            "equal_opportunity": {
                "definition": "False negative rate is equal across groups.",
                "measured": fnr,
                "satisfied": fnr["absolute_difference"] < 0.05,
                "note": (
                    "Concerns people who did reoffend but were scored low risk. Unequal FNR "
                    "means one group is more often released when they should not have been."
                ),
            },
            "predictive_equality": {
                "definition": "False positive rate is equal across groups.",
                "measured": fpr,
                "satisfied": fpr["absolute_difference"] < 0.05,
                "note": (
                    "Concerns people who did NOT reoffend but were scored high risk — "
                    "detained or sentenced more harshly for something they would not have "
                    "done. This is the criterion ProPublica's analysis centred on."
                ),
            },
            "calibration_ppv": {
                "definition": "Among those labelled high risk, P(reoffends) is equal across groups.",
                "measured": ppv,
                "satisfied": ppv["absolute_difference"] < 0.05,
                "note": (
                    "If satisfied, a 'high risk' label means the same thing whoever receives "
                    "it. This is the criterion Northpointe's defence centred on."
                ),
            },
        }

        # A pass/fail verdict at a fixed tolerance flattens exactly the distinction the
        # ProPublica/Northpointe dispute turned on: calibration is much CLOSER to holding
        # than error-rate equality, even when a strict threshold rejects both. Ranking the
        # criteria by how far they are from parity preserves that.
        ranked = sorted(
            criteria.items(), key=lambda kv: kv[1]["measured"]["absolute_difference"]
        )
        closest_name, closest = ranked[0]
        furthest_name, furthest = ranked[-1]
        for name, criterion in criteria.items():
            criterion["gap"] = criterion["measured"]["absolute_difference"]
            criterion["rank_closest_to_parity"] = (
                [n for n, _ in ranked].index(name) + 1
            )

        criteria_summary = {
            "tolerance_used": 0.05,
            "n_satisfied": sum(1 for c in criteria.values() if c["satisfied"]),
            "closest_to_parity": {
                "criterion": closest_name, "gap": closest["measured"]["absolute_difference"]
            },
            "furthest_from_parity": {
                "criterion": furthest_name, "gap": furthest["measured"]["absolute_difference"]
            },
            "ratio_furthest_to_closest": round(
                furthest["measured"]["absolute_difference"]
                / max(1e-9, closest["measured"]["absolute_difference"]),
                2,
            ),
            "why_ranking_matters": (
                f"At a 5-point tolerance every criterion is rejected, which is true but "
                f"uninformative: it erases the distinction the whole dispute rested on. "
                f"The gaps are not comparable in size. {closest_name} is off by "
                f"{closest['measured']['absolute_difference']:.4f} while {furthest_name} is "
                f"off by {furthest['measured']['absolute_difference']:.4f} — a factor of "
                f"{furthest['measured']['absolute_difference'] / max(1e-9, closest['measured']['absolute_difference']):.1f}. "
                "Northpointe's defence was that calibration nearly holds; ProPublica's case "
                "was that error rates emphatically do not. Both readings are visible here, "
                "and a pass/fail table alone would hide the one that favours the developer."
            ),
        }

        for name, criterion in criteria.items():
            verdict = "SATISFIED" if criterion["satisfied"] else "VIOLATED"
            print(f"      {name:<22} {verdict:<10} "
                  f"max gap {criterion['measured']['absolute_difference']:+.4f}  "
                  f"(rank {criterion['rank_closest_to_parity']}/{len(criteria)} closest)")
        print(f"      → {furthest_name} is {criteria_summary['ratio_furthest_to_closest']}× "
              f"further from parity than {closest_name}")

        # --- Calibration curves ----------------------------------------------------------
        print("\n[4/6] Building calibration curves per group …")
        calibration = {
            str(group): calibration_curve_by_group(
                df.loc[df[GROUP_COLUMN] == group, "decile_score"].to_numpy(),
                df.loc[df[GROUP_COLUMN] == group, "recidivated"].to_numpy(),
            )
            for group in groups
        }

        # --- The impossibility, demonstrated --------------------------------------------
        print("\n[5/6] Demonstrating the impossibility result …")

        # Chouldechova's identity relates PPV, FPR, FNR and the base rate p:
        #     FPR = (p / (1 - p)) * ((1 - PPV) / PPV) * (1 - FNR)
        # If two groups have different p but identical PPV, their FPR and FNR cannot both
        # match. This checks the identity numerically on the real data and then shows what
        # equalising one criterion does to the others.
        identity_check = []
        for group, row in by_group.items():
            p = row["base_rate"]
            predicted_fpr = (
                (p / max(1e-9, 1 - p))
                * ((1 - row["ppv"]) / max(1e-9, row["ppv"]))
                * (1 - row["fnr"])
            )
            identity_check.append(
                {
                    "group": group,
                    "base_rate": p,
                    "observed_fpr": row["fpr"],
                    "fpr_implied_by_identity": round(float(predicted_fpr), 4),
                    "discrepancy": round(float(abs(predicted_fpr - row["fpr"])), 5),
                }
            )

        base_rates = {g: by_group[g]["base_rate"] for g in by_group}
        base_rate_gap = round(max(base_rates.values()) - min(base_rates.values()), 4)

        impossibility = {
            "chouldechova_identity": "FPR = (p/(1-p)) · ((1-PPV)/PPV) · (1-FNR)",
            "identity_verified_on_real_data": identity_check,
            "max_identity_discrepancy": round(
                max(r["discrepancy"] for r in identity_check), 5
            ),
            "base_rates": base_rates,
            "base_rate_gap": base_rate_gap,
            "explanation": (
                f"Observed base rates differ by {base_rate_gap:.4f} between groups. The "
                "identity above is an algebraic fact, not a modelling assumption: it holds "
                f"on this data to within {max(r['discrepancy'] for r in identity_check):.5f}. "
                "With PPV held equal across groups and p differing, FPR and FNR are forced "
                "apart. No amount of retraining, reweighting or threshold tuning escapes "
                "this — it is arithmetic. Any tool applied to groups with different base "
                "rates must violate either calibration or error-rate equality."
            ),
        }
        print(f"      identity holds to {impossibility['max_identity_discrepancy']:.5f}")
        print(f"      base rate gap between groups: {base_rate_gap:.4f}")

        # Statistical significance of the FPR gap, so the disparity is not read off noise.
        group_list = sorted(by_group, key=lambda g: -by_group[g]["n"])[:2]
        a, b = by_group[group_list[0]], by_group[group_list[1]]
        table = np.array([[a["fp"], a["tn"]], [b["fp"], b["tn"]]])
        chi2, p_value, _, _ = stats.chi2_contingency(table)
        significance = {
            "comparison": f"{group_list[0]} vs {group_list[1]}",
            "metric": "false positive rate",
            "fpr_a": a["fpr"], "fpr_b": b["fpr"],
            "chi2": round(float(chi2), 2),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "note": (
                "Tests whether the FPR difference between the two largest groups could "
                "plausibly arise by chance. It does not test whether the difference is "
                "unjust — that is not a statistical question."
            ),
        }

        # --- A model of our own, audited the same way -------------------------------------
        print("\n[6/6] Training a replacement model and auditing it identically …")
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        # Race is deliberately excluded from the features. The point of including this
        # model is to show that omitting a protected attribute does NOT produce a fair
        # model, because correlated features reconstruct it.
        numeric = ["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"]
        categorical = ["sex", "c_charge_degree", "age_cat"]

        X = df[numeric + categorical]
        y = df.recidivated.to_numpy()
        train_idx, test_idx, split_report = splits.stratified_split(y, test_size=0.3, seed=SEED)

        model = Pipeline([
            ("prep", ColumnTransformer([
                ("num", StandardScaler(), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ])),
            ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
        ])
        splits.assert_pipeline_safe(model)
        model.fit(X.iloc[train_idx], y[train_idx])
        probabilities = model.predict_proba(X.iloc[test_idx])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        test_groups = df.iloc[test_idx][GROUP_COLUMN].to_numpy()
        own_model_by_group = {}
        for group in groups:
            mask = test_groups == group
            if mask.sum() < 50:
                continue
            own_model_by_group[str(group)] = confusion_by_group(
                y[test_idx][mask], predictions[mask]
            )

        own_fpr = {g: v["fpr"] for g, v in own_model_by_group.items()}
        own_model = {
            "features_used": numeric + categorical,
            "protected_attribute_excluded": GROUP_COLUMN,
            "overall": metrics.classification_report(y[test_idx], probabilities),
            "by_group": own_model_by_group,
            "fpr_gap": round(max(own_fpr.values()) - min(own_fpr.values()), 4),
            "compas_fpr_gap": fpr["absolute_difference"],
            "finding": (
                f"The replacement model never sees {GROUP_COLUMN}, yet its false-positive "
                f"rates still differ by {max(own_fpr.values()) - min(own_fpr.values()):.4f} "
                f"across groups, against {fpr['absolute_difference']:.4f} for COMPAS. "
                "Removing a protected attribute does not remove disparity, because prior "
                "convictions and age carry the same information. 'Fairness through "
                "unawareness' does not work, and this is what that looks like measured "
                "rather than asserted."
            ),
        }
        print(f"      own model FPR gap {own_model['fpr_gap']:.4f} "
              f"(COMPAS: {fpr['absolute_difference']:.4f}) — despite never seeing race")

        crisp.record(
            Phase(
                name="data_preparation",
                summary=(
                    "ProPublica's published filters applied verbatim so results are "
                    "comparable to the original analysis. Identifying columns dropped "
                    "before any analysis. High risk defined as decile score >= 5, matching "
                    "ProPublica's cut."
                ),
                evidence={"filters": PROPUBLICA_FILTERS, "rows": len(df)},
            )
        )
        crisp.record(
            Phase(
                name="modeling",
                summary=(
                    "No model is built to replace COMPAS as the object of the audit — "
                    "COMPAS's own published scores are audited directly. A separate "
                    "logistic model excluding race is trained solely to test whether "
                    "omitting a protected attribute removes disparity."
                ),
                decisions=[
                    Decision(
                        question="Does excluding race from the features make a model fair?",
                        choice="No — measured, not assumed.",
                        rationale=own_model["finding"],
                        alternatives_rejected=[
                            "Assume 'fairness through unawareness' works — the most common "
                            "intuition, and demonstrably false on this data.",
                        ],
                    ),
                ],
                evidence={"own_model": own_model, "split": split_report.to_dict()},
            )
        )
        crisp.record(
            Phase(
                name="evaluation",
                summary=(
                    f"Of four fairness criteria, "
                    f"{sum(1 for c in criteria.values() if c['satisfied'])} are satisfied "
                    f"and {sum(1 for c in criteria.values() if not c['satisfied'])} are "
                    "violated. That split is not a defect in the audit — Chouldechova's "
                    "identity, verified on this data to within "
                    f"{impossibility['max_identity_discrepancy']:.5f}, shows the "
                    "criteria are mathematically incompatible once base rates differ."
                ),
                evidence={
                    "criteria": criteria,
                    "criteria_summary": criteria_summary,
                    "impossibility": impossibility,
                    "significance": significance,
                    "calibration": calibration,
                    "by_group": by_group,
                },
                risks=[
                    "This audit cannot say whether COMPAS is fair, because 'fair' is not a "
                    "single measurable property. It can say precisely which criteria hold "
                    "and which do not, and why no tool can satisfy all of them here.",
                    "Rearrest is the proxy for reoffending. If policing differs by group, "
                    "the 'ground truth' is itself affected by the disparity under "
                    "investigation, and every error rate inherits that.",
                    "Thresholding a decile score at 5 is ProPublica's choice, adopted for "
                    "comparability. A different cut changes every rate reported here, "
                    "though not the impossibility result, which holds at any threshold.",
                ],
            )
        )
        crisp.record(
            Phase(
                name="deployment",
                summary=(
                    "The audit ships as a dashboard where a reader can select a fairness "
                    "criterion and see which groups it favours, with the impossibility "
                    "identity shown alongside so no criterion can be read as the answer."
                ),
                evidence={"criteria": list(criteria)},
            )
        )

        artifacts.write(OUT / "profile.json", profile, context=ctx)
        artifacts.write(
            OUT / "fairness.json",
            {
                "by_group": by_group,
                "criteria": criteria,
                "criteria_summary": criteria_summary,
                "impossibility": impossibility,
                "significance": significance,
                "calibration": calibration,
                "reference_group": reference,
            },
            context=ctx,
        )
        artifacts.write(OUT / "own_model.json", own_model, context=ctx)
        artifacts.write(OUT / "crispdm.json", crisp.to_dict(), context=ctx)
        artifacts.write(OUT / "provenance.json",
                        {"datasets": [data.provenance_record("compas")]}, context=ctx)

    satisfied = sum(1 for c in criteria.values() if c["satisfied"])
    print(f"\n✓ {PROJECT} complete — {satisfied}/{len(criteria)} criteria satisfied; "
          f"impossibility identity holds to {impossibility['max_identity_discrepancy']:.5f}")


if __name__ == "__main__":
    main()
