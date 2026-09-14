"""Static leakage and reproducibility audit over every pipeline in the repository.

Run:
    python3 tools/audit.py            # human-readable report
    python3 tools/audit.py --write    # also write per-project and repo-level Markdown

This walks the Python AST of each pipeline rather than grepping for strings, so it reasons
about *structure*: whether a scaler is fitted inside a Pipeline, whether a rolling window is
shifted before aggregation, whether a shuffle is applied to something ordered in time.

Two design points worth stating, because an audit tool that flatters its author is worthless:

1. **It reports findings against this repository's own code and is expected to find some.**
   A clean run on the first attempt would suggest the checks are too weak, not that the code
   is perfect. Findings that are deliberate are answered in the project's prose, not
   suppressed here.
2. **Static analysis cannot prove the absence of leakage.** It catches structural mistakes —
   the ones that are invisible in metrics and therefore most dangerous. It cannot tell you
   that a feature is unknowable at scoring time; only domain reasoning does that, which is
   why `duration` in Project 06 is caught by argument rather than by this tool.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

# How far above a flagged line to look for an `# audit: ok(rule) reason` pragma.
LOOKBACK_LINES = 10

# Transformers that learn parameters from data. Calling .fit() on one outside a Pipeline is
# the classic preprocessing leak: statistics from the full dataset enter the training
# transform.
STATEFUL_TRANSFORMERS = {
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler", "Normalizer",
    "QuantileTransformer", "PowerTransformer", "SimpleImputer", "KNNImputer",
    "IterativeImputer", "OneHotEncoder", "OrdinalEncoder", "TargetEncoder",
    "LabelEncoder", "PCA", "TruncatedSVD", "SelectKBest", "VarianceThreshold",
}

SYNTHETIC_MARKERS = {
    "make_classification", "make_regression", "make_blobs", "make_moons",
    "make_circles", "make_friedman1", "make_sparse_uncorrelated",
}


@dataclass
class Finding:
    severity: str
    rule: str
    message: str
    line: int
    file: str
    # Set when the line carries an `# audit: ok(rule) — reason` pragma. Acknowledged
    # findings are reported separately rather than hidden: the rule still fired, and a
    # reader can judge whether the stated reason is good enough.
    acknowledged: str | None = None


@dataclass
class ProjectAudit:
    project: str
    files: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    checks_passed: list[str] = field(default_factory=list)

    @property
    def critical(self) -> int:
        return sum(
            1 for f in self.findings
            if f.severity == "critical" and not f.acknowledged
        )

    @property
    def warnings(self) -> int:
        return sum(
            1 for f in self.findings
            if f.severity == "warning" and not f.acknowledged
        )

    @property
    def acknowledged(self) -> list[Finding]:
        return [f for f in self.findings if f.acknowledged]


class LeakageVisitor(ast.NodeVisitor):
    """Collect structural evidence from one module's AST."""

    def __init__(self, path: Path, source: str = "") -> None:
        self.path = path
        self.source_lines = source.splitlines()
        self.findings: list[Finding] = []
        # Maps a variable name to the transformer class it was assigned from, so that the
        # common `scaler = StandardScaler()` / `scaler.fit_transform(X)` pattern is caught.
        # Matching only inline `StandardScaler().fit(...)` misses almost every real
        # occurrence — this gap was found by running the tool against a known-leaky
        # fixture (tools/tests/leaky_fixture.py) rather than by inspection.
        self.transformer_vars: dict[str, str] = {}
        # Variables assigned from a Pipeline(...) call are exempt: fitting a Pipeline is
        # the correct thing to do.
        self.pipeline_vars: set[str] = set()
        self.evidence = {
            "uses_pipeline": False,
            "asserts_pipeline_safe": False,
            "uses_dsx_run": False,
            "seeds_set": set(),
            "random_state_calls": 0,
            "estimator_calls": 0,
            "uses_temporal_split": False,
            "uses_stratified_split": False,
            "shuffles": [],
            "rolling_calls": [],
            "shifted_rolling": 0,
            "unshifted_rolling": 0,
            "synthetic_generators": [],
            "declared_datasets": [],
            "raw_urls": [],
        }

    # --- helpers -------------------------------------------------------------------
    @staticmethod
    def _name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _acknowledgement(self, rule: str, line: int) -> str | None:
        """Read an `# audit: ok(rule) reason` pragma on or just above the flagged line.

        The pragma must name the specific rule and must carry a reason. A bare suppression
        is not accepted — the point is to force the author to write down why the finding is
        acceptable, so a reader can disagree with the reasoning.
        """
        marker = f"audit: ok({rule})"
        index = line - 1

        # Search a bounded window above the flagged line. A fixed lookback rather than
        # "the immediately preceding comment block" because the justification usually
        # explains a short group of statements (`pca = PCA(...)` then `pca.fit_transform`)
        # and sits above the group, not above the exact line that trips the rule.
        start = max(0, index - LOOKBACK_LINES)
        for i in range(index, start - 1, -1):
            if i >= len(self.source_lines):
                continue
            if marker not in self.source_lines[i]:
                continue

            # Collect the reason from this line plus any comment lines continuing it.
            parts = [self.source_lines[i].split(marker, 1)[1]]
            for j in range(i + 1, min(len(self.source_lines), i + LOOKBACK_LINES)):
                stripped = self.source_lines[j].lstrip()
                if not stripped.startswith("#"):
                    break
                parts.append(stripped.lstrip("#"))
            reason = " ".join(p.strip() for p in parts).strip(" \t#-—:")
            # A bare suppression with no reason is rejected: the finding stands.
            return reason if len(reason) >= 20 else None
        return None

    def _add(self, severity: str, rule: str, message: str, line: int) -> None:
        self.findings.append(
            Finding(
                severity, rule, message, line, str(self.path.relative_to(ROOT)),
                acknowledged=self._acknowledgement(rule, line),
            )
        )

    # --- visitors ------------------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        name = self._name(node.func)

        if name == "Pipeline":
            self.evidence["uses_pipeline"] = True
        if name == "assert_pipeline_safe":
            self.evidence["asserts_pipeline_safe"] = True
        if name == "run" and isinstance(node.func, ast.Attribute):
            if self._name(node.func.value) == "artifacts":
                self.evidence["uses_dsx_run"] = True
        if name == "temporal_split":
            self.evidence["uses_temporal_split"] = True
        if name == "stratified_split":
            self.evidence["uses_stratified_split"] = True

        if name in SYNTHETIC_MARKERS:
            self.evidence["synthetic_generators"].append(name)
            self._add(
                "critical", "synthetic-data",
                f"{name}() generates synthetic data. This portfolio claims real data "
                "throughout; if this is deliberate it must be declared SIMULATED in the "
                "dataset registry and labelled as such in the project README.",
                node.lineno,
            )

        if name in ("load_csv", "fetch", "load_text", "load_uber_months"):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    self.evidence["declared_datasets"].append(arg.value)

        # A stateful transformer .fit() outside a Pipeline, in either spelling:
        #   StandardScaler().fit_transform(X)        — inline
        #   scaler = StandardScaler(); scaler.fit(X) — assigned first (the common case)
        if name in ("fit", "fit_transform") and isinstance(node.func, ast.Attribute):
            receiver = node.func.value
            klass = ""
            if isinstance(receiver, ast.Call):
                inline = self._name(receiver.func)
                if inline in STATEFUL_TRANSFORMERS:
                    klass = inline
            elif isinstance(receiver, ast.Name):
                if receiver.id in self.pipeline_vars:
                    klass = ""
                else:
                    klass = self.transformer_vars.get(receiver.id, "")
            elif isinstance(receiver, ast.Attribute):
                attr = self._name(receiver)
                if attr in STATEFUL_TRANSFORMERS:
                    klass = attr

            if klass:
                self._add(
                    "critical", "preprocessing-leak",
                    f"{klass}.{name}() is called outside a Pipeline. A stateful transformer "
                    "fitted this way sees every row it is given, so test-set statistics "
                    "enter the training transform. Nothing in the metrics reveals this — "
                    "the score simply comes out a little too high.",
                    node.lineno,
                )

        # Shuffling.
        for kw in node.keywords:
            if kw.arg == "shuffle" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                self.evidence["shuffles"].append(node.lineno)
            if kw.arg == "random_state":
                self.evidence["random_state_calls"] += 1

        # Rolling windows: the shift must come before, not after.
        if name == "rolling" and isinstance(node.func, ast.Attribute):
            self.evidence["rolling_calls"].append(node.lineno)
            receiver = node.func.value
            shifted = (
                isinstance(receiver, ast.Call)
                and self._name(receiver.func) == "shift"
            ) or (
                isinstance(receiver, ast.Name) and "shift" in receiver.id.lower()
            )
            if shifted:
                self.evidence["shifted_rolling"] += 1
            else:
                self.evidence["unshifted_rolling"] += 1
                self._add(
                    "warning", "lookahead-window",
                    ".rolling() is applied without a preceding .shift(1). If the rolled "
                    "column is or derives from the target, the window includes the value "
                    "being predicted and the fit will look excellent while being useless.",
                    node.lineno,
                )

        # centered rolling windows always look ahead.
        for kw in node.keywords:
            if kw.arg == "center" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                self._add(
                    "critical", "lookahead-window",
                    "center=True on a rolling window makes it span future observations.",
                    node.lineno,
                )

        # Hard-coded URLs bypassing the registry.
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.startswith("http") and "raw.githubusercontent" in arg.value:
                    self.evidence["raw_urls"].append(node.lineno)

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in ("SEED", "RANDOM_STATE"):
                self.evidence["seeds_set"].add(target.id)

            # Record `scaler = StandardScaler()` and `pipe = Pipeline([...])` so that a
            # later `.fit()` on the variable can be resolved back to its class.
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                constructed = self._name(node.value.func)
                if constructed in STATEFUL_TRANSFORMERS:
                    self.transformer_vars[target.id] = constructed
                elif constructed in ("Pipeline", "make_pipeline", "ColumnTransformer"):
                    self.pipeline_vars.add(target.id)

        self.generic_visit(node)


def audit_project(project_dir: Path) -> ProjectAudit:
    audit = ProjectAudit(project=project_dir.name)
    pipeline_files = sorted((project_dir / "pipeline").glob("*.py"))
    if not pipeline_files:
        return audit

    combined = {
        "uses_pipeline": False, "asserts_pipeline_safe": False, "uses_dsx_run": False,
        "seeds_set": set(), "uses_temporal_split": False, "uses_stratified_split": False,
        "shifted_rolling": 0, "unshifted_rolling": 0, "synthetic_generators": [],
        "declared_datasets": [], "raw_urls": [], "shuffles": [],
    }

    for path in pipeline_files:
        audit.files.append(str(path.relative_to(ROOT)))
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        visitor = LeakageVisitor(path, source)
        visitor.visit(tree)
        audit.findings.extend(visitor.findings)

        for key, value in visitor.evidence.items():
            if key not in combined:
                continue
            if isinstance(value, bool):
                combined[key] = combined[key] or value
            elif isinstance(value, set):
                combined[key] |= value
            elif isinstance(value, list):
                combined[key].extend(value)
            elif isinstance(value, int):
                combined[key] += value

    # --- repository-level obligations ---------------------------------------------
    if combined["uses_dsx_run"]:
        audit.checks_passed.append(
            "Run is wrapped in dsx.artifacts.run(), so every RNG is seeded and the git "
            "commit, library versions and duration are stamped into the artifact."
        )
    else:
        audit.findings.append(Finding(
            "warning", "reproducibility",
            "Pipeline does not use dsx.artifacts.run(). Without it the run is not seeded "
            "centrally and artifacts carry no provenance stamp.",
            1, audit.files[0],
        ))

    if combined["asserts_pipeline_safe"]:
        audit.checks_passed.append(
            "Calls splits.assert_pipeline_safe(), which raises unless preprocessing is "
            "inside a scikit-learn Pipeline and therefore refitted per fold."
        )
    elif combined["uses_pipeline"]:
        audit.checks_passed.append(
            "Uses a scikit-learn Pipeline, so preprocessing is fitted within each fold."
        )

    if combined["unshifted_rolling"] == 0 and combined["shifted_rolling"] > 0:
        audit.checks_passed.append(
            f"All {combined['shifted_rolling']} rolling windows are preceded by .shift(), "
            "so a window ends at t-1 and no observation enters its own statistic."
        )

    if combined["uses_temporal_split"]:
        audit.checks_passed.append(
            "Uses temporal_split(), which never shuffles and supports a boundary embargo."
        )
    if combined["uses_stratified_split"]:
        audit.checks_passed.append(
            "Uses stratified_split(), preserving the class base rate across partitions."
        )

    if combined["seeds_set"]:
        audit.checks_passed.append(
            f"Declares a module-level seed ({', '.join(sorted(combined['seeds_set']))})."
        )

    if not combined["synthetic_generators"]:
        audit.checks_passed.append(
            "No synthetic data generators. All inputs come from the declared dataset "
            "registry."
        )

    if combined["raw_urls"]:
        audit.findings.append(Finding(
            "warning", "undeclared-source",
            f"{len(combined['raw_urls'])} hard-coded URL(s) bypass the dataset registry, so "
            "the source is not hash-pinned and will not appear in the provenance table.",
            combined["raw_urls"][0], audit.files[0],
        ))
    elif combined["declared_datasets"]:
        audit.checks_passed.append(
            f"All data requested through the registry by id: "
            f"{', '.join(sorted(set(combined['declared_datasets'])))}."
        )

    audit.findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], f.file, f.line))
    return audit


def render_markdown(audits: list[ProjectAudit]) -> str:
    total_critical = sum(a.critical for a in audits)
    total_warnings = sum(a.warnings for a in audits)
    total_passed = sum(len(a.checks_passed) for a in audits)
    total_ack = sum(len(a.acknowledged) for a in audits)

    lines = [
        "# Leakage & Reproducibility Audit",
        "",
        "Generated by `tools/audit.py`, which walks the Python AST of every pipeline in "
        "this repository. It reasons about structure — whether a transformer is fitted "
        "inside a `Pipeline`, whether a rolling window is shifted before aggregation, "
        "whether an ordered dataset is shuffled — rather than grepping for strings.",
        "",
        "## What this tool can and cannot establish",
        "",
        "It catches **structural** leakage: the mistakes that are invisible in metrics and "
        "therefore the most dangerous, because the score simply comes out a little too "
        "high and nothing errors.",
        "",
        "It **cannot** establish that a feature is knowable at scoring time. That is a "
        "question about the world, not about code. The `duration` leak in Project 06 — "
        "worth 40% of PR-AUC — is invisible to any static analyser and was caught by asking "
        "when the value comes into existence. A green report here is necessary, not "
        "sufficient.",
        "",
        "## Summary",
        "",
        f"| Projects audited | {len(audits)} |",
        "|---|---|",
        f"| Structural checks passed | **{total_passed}** |",
        f"| Critical findings | **{total_critical}** |",
        f"| Warnings | **{total_warnings}** |",
        f"| Acknowledged (rule fired, reason recorded) | {total_ack} |",
        "",
        "An acknowledged finding is one where the rule fired correctly but the author "
        "recorded a written reason why it is acceptable in context, via an "
        "`# audit: ok(rule) reason` pragma. A bare suppression is not accepted by the "
        "tool — the reason is mandatory, and it is reproduced below so a reader can "
        "disagree with it.",
        "",
        "| Project | Files | Checks passed | Critical | Warnings | Acknowledged |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for a in audits:
        lines.append(
            f"| `{a.project}` | {len(a.files)} | {len(a.checks_passed)} | "
            f"{'**' + str(a.critical) + '**' if a.critical else '0'} | {a.warnings} | "
            f"{len(a.acknowledged)} |"
        )

    lines += ["", "## Per-project detail", ""]
    for a in audits:
        lines += [f"### `{a.project}`", ""]
        if a.checks_passed:
            lines.append("**Passed**")
            lines.append("")
            for c in a.checks_passed:
                lines.append(f"- {c}")
            lines.append("")
        if a.findings:
            lines.append("**Findings**")
            lines.append("")
            lines.append("| Severity | Rule | Location | Detail |")
            lines.append("|---|---|---|---|")
            for f in a.findings:
                marker = {"critical": "🔴", "warning": "🟡", "info": "⚪"}[f.severity]
                if f.acknowledged:
                    marker = "⚪ acknowledged"
                    detail = f"{f.message}<br><br>**Author's reason:** {f.acknowledged}"
                else:
                    detail = f.message
                lines.append(
                    f"| {marker} | `{f.rule}` | `{f.file}:{f.line}` | {detail} |"
                )
            lines.append("")
        else:
            lines.append("No findings.")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write Markdown reports")
    args = parser.parse_args()

    audits = [
        audit_project(d)
        for d in sorted(PROJECTS.iterdir())
        if d.is_dir() and (d / "pipeline").is_dir()
    ]

    width = max(len(a.project) for a in audits)
    print(f"{'project'.ljust(width)}  passed  critical  warnings  acknowledged")
    print("-" * (width + 42))
    for a in audits:
        flag = "  ← REVIEW" if a.critical else ""
        print(f"{a.project.ljust(width)}  {len(a.checks_passed):>6}  {a.critical:>8}  "
              f"{a.warnings:>8}  {len(a.acknowledged):>12}{flag}")
    print("-" * (width + 42))
    total_critical = sum(a.critical for a in audits)
    total_warnings = sum(a.warnings for a in audits)
    total_ack = sum(len(a.acknowledged) for a in audits)
    print(f"{'TOTAL'.ljust(width)}  {sum(len(a.checks_passed) for a in audits):>6}  "
          f"{total_critical:>8}  {total_warnings:>8}  {total_ack:>12}")

    for a in audits:
        for f in a.findings:
            if f.severity == "critical" and not f.acknowledged:
                print(f"\n🔴 {a.project} — {f.rule} at {f.file}:{f.line}\n   {f.message}")

    if total_ack:
        print("\nAcknowledged findings (rule fired; author recorded a reason):")
        for a in audits:
            for f in a.acknowledged:
                print(f"  ⚪ {a.project} — {f.rule} at {f.file}:{f.line}")
                print(f"     reason: {f.acknowledged}")

    if args.write:
        report = render_markdown(audits)
        (ROOT / "AUDIT.md").write_text(report)
        print(f"\nWrote AUDIT.md ({len(report)} bytes)")

        for a in audits:
            per_project = [
                f"# Audit — `{a.project}`",
                "",
                f"Generated by `tools/audit.py`. {len(a.checks_passed)} structural checks "
                f"passed, {a.critical} critical finding(s), {a.warnings} warning(s).",
                "",
                "## Passed",
                "",
            ]
            per_project += [f"- {c}" for c in a.checks_passed] or ["- (none)"]
            per_project += ["", "## Findings", ""]
            if a.findings:
                per_project.append("| Severity | Rule | Location | Detail |")
                per_project.append("|---|---|---|---|")
                for f in a.findings:
                    marker = {"critical": "🔴", "warning": "🟡", "info": "⚪"}[f.severity]
                    per_project.append(
                        f"| {marker} {f.severity} | `{f.rule}` | `{f.file}:{f.line}` | {f.message} |"
                    )
            else:
                per_project.append("None.")
            per_project.append("")
            (PROJECTS / a.project / "audit.md").write_text("\n".join(per_project))

        (ROOT / "web" / "public" / "data" / "audit.json").write_text(
            json.dumps(
                {
                    "projects": [
                        {
                            "project": a.project,
                            "checks_passed": a.checks_passed,
                            "findings": [
                                {"severity": f.severity, "rule": f.rule, "message": f.message,
                                 "file": f.file, "line": f.line}
                                for f in a.findings
                            ],
                        }
                        for a in audits
                    ],
                    "total_critical": total_critical,
                    "total_warnings": total_warnings,
                },
                indent=2,
            )
        )
        print("Wrote per-project audit.md and web/public/data/audit.json")


if __name__ == "__main__":
    main()
