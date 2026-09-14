"""Regression tests for the leakage audit scanner.

An audit tool is only worth running if it actually detects the things it claims to. These
tests run it against fixtures containing known mistakes and assert that each is caught.

This suite exists because the first version of the scanner reported zero findings across
the whole repository, which looked like a clean bill of health and was in fact a bug: it
matched only `StandardScaler().fit(X)` written inline and missed the far more common
`scaler = StandardScaler()` followed by `scaler.fit(X)`. Testing against deliberately
broken code found that in one run.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import audit as A  # noqa: E402


def _audit_fixture(tmp_path: Path, fixture: str) -> A.ProjectAudit:
    """Run the scanner over one fixture file laid out like a real project."""
    project = tmp_path / "projects" / "fixture"
    (project / "pipeline").mkdir(parents=True)
    shutil.copy(ROOT / "tools" / "tests" / fixture, project / "pipeline" / "build.py")

    original_root, original_projects = A.ROOT, A.PROJECTS
    A.ROOT, A.PROJECTS = tmp_path, tmp_path / "projects"
    try:
        return A.audit_project(project)
    finally:
        A.ROOT, A.PROJECTS = original_root, original_projects


@pytest.fixture
def leaky(tmp_path: Path) -> A.ProjectAudit:
    return _audit_fixture(tmp_path, "leaky_fixture.py")


def test_detects_synthetic_data(leaky: A.ProjectAudit) -> None:
    rules = {f.rule for f in leaky.findings}
    assert "synthetic-data" in rules


def test_detects_preprocessing_leak_via_assigned_variable(leaky: A.ProjectAudit) -> None:
    """`scaler = StandardScaler(); scaler.fit_transform(X)` must be caught.

    This is the regression: matching only the inline spelling missed every realistic case.
    """
    findings = [f for f in leaky.findings if f.rule == "preprocessing-leak"]
    assert findings, "assigned-then-fitted transformer was not detected"
    assert findings[0].severity == "critical"


def test_detects_unshifted_rolling_window(leaky: A.ProjectAudit) -> None:
    assert any(f.rule == "lookahead-window" for f in leaky.findings)


def test_detects_centered_rolling_window(leaky: A.ProjectAudit) -> None:
    centered = [
        f for f in leaky.findings
        if f.rule == "lookahead-window" and f.severity == "critical"
    ]
    assert centered, "center=True window was not flagged as critical"


def test_detects_undeclared_url(leaky: A.ProjectAudit) -> None:
    assert any(f.rule == "undeclared-source" for f in leaky.findings)


def test_flags_missing_run_context(leaky: A.ProjectAudit) -> None:
    assert any(f.rule == "reproducibility" for f in leaky.findings)


def test_bare_suppression_is_rejected(tmp_path: Path) -> None:
    """An `# audit: ok(rule)` with no reason must leave the finding standing."""
    result = _audit_fixture(tmp_path, "bare_suppression_fixture.py")
    leaks = [f for f in result.findings if f.rule == "preprocessing-leak"]
    assert len(leaks) == 1
    assert leaks[0].acknowledged is None, "bare suppression was wrongly accepted"
    assert result.critical == 1


def test_real_repository_has_no_unacknowledged_critical_findings() -> None:
    """The repository's own pipelines must stay clean.

    Acknowledged findings are permitted — they carry a written justification that a reader
    can evaluate — but an unacknowledged critical finding fails the build.
    """
    audits = [
        A.audit_project(d)
        for d in sorted((ROOT / "projects").iterdir())
        if d.is_dir() and (d / "pipeline").is_dir()
    ]
    assert audits, "no projects found"
    offenders = {a.project: a.critical for a in audits if a.critical}
    assert not offenders, f"unacknowledged critical findings: {offenders}"


def test_every_acknowledgement_carries_a_substantive_reason() -> None:
    """Acknowledgements must explain themselves, not merely exist."""
    for directory in sorted((ROOT / "projects").iterdir()):
        if not (directory / "pipeline").is_dir():
            continue
        for finding in A.audit_project(directory).acknowledged:
            assert finding.acknowledged and len(finding.acknowledged) >= 40, (
                f"{directory.name}: acknowledgement for {finding.rule} is too short "
                "to be a real justification"
            )
