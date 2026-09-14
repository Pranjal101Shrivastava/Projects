"""CRISP-DM phase recording.

CRISP-DM is usually applied as a set of headings: a document gets six sections and the
methodology is declared satisfied. That is not verifiable. Here each phase is recorded as
a structured object carrying the decisions taken, the evidence produced and - most
usefully - the *rejected alternatives*, then serialised into the project artifact.

The rejected-alternatives field is the one that does real work. A modelling write-up that
lists only what was chosen is unfalsifiable; one that says "we did not use accuracy
because prevalence is 0.0017" can be argued with, and is therefore worth reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

PhaseName = Literal[
    "business_understanding",
    "data_understanding",
    "data_preparation",
    "modeling",
    "evaluation",
    "deployment",
]

PHASE_ORDER: tuple[PhaseName, ...] = (
    "business_understanding",
    "data_understanding",
    "data_preparation",
    "modeling",
    "evaluation",
    "deployment",
)

PHASE_TITLES: dict[str, str] = {
    "business_understanding": "Business Understanding",
    "data_understanding": "Data Understanding",
    "data_preparation": "Data Preparation",
    "modeling": "Modeling",
    "evaluation": "Evaluation",
    "deployment": "Deployment",
}


@dataclass
class Decision:
    """One consequential choice, with the reasoning that produced it."""

    question: str
    choice: str
    rationale: str
    alternatives_rejected: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "choice": self.choice,
            "rationale": self.rationale,
            "alternatives_rejected": self.alternatives_rejected,
        }


@dataclass
class Phase:
    """A single CRISP-DM phase."""

    name: PhaseName
    summary: str
    decisions: list[Decision] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    risks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "title": PHASE_TITLES[self.name],
            "summary": self.summary,
            "decisions": [d.to_dict() for d in self.decisions],
            "evidence": self.evidence,
            "risks": self.risks,
        }


class CrispDm:
    """Collects phases for one project and serialises them.

    Phases may be recorded out of order - real projects loop back from evaluation to data
    preparation, and pretending otherwise would be dishonest - but they always serialise
    in canonical order for presentation.
    """

    def __init__(self, project: str, business_question: str) -> None:
        self.project = project
        self.business_question = business_question
        self._phases: dict[str, Phase] = {}

    def record(self, phase: Phase) -> Phase:
        self._phases[phase.name] = phase
        return phase

    def to_dict(self) -> dict:
        missing = [p for p in PHASE_ORDER if p not in self._phases]
        return {
            "project": self.project,
            "business_question": self.business_question,
            "phases": [
                self._phases[name].to_dict() for name in PHASE_ORDER if name in self._phases
            ],
            "phases_recorded": len(self._phases),
            "phases_missing": missing,
            "complete": not missing,
        }
