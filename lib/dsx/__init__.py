"""dsx - the shared data science toolkit backing every project in this portfolio.

The eight projects here deliberately share one library rather than each carrying its own
copy of the same helpers. That has a specific payoff for a portfolio whose central claim
is methodological rigour: the leakage-safe split, the imbalance-aware metric and the
provenance-stamped artifact writer are implemented *once*, so an auditor checks them once
and then knows the property holds everywhere it is imported.

Modules
-------
``data``
    Dataset registry. Declares every source with its true origin, licence and a REAL vs
    SIMULATED marker, caches downloads under ``.data/`` and pins them by SHA-256.
``splits``
    Train/test splitting that resists preprocessing, temporal and group leakage.
``metrics``
    Scores reported alongside their no-skill baseline, so a figure can be read as skill
    rather than as an impressive-sounding constant.
``artifacts``
    Run seeding, provenance stamping and JSON artifact IO. Every number in the docs comes
    from here.
``crispdm``
    Small structures for recording the six CRISP-DM phases as machine-readable evidence
    rather than as section headings in a document.
"""

from . import artifacts, crispdm, data, metrics, splits

__all__ = ["artifacts", "crispdm", "data", "metrics", "splits"]
__version__ = "1.0.0"
