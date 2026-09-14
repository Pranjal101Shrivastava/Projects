"""Leakage-safe data splitting.

Every split in this portfolio goes through one of these helpers. They exist because the
three most common ways to accidentally inflate a score are all splitting mistakes:

1. **Preprocessing leakage** - fitting a scaler, imputer or encoder on the full dataset
   before splitting, so test-set statistics bleed into the training transform. The fix is
   not a helper function but a discipline: fit inside a ``Pipeline``, never outside. See
   :func:`assert_pipeline_safe`.
2. **Temporal leakage** - shuffling a time series, so the model trains on the future and
   is scored on the past. :func:`temporal_split` refuses to shuffle.
3. **Group leakage** - the same entity appearing in both train and test, so the model
   memorises the entity rather than learning the pattern. :func:`grouped_split` keeps
   groups whole.

Each function returns index arrays rather than sliced frames, which keeps the caller
honest about what it is doing and makes the split auditable after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, train_test_split


@dataclass(frozen=True)
class SplitReport:
    """Description of a split, emitted into artifacts so reviewers can check it."""

    strategy: str
    n_train: int
    n_test: int
    rationale: str
    shuffled: bool
    detail: dict

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "train_fraction": round(self.n_train / max(1, self.n_train + self.n_test), 4),
            "shuffled": self.shuffled,
            "rationale": self.rationale,
            **self.detail,
        }


def stratified_split(
    y: np.ndarray | pd.Series,
    *,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, SplitReport]:
    """Stratified train/test split for i.i.d. classification data.

    Stratification matters most exactly where it is most often skipped: a rare positive
    class. With 492 positives in 284,807 rows, an unstratified split can hand a fold a
    materially different base rate and make fold-to-fold variance look like signal.
    """
    y = np.asarray(y)
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=y, shuffle=True
    )
    positives_train = int(np.sum(y[train_idx] == 1))
    positives_test = int(np.sum(y[test_idx] == 1))
    report = SplitReport(
        strategy="stratified_holdout",
        n_train=len(train_idx),
        n_test=len(test_idx),
        shuffled=True,
        rationale=(
            "Observations are exchangeable, so shuffling is valid. Stratifying on the "
            "target preserves the class base rate in both partitions."
        ),
        detail={
            "seed": seed,
            "positives_train": positives_train,
            "positives_test": positives_test,
            "base_rate_train": round(positives_train / max(1, len(train_idx)), 6),
            "base_rate_test": round(positives_test / max(1, len(test_idx)), 6),
        },
    )
    return train_idx, test_idx, report


def temporal_split(
    n: int,
    *,
    test_size: float = 0.2,
    embargo: int = 0,
) -> tuple[np.ndarray, np.ndarray, SplitReport]:
    """Chronological split. Never shuffles.

    The training partition is strictly the earliest ``1 - test_size`` of the sequence and
    the test partition is strictly the latest. An ``embargo`` of *k* observations drops
    the *k* rows straddling the boundary, which matters whenever features use rolling
    windows: without it, a window computed at the start of the test period has already
    seen training rows, and the two partitions are no longer independent.

    Parameters
    ----------
    n:
        Number of observations, assumed already sorted oldest-first.
    test_size:
        Fraction of the tail held out.
    embargo:
        Observations discarded at the boundary. Set this to the widest rolling window
        used in feature engineering.
    """
    if n < 2:
        raise ValueError("temporal_split needs at least two observations.")

    cut = int(n * (1 - test_size))
    train_idx = np.arange(0, max(1, cut - embargo))
    test_idx = np.arange(cut, n)

    report = SplitReport(
        strategy="chronological_holdout",
        n_train=len(train_idx),
        n_test=len(test_idx),
        shuffled=False,
        rationale=(
            "Observations are ordered in time, so any shuffle would train on the future "
            "and score on the past. The split is a single cut with no reordering."
            + (
                f" An embargo of {embargo} observations is dropped at the boundary so "
                "rolling-window features in the test period cannot span training rows."
                if embargo
                else ""
            )
        ),
        detail={
            "embargo": embargo,
            "train_range": [0, int(max(1, cut - embargo)) - 1],
            "test_range": [int(cut), n - 1],
        },
    )
    return train_idx, test_idx, report


def grouped_split(
    groups: np.ndarray | pd.Series,
    *,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, SplitReport]:
    """Split so that no group appears on both sides.

    Used where rows repeat an entity - a customer with many invoices, a location with
    many trips. Splitting rows at random would let the model memorise the entity and
    report a score it cannot reproduce on an entity it has never seen.
    """
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_test_groups = max(1, int(len(unique) * test_size))
    test_groups = set(shuffled[:n_test_groups].tolist())

    mask = np.array([g in test_groups for g in groups])
    test_idx = np.flatnonzero(mask)
    train_idx = np.flatnonzero(~mask)

    report = SplitReport(
        strategy="grouped_holdout",
        n_train=len(train_idx),
        n_test=len(test_idx),
        shuffled=True,
        rationale=(
            "Rows repeat entities, so the split is taken over groups rather than rows. "
            "No group contributes rows to both partitions, which prevents the model from "
            "scoring well by memorising entities it will not meet at inference time."
        ),
        detail={
            "seed": seed,
            "n_groups_total": int(len(unique)),
            "n_groups_test": int(n_test_groups),
        },
    )
    return train_idx, test_idx, report


def purged_walk_forward(
    n: int,
    *,
    n_splits: int = 5,
    embargo: int = 0,
):
    """Yield expanding-window train/validation folds for sequential data.

    Wraps scikit-learn's :class:`TimeSeriesSplit` and additionally purges ``embargo``
    observations from the tail of each training fold. Standard ``TimeSeriesSplit`` leaves
    the training fold touching the validation fold, which leaks whenever features look
    backwards over a window.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in splitter.split(np.arange(n)):
        if embargo:
            train_idx = train_idx[: max(1, len(train_idx) - embargo)]
        yield train_idx, val_idx


def stratified_folds(y: np.ndarray | pd.Series, *, n_splits: int = 5, seed: int = 42):
    """Yield stratified CV folds, preserving the class base rate in each."""
    y = np.asarray(y)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from splitter.split(np.zeros(len(y)), y)


def assert_pipeline_safe(estimator) -> None:
    """Fail loudly if a model is not wrapped in a Pipeline.

    Preprocessing leakage is invisible in the metrics: the score simply comes out a
    little too high and nothing errors. The only structural defence is to require that
    every transform lives inside a ``Pipeline``, so ``fit`` on a training fold cannot
    touch validation rows. This helper is called at the top of each training script and
    the audit scanner checks that the call is present.
    """
    from sklearn.pipeline import Pipeline

    if not isinstance(estimator, Pipeline):
        raise TypeError(
            "Estimator must be a sklearn Pipeline so that preprocessing is fitted "
            "strictly inside each training fold. Fitting a scaler, imputer or encoder "
            f"outside the pipeline leaks test statistics into training. Got "
            f"{type(estimator).__name__}."
        )
