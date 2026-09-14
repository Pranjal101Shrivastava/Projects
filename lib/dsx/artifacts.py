"""Artifact writing and run provenance.

Every number that appears in a README, paper or dashboard in this repository is written
here by a training script and read back from JSON. Nothing is typed by hand into prose.
That rule is what makes the documentation checkable: if a claim in a paper disagrees with
the artifact, the artifact wins and the paper is wrong.

Each artifact carries a ``_run`` block recording the git commit, library versions, random
seed and wall-clock duration that produced it, so a figure can always be traced back to
the exact code that generated it.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _library_versions() -> dict:
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "sklearn", "lightgbm", "torch"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            continue
    return versions


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that understands NumPy scalars and arrays.

    Without this, a stray ``np.float32`` deep inside a nested result dict raises at write
    time, which is a tedious failure mode when it happens after a long training run.
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            value = float(o)
            return None if (np.isnan(value) or np.isinf(value)) else value
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


@dataclass
class RunContext:
    """Bookkeeping for a single training run."""

    project: str
    seed: int
    started: float

    def block(self) -> dict:
        return {
            "project": self.project,
            "seed": self.seed,
            "git_commit": _git_commit(),
            "libraries": _library_versions(),
            "duration_seconds": round(time.time() - self.started, 2),
            "generated_at_unix": int(time.time()),
        }


@contextmanager
def run(project: str, *, seed: int = 42):
    """Seed every RNG in play and yield a run context.

    Seeding is done here rather than in each script so that no script can forget. The
    audit scanner treats absence of a ``dsx.artifacts.run`` block in a training script as
    a reproducibility finding.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass

    context = RunContext(project=project, seed=seed, started=time.time())
    print(f"[{project}] run started (seed={seed}, commit={_git_commit()})")
    yield context
    print(f"[{project}] run finished in {round(time.time() - context.started, 2)}s")


def write(path: Path | str, payload: dict, *, context: RunContext | None = None) -> Path:
    """Write an artifact as pretty JSON, stamped with run provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    if context is not None:
        body["_run"] = context.block()
    path.write_text(json.dumps(body, indent=2, cls=NumpyEncoder))
    size_kb = path.stat().st_size / 1024
    print(f"  wrote {path.relative_to(ROOT)} ({size_kb:.1f} KB)")
    return path


def read(path: Path | str) -> dict:
    """Read a previously written artifact."""
    return json.loads(Path(path).read_text())


def histogram(values: np.ndarray, *, bins: int = 30) -> dict:
    """Bin a distribution for client-side rendering.

    Frontends receive binned counts rather than raw arrays. A 284,807-row column is 2 MB
    of JSON as raw values and about 1 KB as thirty bins, and no chart can display more
    than the bins anyway.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"bins": [], "counts": [], "n": 0}
    counts, edges = np.histogram(values, bins=bins)
    return {
        "bin_edges": [round(float(e), 6) for e in edges],
        "bin_centers": [round(float((edges[i] + edges[i + 1]) / 2), 6) for i in range(len(counts))],
        "counts": [int(c) for c in counts],
        "n": int(len(values)),
        "mean": round(float(values.mean()), 6),
        "median": round(float(np.median(values)), 6),
        "std": round(float(values.std()), 6),
        "p01": round(float(np.percentile(values, 1)), 6),
        "p25": round(float(np.percentile(values, 25)), 6),
        "p75": round(float(np.percentile(values, 75)), 6),
        "p99": round(float(np.percentile(values, 99)), 6),
        "min": round(float(values.min()), 6),
        "max": round(float(values.max()), 6),
    }


def downsample(x: np.ndarray, y: np.ndarray, *, max_points: int = 600) -> dict:
    """Uniformly thin a scatter series to a size a browser can render smoothly."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) <= max_points:
        idx = np.arange(len(x))
    else:
        idx = np.linspace(0, len(x) - 1, max_points).astype(int)
    return {
        "x": [round(float(v), 6) for v in x[idx]],
        "y": [round(float(v), 6) for v in y[idx]],
        "n_original": int(len(x)),
        "n_plotted": int(len(idx)),
    }
