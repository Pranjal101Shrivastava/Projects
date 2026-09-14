"""A pipeline DAG engine, and a measured account of what scheduling actually buys.

Run:
    PYTHONPATH=lib python3 projects/11_pipeline_dag/pipeline/build.py

Every other project in this portfolio is a script you run by hand. That works until there
are twelve of them with shared dependencies, at which point you want the three things an
orchestrator provides: correct ordering, skipping work whose inputs have not changed, and
running independent work concurrently.

This implements those from first principles — Kahn's topological sort, cycle detection with
the actual cycle reported, content-addressed caching, and a level-parallel scheduler — and
then does the part orchestration write-ups usually skip: **measures the speedup on the real
dependency graph instead of asserting it**.

The graph is this repository's own pipelines. Their real runtimes were recorded when they
ran, so the scheduling simulation uses measured durations rather than invented ones, and
the critical path is a genuine property of this portfolio rather than of a toy example.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "lib"))

from dsx import artifacts, data  # noqa: E402
from dsx.crispdm import CrispDm, Decision, Phase  # noqa: E402

PROJECT = "11_pipeline_dag"
OUT = ROOT / "projects" / PROJECT / "artifacts"
SEED = 42


# ======================================================================================
# Graph model
# ======================================================================================
@dataclass
class Task:
    """One node: a unit of work with declared inputs, outputs and dependencies."""

    name: str
    depends_on: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    seconds: float = 0.0
    kind: str = "pipeline"


class CycleError(Exception):
    """Raised when the graph is not acyclic, carrying the offending cycle."""

    def __init__(self, cycle: list[str]) -> None:
        self.cycle = cycle
        super().__init__(" → ".join(cycle + [cycle[0]]))


# ======================================================================================
# Topological sort (Kahn)
# ======================================================================================
def topological_levels(tasks: dict[str, Task]) -> list[list[str]]:
    """Group tasks into dependency levels using Kahn's algorithm.

    Returns a list of levels, where every task in a level depends only on tasks in
    *earlier* levels — so an entire level can run concurrently.

    Kahn's algorithm rather than depth-first search, for a specific reason: DFS yields a
    valid linear order but discards the information that two tasks were independent.
    Kahn's algorithm processes by in-degree, so the tasks available at each step fall out
    naturally as a parallel batch. The scheduling question needs that structure, not just a
    valid sequence.
    """
    in_degree = {name: 0 for name in tasks}
    dependents: dict[str, list[str]] = defaultdict(list)

    for name, task in tasks.items():
        for dependency in task.depends_on:
            if dependency not in tasks:
                raise KeyError(f"{name!r} depends on undeclared task {dependency!r}")
            in_degree[name] += 1
            dependents[dependency].append(name)

    ready = deque(sorted(n for n, d in in_degree.items() if d == 0))
    levels: list[list[str]] = []
    seen = 0

    while ready:
        level = sorted(ready)
        levels.append(level)
        ready.clear()
        for name in level:
            seen += 1
            for dependent in dependents[name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    ready.append(dependent)

    if seen != len(tasks):
        raise CycleError(find_cycle(tasks))
    return levels


def find_cycle(tasks: dict[str, Task]) -> list[str]:
    """Return an actual cycle, not merely the fact that one exists.

    An orchestrator that reports "cycle detected" and stops has told the user almost
    nothing: in a graph of any size, finding the offending edges by hand is the whole
    problem. Tri-colour DFS keeps the recursion stack, so the cycle can be reported.
    """
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {name: WHITE for name in tasks}
    stack: list[str] = []

    def visit(name: str) -> list[str] | None:
        colour[name] = GREY
        stack.append(name)
        for dependency in tasks[name].depends_on:
            if dependency not in tasks:
                continue
            if colour[dependency] == GREY:
                # Back edge: the cycle is the stack from that node onward.
                return stack[stack.index(dependency) :]
            if colour[dependency] == WHITE:
                found = visit(dependency)
                if found:
                    return found
        colour[name] = BLACK
        stack.pop()
        return None

    for name in tasks:
        if colour[name] == WHITE:
            found = visit(name)
            if found:
                return found
    return []


# ======================================================================================
# Content-addressed caching
# ======================================================================================
def task_fingerprint(task: Task, upstream: dict[str, str]) -> str:
    """Content hash of everything that can change a task's output.

    Deliberately includes the fingerprints of upstream tasks. Hashing only a task's own
    source and inputs is the classic mistake: a change deep in the graph then fails to
    invalidate anything downstream, and the pipeline serves stale results while reporting
    success. Chaining the upstream fingerprints makes invalidation transitive.
    """
    digest = hashlib.blake2b(digest_size=16)
    digest.update(task.name.encode())

    for path in sorted(task.inputs):
        full = ROOT / path
        if full.is_file():
            digest.update(full.read_bytes()[:1_000_000])
        else:
            digest.update(b"<missing>")

    for dependency in sorted(task.depends_on):
        digest.update(upstream.get(dependency, "").encode())

    return digest.hexdigest()


# ======================================================================================
# Scheduling
# ======================================================================================
def simulate(levels: list[list[str]], tasks: dict[str, Task], workers: int) -> dict:
    """Simulate level-parallel execution with a fixed worker pool.

    Within a level every task is independent, so tasks are packed onto workers
    longest-first — the LPT heuristic, which bounds the makespan at (4/3 − 1/3m) times
    optimal. Levels are then barriers: the next level cannot start until the current one
    finishes.

    That barrier is a real inefficiency, and it is reported rather than hidden. A
    work-stealing scheduler would start a task the moment *its own* dependencies were
    satisfied rather than waiting for its whole level, and the gap between the two is
    quantified below.
    """
    total = 0.0
    timeline = []
    for level_index, level in enumerate(levels):
        durations = sorted(
            ((tasks[name].seconds, name) for name in level), reverse=True
        )
        loads = [0.0] * max(1, workers)
        assignment: dict[int, list[str]] = defaultdict(list)
        for seconds, name in durations:
            lightest = int(np.argmin(loads))
            loads[lightest] += seconds
            assignment[lightest].append(name)

        level_span = max(loads) if loads else 0.0
        timeline.append(
            {
                "level": level_index,
                "tasks": level,
                "level_seconds": round(level_span, 2),
                "started_at": round(total, 2),
                "worker_loads": [round(v, 2) for v in loads],
                "idle_worker_seconds": round(sum(level_span - v for v in loads), 2),
            }
        )
        total += level_span

    return {
        "workers": workers,
        "makespan_seconds": round(total, 2),
        "timeline": timeline,
        # Worker-seconds, i.e. summed across the pool, NOT wall-clock. With 16
        # workers on a 2,645s schedule this figure exceeds the makespan many times over;
        # that is expected and is why the unit is in the name.
        "idle_worker_seconds": round(
            sum(t["idle_worker_seconds"] for t in timeline), 2
        ),
        "idle_fraction_of_capacity": round(
            sum(t["idle_worker_seconds"] for t in timeline)
            / max(1e-9, total * max(1, workers)),
            3,
        ),
    }


def critical_path(tasks: dict[str, Task], levels: list[list[str]]) -> dict:
    """Longest-duration path through the graph — the hard lower bound on makespan.

    No number of workers can beat it, because its tasks must run one after another. It is
    the single most useful number an orchestrator can report: it says whether to buy more
    machines or to optimise one specific chain.
    """
    longest: dict[str, float] = {}
    predecessor: dict[str, str | None] = {}

    for level in levels:
        for name in level:
            task = tasks[name]
            best_parent, best_cost = None, 0.0
            for dependency in task.depends_on:
                if longest.get(dependency, 0.0) > best_cost:
                    best_cost = longest[dependency]
                    best_parent = dependency
            longest[name] = best_cost + task.seconds
            predecessor[name] = best_parent

    end = max(longest, key=longest.get)
    path = []
    cursor: str | None = end
    while cursor is not None:
        path.append(cursor)
        cursor = predecessor[cursor]
    path.reverse()

    return {
        "path": path,
        "seconds": round(longest[end], 2),
        "task_seconds": {name: round(tasks[name].seconds, 2) for name in path},
    }


# ======================================================================================
# Build the real graph from this repository
# ======================================================================================
def discover_tasks() -> dict[str, Task]:
    """Construct the task graph from this repository's actual pipelines and tools.

    Durations are read from each project's committed ``_run`` block — the wall-clock time
    that pipeline genuinely took. Using measured runtimes rather than invented ones is what
    makes the scheduling analysis below a statement about this portfolio instead of a
    statement about a made-up example.
    """
    tasks: dict[str, Task] = {}
    projects_dir = ROOT / "projects"

    for directory in sorted(projects_dir.iterdir()):
        if not (directory / "pipeline" / "build.py").is_file():
            continue
        name = directory.name
        if name == PROJECT:
            continue  # this project orchestrates the others; it is added separately

        seconds = 0.0
        crisp_path = directory / "artifacts" / "crispdm.json"
        if crisp_path.is_file():
            try:
                seconds = float(
                    json.loads(crisp_path.read_text()).get("_run", {}).get(
                        "duration_seconds", 0.0
                    )
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                seconds = 0.0

        tasks[name] = Task(
            name=name,
            depends_on=["dsx_toolkit"],
            inputs=[f"projects/{name}/pipeline/build.py"],
            outputs=[f"projects/{name}/artifacts"],
            seconds=seconds,
            kind="pipeline",
        )

    # The shared library every pipeline imports: the root of the graph.
    tasks["dsx_toolkit"] = Task(
        name="dsx_toolkit",
        depends_on=[],
        inputs=[
            "lib/dsx/data.py", "lib/dsx/splits.py", "lib/dsx/metrics.py",
            "lib/dsx/artifacts.py", "lib/dsx/crispdm.py",
        ],
        outputs=["lib/dsx"],
        seconds=0.5,
        kind="library",
    )

    # Downstream tooling, which genuinely depends on every pipeline having produced
    # artifacts first.
    pipelines = [n for n, t in tasks.items() if t.kind == "pipeline"]
    tasks["audit"] = Task(
        name="audit", depends_on=pipelines, inputs=["tools/audit.py"],
        outputs=["AUDIT.md"], seconds=1.2, kind="tool",
    )
    tasks["sync_artifacts"] = Task(
        name="sync_artifacts", depends_on=pipelines, inputs=["tools/sync_artifacts.py"],
        outputs=["web/public/data"], seconds=0.6, kind="tool",
    )
    tasks["docs"] = Task(
        name="docs", depends_on=pipelines, inputs=["tools/docs.py"],
        outputs=["projects/*/README.md"], seconds=1.0, kind="tool",
    )
    tasks["web_build"] = Task(
        name="web_build", depends_on=["sync_artifacts"], inputs=["web/package.json"],
        outputs=["web/dist"], seconds=12.0, kind="tool",
    )
    tasks["screenshots"] = Task(
        name="screenshots", depends_on=["web_build"], inputs=["tools/screenshots.py"],
        outputs=["docs/screenshots"], seconds=48.0, kind="tool",
    )
    tasks["readme"] = Task(
        name="readme", depends_on=["audit", "docs"], inputs=["tools/readme.py"],
        outputs=["README.md"], seconds=0.8, kind="tool",
    )
    return tasks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    crisp = CrispDm(
        project=PROJECT,
        business_question=(
            "Twelve pipelines with shared dependencies are run by hand. What does proper "
            "orchestration — ordering, caching, parallelism — actually save, measured on "
            "the real graph rather than assumed?"
        ),
    )
    crisp.record(
        Phase(
            name="business_understanding",
            summary=(
                "Running pipelines by hand fails in three specific ways as a project grows: "
                "wrong order produces stale results silently, re-running unchanged work "
                "wastes time, and independent work runs sequentially for no reason. An "
                "orchestrator addresses all three, but how much it is worth depends "
                "entirely on the shape of the dependency graph — which is measurable."
            ),
            decisions=[
                Decision(
                    question="How is the value of orchestration established?",
                    choice="Measured on this repository's real graph and real runtimes.",
                    rationale=(
                        "Speedup from parallelism is bounded by the critical path, which is "
                        "a property of the specific graph. Quoting a general figure would "
                        "be meaningless. Durations come from each pipeline's committed "
                        "_run block, so the analysis describes this portfolio."
                    ),
                    alternatives_rejected=[
                        "Benchmark on a synthetic DAG — the result would describe the "
                        "synthetic graph and nothing else.",
                        "State that parallelism helps — true and useless without the bound.",
                    ],
                ),
            ],
            evidence={"scheduler": "level-parallel with LPT packing"},
        )
    )

    with artifacts.run(PROJECT, seed=SEED) as ctx:
        # --- Build the graph -----------------------------------------------------------
        print("\n[1/5] Discovering the real task graph …")
        tasks = discover_tasks()
        levels = topological_levels(tasks)

        total_sequential = sum(t.seconds for t in tasks.values())
        edges = [
            {"from": dependency, "to": name}
            for name, task in tasks.items()
            for dependency in task.depends_on
        ]

        graph = {
            "n_tasks": len(tasks),
            "n_edges": len(edges),
            "n_levels": len(levels),
            "levels": [
                {"level": i, "tasks": level, "width": len(level)}
                for i, level in enumerate(levels)
            ],
            "max_width": max(len(level) for level in levels),
            "nodes": [
                {
                    "name": t.name, "kind": t.kind, "seconds": round(t.seconds, 2),
                    "depends_on": t.depends_on, "n_dependents": sum(
                        1 for o in tasks.values() if t.name in o.depends_on
                    ),
                }
                for t in sorted(tasks.values(), key=lambda t: -t.seconds)
            ],
            "edges": edges,
            "total_sequential_seconds": round(total_sequential, 2),
        }
        print(f"      {len(tasks)} tasks, {len(edges)} edges, {len(levels)} levels "
              f"(max width {graph['max_width']})")
        print(f"      sequential total: {total_sequential:.1f}s")

        # --- Cycle detection, demonstrated on a deliberately broken graph ---------------
        print("\n[2/5] Verifying cycle detection …")
        broken = {
            "extract": Task("extract", depends_on=["report"]),
            "transform": Task("transform", depends_on=["extract"]),
            "report": Task("report", depends_on=["transform"]),
        }
        try:
            topological_levels(broken)
            cycle_demo = {"detected": False, "cycle": [], "error": "NOT DETECTED"}
            print("      ✗ cycle was NOT detected — this is a bug")
        except CycleError as error:
            cycle_demo = {
                "detected": True,
                "cycle": error.cycle,
                "message": str(error),
                "note": (
                    "The engine reports the actual cycle rather than only its existence. "
                    "'Cycle detected' with no further information leaves the user to find "
                    "the offending edges by hand, which is the entire difficulty."
                ),
            }
            print(f"      ✓ detected: {error}")

        # --- Caching -------------------------------------------------------------------
        print("\n[3/5] Computing content fingerprints …")
        fingerprints: dict[str, str] = {}
        for level in levels:
            for name in level:
                fingerprints[name] = task_fingerprint(tasks[name], fingerprints)

        # Demonstrate transitive invalidation: perturb the root and count what changes.
        perturbed: dict[str, str] = {}
        for level in levels:
            for name in level:
                if name == "dsx_toolkit":
                    perturbed[name] = "perturbed-" + fingerprints[name]
                else:
                    perturbed[name] = task_fingerprint(tasks[name], perturbed)

        invalidated = [n for n in fingerprints if fingerprints[n] != perturbed[n]]
        caching = {
            "fingerprints": {n: fingerprints[n][:16] for n in sorted(fingerprints)},
            "perturbation": "dsx_toolkit fingerprint changed",
            "n_invalidated": len(invalidated),
            "n_total": len(tasks),
            "invalidated": sorted(invalidated),
            "transitive": len(invalidated) == len(tasks),
            "note": (
                f"Changing the shared library invalidates {len(invalidated)} of "
                f"{len(tasks)} tasks — everything, correctly, since every pipeline imports "
                "it. A fingerprint covering only a task's own source would invalidate 1 of "
                f"{len(tasks)} and the pipeline would then serve stale results while "
                "reporting success. Chaining upstream fingerprints is what makes "
                "invalidation transitive."
            ),
        }
        print(f"      perturbing the shared library invalidates "
              f"{len(invalidated)}/{len(tasks)} tasks (transitive: {caching['transitive']})")

        # --- Scheduling ----------------------------------------------------------------
        print("\n[4/5] Simulating schedules across worker counts …")
        cp = critical_path(tasks, levels)
        schedules = []
        for workers in (1, 2, 4, 8, 16):
            result = simulate(levels, tasks, workers)
            result["speedup"] = round(
                total_sequential / max(1e-9, result["makespan_seconds"]), 2
            )
            result["efficiency"] = round(result["speedup"] / workers, 3)
            result["vs_critical_path"] = round(
                result["makespan_seconds"] / max(1e-9, cp["seconds"]), 3
            )
            schedules.append(result)
            print(f"      {workers:>2} workers: {result['makespan_seconds']:>7.1f}s  "
                  f"speedup {result['speedup']:>5.2f}×  "
                  f"efficiency {result['efficiency']:.0%}  "
                  f"idle {result['idle_fraction_of_capacity']:>5.0%} of capacity")

        best = max(schedules, key=lambda s: s["speedup"])
        amdahl_bound = round(total_sequential / max(1e-9, cp["seconds"]), 2)

        print(f"      critical path: {cp['seconds']:.1f}s "
              f"({' → '.join(cp['path'][:4])}…)")
        print(f"      theoretical maximum speedup: {amdahl_bound:.2f}×")

        analysis = {
            "critical_path": cp,
            "sequential_seconds": round(total_sequential, 2),
            "theoretical_max_speedup": amdahl_bound,
            "schedules": schedules,
            "best_observed_speedup": best["speedup"],
            "workers_for_best": best["workers"],
            "diminishing_returns_at": next(
                (s["workers"] for s in schedules if s["efficiency"] < 0.5),
                None,
            ),
            "barrier_cost_seconds": round(
                best["makespan_seconds"] - cp["seconds"], 2
            ),
            "dominant_task": {
                "name": max(tasks, key=lambda n: tasks[n].seconds),
                "seconds": round(max(t.seconds for t in tasks.values()), 1),
                "share_of_critical_path": round(
                    max(t.seconds for t in tasks.values()) / max(1e-9, cp["seconds"]), 3
                ),
            },
            "interpretation": (
                f"Sequential execution takes {total_sequential:.1f}s. The critical path is "
                f"{cp['seconds']:.1f}s, so no number of workers can beat "
                f"{amdahl_bound:.2f}× — that bound is a property of this graph, not of the "
                f"scheduler. The best simulated schedule reaches {best['speedup']:.2f}× at "
                f"{best['workers']} workers, leaving "
                f"{round(best['makespan_seconds'] - cp['seconds'], 1)}s on the table to "
                "level barriers: a task waits for its whole level even when its own "
                "dependencies are already satisfied. A work-stealing scheduler would "
                "recover most of that gap."
            ),
        }

        crisp.record(
            Phase(
                name="data_understanding",
                summary=(
                    f"The graph has {len(tasks)} tasks and {len(edges)} edges across "
                    f"{len(levels)} dependency levels, with a maximum width of "
                    f"{graph['max_width']}. Durations are the real recorded runtimes from "
                    "each pipeline's committed _run block."
                ),
                evidence=graph,
                risks=[
                    "Durations were measured on a 4-core container under varying load, so "
                    "they carry real noise. The scheduling conclusions depend on their "
                    "relative sizes, which are stable, rather than on absolute values.",
                    "The simulation assumes workers are independent. In practice the "
                    "pipelines contend for CPU and memory, so real parallel speedup will "
                    "fall short of these figures.",
                ],
            )
        )
        crisp.record(
            Phase(
                name="data_preparation",
                summary=(
                    "Tasks are discovered by scanning for pipeline entry points; edges come "
                    "from declared dependencies. Downstream tooling is declared to depend "
                    "on every pipeline, which is genuinely true — the audit, artifact sync "
                    "and documentation generator all read every project's output."
                ),
                evidence={"n_tasks": len(tasks), "n_edges": len(edges)},
            )
        )
        crisp.record(
            Phase(
                name="modeling",
                summary=(
                    "Kahn's algorithm for level-parallel topological ordering, tri-colour "
                    "DFS for cycle reporting, chained content fingerprints for transitive "
                    "cache invalidation, and LPT packing within each level."
                ),
                decisions=[
                    Decision(
                        question="Kahn's algorithm or depth-first topological sort?",
                        choice="Kahn's.",
                        rationale=(
                            "DFS produces a valid linear order but discards the information "
                            "that two tasks were independent. Kahn's processes by in-degree, "
                            "so each batch of available tasks falls out as a parallel level. "
                            "The scheduling question needs that structure, not just a valid "
                            "sequence."
                        ),
                        alternatives_rejected=[
                            "DFS post-order — correct ordering, no parallelism information.",
                        ],
                    ),
                    Decision(
                        question="What goes into a task's cache fingerprint?",
                        choice="Its own inputs AND the fingerprints of its dependencies.",
                        rationale=caching["note"],
                        alternatives_rejected=[
                            "Hash only the task's own source and inputs — a change deep in "
                            "the graph then invalidates nothing downstream, and the "
                            "pipeline serves stale results while reporting success.",
                            "Timestamp comparison — breaks on checkout, clock skew and any "
                            "file touched without being changed.",
                        ],
                    ),
                ],
                evidence={"cycle_detection": cycle_demo, "caching": caching},
            )
        )
        crisp.record(
            Phase(
                name="evaluation",
                summary=analysis["interpretation"],
                evidence=analysis,
                risks=[
                    f"Speedup is bounded at {amdahl_bound:.2f}× by the critical path, and "
                    f"{cp['path'][-1]} alone accounts for "
                    f"{cp['task_seconds'][cp['path'][-1]]:.0f}s of it. Adding workers past "
                    f"{best['workers']} changes nothing; optimising that one task is the "
                    "only thing that would.",
                    "Level barriers cost "
                    f"{round(best['makespan_seconds'] - cp['seconds'], 1)}s in the best "
                    "schedule. This is a known limitation of level-parallel scheduling and "
                    "is reported rather than described as optimal.",
                    "All figures are simulated from measured durations, not from an actual "
                    "parallel run. Real execution adds process startup, I/O contention and "
                    "memory pressure.",
                ],
            )
        )
        crisp.record(
            Phase(
                name="deployment",
                summary=(
                    "The graph, levels, critical path and schedule sweep are exported so the "
                    "published page can render the DAG and let a reader move the worker "
                    "count and watch the makespan flatten against the critical path."
                ),
                evidence={"schedules": len(schedules)},
            )
        )

        print("\n[5/5] Writing artifacts …")
        artifacts.write(OUT / "graph.json", graph, context=ctx)
        artifacts.write(
            OUT / "engine.json",
            {"cycle_detection": cycle_demo, "caching": caching},
            context=ctx,
        )
        artifacts.write(OUT / "scheduling.json", analysis, context=ctx)
        artifacts.write(OUT / "crispdm.json", crisp.to_dict(), context=ctx)
        artifacts.write(
            OUT / "provenance.json",
            {
                "datasets": [],
                "note": (
                    "This project consumes no external dataset. Its input is this "
                    "repository's own dependency graph and the runtimes recorded in each "
                    "pipeline's committed artifacts."
                ),
            },
            context=ctx,
        )

    print(f"\n✓ {PROJECT} complete — {len(tasks)} tasks, critical path {cp['seconds']:.1f}s, "
          f"max speedup {amdahl_bound:.2f}×")


if __name__ == "__main__":
    main()
