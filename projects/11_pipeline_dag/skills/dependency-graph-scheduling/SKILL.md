---
name: dependency-graph-scheduling
description: Decide whether parallelising a pipeline can help before building the orchestration
---

# Decide whether parallelising a pipeline can help before building the orchestration

*Derived from [`projects/11_pipeline_dag`](../../), which applied this procedure to: .*

This is a **transferable procedure**, not instructions for rerunning the script in this
repository. It states what to do on a new dataset, where the judgement calls are, and which
failure modes this project actually hit.

## When to use this

Someone has proposed parallelising a pipeline, adding CI runners, or adopting an
orchestrator, and you would like to know in advance whether it will help.

## Procedure

### 1. Write the graph down before optimising anything

Tasks and their real data dependencies. Not the order you happen to run them in — the order
they *must* run in. The difference between those two is the parallelism you are looking for.

### 2. Sort with Kahn's algorithm, not depth-first

DFS gives you one valid linear order and discards the width information. Kahn's algorithm
peels off layers of zero in-degree, and each layer's width is exactly how many tasks can run
at once. That is the number a scheduler consumes.

### 3. Compute the critical path first — it bounds everything

Longest-path relaxation over the topological order. Speedup can never exceed
sequential ÷ critical path, at any worker count, with any scheduler. Compute this number
before writing any scheduling code: it frequently ends the discussion.

### 4. Measure task durations, and look at their distribution

If one task dominates the critical path, no orchestration will help and the only useful moves
are making that task faster or caching it. This is common and rarely checked.

### 5. Normalise your idle metric

Raw idle worker-seconds grow with fleet size and read as a worsening problem when nothing has
changed. Report idle ÷ available capacity so the numbers are comparable across worker counts.

### 6. Make cycle detection report the cycle

Tri-colour DFS: an edge into a grey (in-progress) node closes a cycle, and the stack between
them is the cycle. `CycleError: cycle detected` on a hundred-node graph is barely better than
silence.

### 7. Chain cache fingerprints through dependencies

fingerprint(t) = H(content(t) ‖ fingerprints of dependencies). Invalidation becomes transitive
by construction — no separate graph walk, no bookkeeping to get wrong. Verify it by
perturbing a root node and confirming everything downstream invalidates.

## Decision points requiring judgement

**Level-synchronous scheduling is pessimistic.** A work-stealing executor starts a task the
moment its own dependencies finish rather than waiting for its whole level. Your measured
speedups are a lower bound — but the critical-path ceiling binds both.

**Durations are machine-specific.** Move to different hardware and the critical path may run
through entirely different tasks. The method transfers; the seconds do not.

## Failure modes this project hit

**Reporting a speedup without its ceiling.** 1.50× sounds like a poor result until you know
the maximum is 1.51×, at which point it is near-optimal. A scheduling number without the
ceiling beside it cannot be interpreted in either direction.

**Fingerprints that cover code but not the world.** A task reading a clock, an environment
variable or a network resource can produce different output under an unchanged fingerprint,
and the cache will serve the stale entry. Know that this is the failure mode you have
accepted.

---

## Reproduce the reference implementation

```bash
export PYTHONPATH=lib
python3 projects/11_pipeline_dag/pipeline/build.py
python3 tools/audit.py
```

Results, with their run provenance, land in
[`projects/11_pipeline_dag/artifacts/`](../../artifacts/).
