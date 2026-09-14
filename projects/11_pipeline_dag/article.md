# I built a DAG scheduler, then measured that my pipeline didn't need one

*18 tasks, 48 edges, and a 1.51× ceiling*

Every data engineering post about DAGs ends the same way: express your pipeline as a graph,
add workers, go faster.

I wrote the scheduler. Then I measured it on my own repository, and the answer was no.

## The graph

18 tasks: a shared library, 11
analysis pipelines, and the tooling that turns their output into a website. Run one at a time,
it takes **3981.0 seconds**.

Kahn's algorithm sorts it into 5 levels, and the widest holds 11
tasks. Ten things that can run at once! This is the moment the blog post tells you to spin up
sixteen workers.

## What actually happened

| Workers | Makespan | Speedup |
|---:|---:|---:|
| 1 | 3981.0s | **1.00×** |
| 2 | 2645.6s | **1.50×** |
| 4 | 2645.2s | **1.51×** |
| 8 | 2645.2s | **1.51×** |
| 16 | 2645.2s | **1.51×** |

Two workers: 1.51×. Sixteen workers: 1.51×. The same number.

At 16 workers, 91% of the fleet does
nothing at all.

## Why

The critical path — the longest chain of tasks that must happen in order — is
**2644.6s**. Nothing can finish before that, at any worker count, ever. So the
ceiling is 3981.0 / 2644.6 =
**1.51×**.

And here is the whole story in one row:

| Task | Seconds | Share of critical path |
|---|---:|---:|
| `07_nano_transformer` | 2583.5s | **97.7%** |

Training a transformer on CPU takes 43 minutes. Everything else in this
repository, combined, takes about
23 minutes. No scheduler can parallelise a
single task with itself.

**1.51× is not a disappointing result — it is 100.0% of
the theoretical maximum.** That reframing is only available if you compute the ceiling, and
almost nobody does.

## The parts I'd actually keep

**Cycle detection that tells you where the cycle is.** Most implementations raise
`CycleError: cycle detected`, which on a graph of a hundred nodes is barely better than
silence. Mine reports:

    extract → report → transform → extract

**Caching that chains through dependencies.** A task's fingerprint hashes its own content
*plus* its dependencies' fingerprints. Change the shared library, and every downstream
fingerprint changes automatically — no invalidation walk, no bookkeeping.

I tested it by perturbing the toolkit: **18 of 18 tasks
invalidated.** Everything, correctly, because every pipeline imports it.

## The honest caveat

My scheduler releases tasks in levels with a barrier between them. A real work-stealing
executor starts a task the moment its own dependencies finish, so my speedups are a *lower*
bound on what Airflow or Dagster would do.

That does not rescue the conclusion. The 1.51× ceiling binds them
too. Better scheduling cannot shorten a chain.

If you want this build to be faster, there is exactly one move: make that one task cheaper, or
cache it. The scheduler was the fun part and the profiler was the useful one.

---

**Live demo:** [https://pranjal101shrivastava.github.io/Projects/#/p/dag-engine](https://pranjal101shrivastava.github.io/Projects/#/p/dag-engine)
**Code:** [github.com/Pranjal101Shrivastava/Projects](https://github.com/Pranjal101Shrivastava/Projects)
