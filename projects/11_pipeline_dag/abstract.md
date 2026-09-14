# Abstract — Critical-Path Limits on Pipeline Parallelism

**Objective.** Determine the achievable benefit of parallel task execution for a real data
science build, and implement the graph machinery — topological ordering, cycle diagnosis and
content-addressed invalidation — required to establish it.

**Data.** The dependency graph of this repository: 18 tasks
(48 edges, 5 topological levels, maximum level width
11), with wall-clock durations measured from actual pipeline execution totalling
3981.0s.

**Method.** Kahn's algorithm was used for topological ordering, retaining level structure so
that per-level width expresses available parallelism. Level-synchronous schedules were
simulated for 1, 2, 4, 8, 16 workers with a
0.6s barrier cost per level boundary. The critical path was
computed by longest-path relaxation over the topological order. Cycle detection uses tri-colour
depth-first search and reports the offending cycle. Cache fingerprints chain a task's content
hash with those of its dependencies; invalidation was verified by perturbation.

**Results.** The critical path is 2644.6s against 3981.0s
sequential, bounding speedup at 1.51×. Observed speedup peaked at
1.51× with 4 workers; adding workers beyond
4 produced no measurable improvement, leaving
91% of capacity idle at 16 workers. A
single task (`07_nano_transformer`, 2583.5s) constitutes
97.7% of the critical path. Perturbing the shared library's
fingerprint invalidated 18 of 18 tasks, confirming
transitive propagation.

**Conclusion.** Parallel orchestration cannot help a graph whose runtime is concentrated in one
serial chain. Reporting the critical-path ceiling alongside any scheduling result is therefore
necessary: a 1.51× measured speedup is near-optimal here, and would
be a failure on a graph with a different shape.

**Keywords.** directed acyclic graph, Kahn's algorithm, critical path, Amdahl's law, content-addressed caching
