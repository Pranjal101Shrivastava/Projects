# Scheduling a Data Science Build: Where the Parallelism Isn't

*Measured on a 18-task, 48-edge dependency graph*

## 1. Problem

Workflow orchestrators are adopted on the premise that expressing a pipeline as a DAG unlocks
parallel execution. That premise is rarely tested against the graph in question. This study
implements the machinery and measures the benefit on a real build.

## 2. Graph construction

Tasks are the pipelines, the shared library, and the downstream tooling steps of this
repository. Edges are genuine data dependencies: a pipeline depends on the toolkit it imports,
the artifact sync depends on every pipeline, the web build depends on the sync, screenshots
depend on the web build.

| Property | Value |
|---|---:|
| Tasks | 18 |
| Edges | 48 |
| Topological levels | 5 |
| Maximum level width | 11 |
| Sequential runtime | 3981.0s |

## 3. Topological ordering

Kahn's algorithm repeatedly removes all nodes of in-degree zero. Unlike a DFS-based
topological sort, which produces a single linear order, this retains the level structure:

| Level | Width | Tasks |
|---:|---:|---|
| 0 | 1 | `dsx_toolkit` |
| 1 | 11 | `01_nyc_mobility`, `02_customer_segmentation`, `03_market_basket`, `04_fraud_detection`, `05_timeseries_forecasting`, `06_automl_tournament`, `07_nano_transformer`, `08_crispdm_academy`, `09_similarity_search`, `10_fairness_audit`, `12_market_backtest` |
| 2 | 3 | `audit`, `docs`, `sync_artifacts` |
| 3 | 2 | `readme`, `web_build` |
| 4 | 1 | `screenshots` |

Level width is exactly the number of tasks eligible to run concurrently, which is what a
scheduler consumes.

## 4. Critical path

Longest-path relaxation over the topological order gives:

| Position | Task | Seconds | Share |
|---:|---|---:|---:|
| 1 | `dsx_toolkit` | 0.5s | 0.0% |
| 2 | `07_nano_transformer` | 2583.5s | 97.7% |
| 3 | `sync_artifacts` | 0.6s | 0.0% |
| 4 | `web_build` | 12.0s | 0.5% |
| 5 | `screenshots` | 48.0s | 1.8% |

Total 2644.6s. By Amdahl's argument the maximum achievable speedup is
3981.0 / 2644.6 = **1.51×**,
regardless of worker count.

## 5. Scheduling results

Level-synchronous simulation, barrier cost 0.6s per boundary:

| Workers | Makespan | Speedup | Efficiency | Idle capacity | vs critical path |
|---:|---:|---:|---:|---:|---:|
| 1 | 3981.0s | 1.00× | 100% | 0% | 1.505× |
| 2 | 2645.6s | 1.50× | 75% | 25% | 1.000× |
| 4 | 2645.2s | 1.51× | 38% | 62% | 1.000× |
| 8 | 2645.2s | 1.51× | 19% | 81% | 1.000× |
| 16 | 2645.2s | 1.51× | 9% | 91% | 1.000× |

Sequential execution takes 3981.0s. The critical path is 2644.6s, so no number of workers can beat 1.51× — that bound is a property of this graph, not of the scheduler. The best simulated schedule reaches 1.51× at 4 workers, leaving 0.6s on the table to level barriers: a task waits for its whole level even when its own dependencies are already satisfied. A work-stealing scheduler would recover most of that gap.

"Idle capacity" is unused worker-seconds divided by available worker-seconds — a normalised
figure, because raw idle worker-seconds necessarily grow with fleet size and are easy to
misread as a worsening problem.

## 6. Cycle detection

Tri-colour depth-first search distinguishes an in-progress node (grey) from a finished one
(black); an edge into a grey node closes a cycle, and the stack between them *is* the cycle.
On a deliberately cyclic test graph the engine reports:

    extract → report → transform → extract

The engine reports the actual cycle rather than only its existence. 'Cycle detected' with no further information leaves the user to find the offending edges by hand, which is the entire difficulty.

## 7. Content-addressed caching

fingerprint(t) = H(content(t) ‖ fingerprint(d₁) ‖ … ‖ fingerprint(dₙ)) over dependencies dᵢ in
topological order.

Because the hash chains through dependencies, invalidation is transitive by construction.
Perturbation test: `dsx_toolkit fingerprint changed` invalidated 18 of
18 tasks.

Changing the shared library invalidates 18 of 18 tasks — everything, correctly, since every pipeline imports it. A fingerprint covering only a task's own source would invalidate 1 of 18 and the pipeline would then serve stale results while reporting success. Chaining upstream fingerprints is what makes invalidation transitive.

## 8. Limitations

- Durations were measured on a 4-core container under varying load, so they carry real noise. The scheduling conclusions depend on their relative sizes, which are stable, rather than on absolute values.
- The simulation assumes workers are independent. In practice the pipelines contend for CPU and memory, so real parallel speedup will fall short of these figures.
- Speedup is bounded at 1.51× by the critical path, and screenshots alone accounts for 48s of it. Adding workers past 4 changes nothing; optimising that one task is the only thing that would.
- Level barriers cost 0.6s in the best schedule. This is a known limitation of level-parallel scheduling and is reported rather than described as optimal.
- All figures are simulated from measured durations, not from an actual parallel run. Real execution adds process startup, I/O contention and memory pressure.

*Generated at commit `dff3e52` · seed 42 · 0.08s · Python 3.11.15 · numpy 2.4.6 · pandas 3.0.5 · sklearn 1.9.1 · lightgbm 4.7.0 · torch 2.14.0+cu130*
