# Video 11 · Pipeline DAG Engine

**Estimated runtime:** ~5:53 at a measured pace (795 spoken words)
**Live page:** https://pranjal101shrivastava.github.io/Projects/#/p/dag-engine
**Project directory:** [`projects/11_pipeline_dag/`](../../projects/11_pipeline_dag/)

> **The one sentence this video has to land:** The critical path caps speedup at 1.51x at any worker count, because one task is 97.7% of it — so 1.50x is near-optimal, not disappointing.

## Before you record

- Open **https://pranjal101shrivastava.github.io/Projects/#/p/dag-engine**.
- Click through the worker-count selector on camera — the flat curve is the whole argument.
- Have the cycle-detection message and the fingerprint table located in advance.

Everything in plain text below is meant to be **spoken**. Lines marked *On screen* and
*Note to self* are directions — do not read them aloud.

---

## [0:00] Cold open

> **On screen —** the speedup-vs-workers chart

Every article about workflow orchestration ends the same way: express your pipeline as a
graph, add workers, go faster.

I wrote the scheduler. Then I measured it on my own repository.

Two workers: 1.50 times faster. Sixteen workers:
1.51 times faster.

That's the same number. Sixteen workers bought me nothing over two, and this video is about
why — and about how you can know that *before* you build the orchestration.

## [0:32] The graph

> **On screen —** the topological levels panel

The graph is this repository's own build. 18 tasks, 48 edges: a
shared library, 11 analysis pipelines, and the tooling that turns their output
into a website.

The edges are real data dependencies. A pipeline depends on the library it imports. The
artifact sync depends on every pipeline. The site build depends on the sync. Screenshots
depend on the site build.

Run it one task at a time: 3981 seconds.

Sorting it gives 5 levels, and the widest holds 11 tasks.
11 things that can run at once! This is the moment the blog post tells you to
spin up sixteen workers.

## [1:17] Why Kahn's algorithm and not depth-first

> **On screen —** the levels list

Quick but important point about the sort.

A depth-first topological sort gives you one valid linear ordering. It's correct, and it
throws away the thing you actually need — it tells you a legal sequence, not what can happen
simultaneously.

Kahn's algorithm peels the graph off in layers. Take everything with no remaining
dependencies, that's layer zero. Remove it, repeat.

The **width of each layer is the available parallelism**. That's the number a scheduler
consumes, and it only exists if you sort this way.

## [1:54] The critical path, which ends the discussion

> **On screen —** the critical path bar chart

Here's the number I'd compute before writing a single line of scheduler.

The critical path is the longest chain of tasks that must happen in order. Nothing can
finish before it completes — at any worker count, with any scheduler, ever.

Here it's 2645 seconds, against 3981 sequential.

So the ceiling is 3981 divided by 2645 —
**1.51 times**. That's the maximum. Full stop.

And here is the whole story in one row: the task `07_nano_transformer` takes
2584 seconds, which is **97.7% of the
critical path**.

It's the transformer from video seven, training on a CPU. Everything else in this repository
put together is a few minutes. No scheduler can parallelise a single task with itself.

Now reframe the opening number. 1.50 times sounds disappointing — until you
know the maximum is 1.51. Then it's
99% of what's achievable.

A scheduling result cannot be interpreted in *either* direction without its ceiling printed
next to it. That's the same argument as the no-skill baseline, applied to systems work.

## [3:08] The number to take to a budget meeting

> **On screen —** the scheduling table, idle capacity column

Look at the idle column at 16 workers:
91% of the fleet is doing nothing, and the makespan
is within a second or so of the two-worker result.

One note on how that's measured. Raw idle *worker-seconds* grow automatically with fleet
size — a bigger fleet idles more by definition — so quoting them makes the problem look like
it's getting worse when nothing has changed. This is idle divided by available capacity,
which is comparable across worker counts.

An earlier version of this project reported the raw figure and it was genuinely misleading,
so it's now named and normalised.

## [3:52] Two pieces I'd keep

> **On screen —** the cycle detection message, then the fingerprint table

Two parts of the engine I think are worth stealing regardless.

**Cycle detection that names the cycle.** Most implementations raise "cycle detected". On a
hundred-node graph, that's barely better than silence. Mine does a tri-colour depth-first
search: an edge into a node that's currently in progress closes a cycle, and the stack
between them *is* the cycle. So it reports:

extract → report → transform → extract

That's a debuggable error message.

**Caching that chains through dependencies.** A task's fingerprint is a hash of its own
content **plus the fingerprints of everything it depends on**. Because the hash chains,
invalidation is transitive by construction — no separate graph walk, no bookkeeping to get
out of sync.

I tested it by perturbing the shared library:
**18 of 18 tasks invalidated.** Everything, correctly,
because every pipeline imports it.

## [4:51] Close

> **On screen —** the limitations section

Two honest caveats.

My scheduler releases tasks in levels, with a barrier between them. A real work-stealing
executor starts a task the moment its own dependencies finish, so my speedups are a **lower**
bound on what Airflow or Dagster would achieve.

That doesn't rescue the conclusion, though — the 1.51 times
ceiling binds them too. Better scheduling cannot shorten a chain.

And these durations come from one machine, one run. On a machine with a GPU the dominant task
collapses and the critical path runs through completely different tasks. The method
transfers. The seconds don't.

If you want this build faster there is exactly one move: make that one task cheaper, or cache
it. Writing the scheduler was the fun part. The profiler was the useful one.

Last video: a trading strategy that finds nothing, and why that's the point.
---

*Generated by [`tools/video_scripts.py`](../../tools/video_scripts.py) from committed
artifacts. Every figure spoken here is the figure on screen.*
