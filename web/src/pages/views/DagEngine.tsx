import { useState } from "react";
import { useArtifacts } from "../../lib/data";
import { BarChart, LineChart, fmt } from "../../components/Charts";
import { Callout, Caveat, MetricTable, Resolved, Section, Stat } from "../../components/UI";

type Node = { name: string; kind: string; seconds: number; depends_on: string[]; n_dependents: number };

type Graph = {
  n_tasks: number; n_edges: number; n_levels: number;
  levels: { level: number; tasks: string[]; width: number }[];
  max_width: number;
  nodes: Node[];
  edges: { from: string; to: string }[];
  total_sequential_seconds: number;
};

type Schedule = {
  workers: number;
  makespan_seconds: number;
  timeline: {
    level: number; tasks: string[]; level_seconds: number; started_at: number;
    worker_loads: number[]; idle_worker_seconds: number;
  }[];
  idle_worker_seconds: number;
  idle_fraction_of_capacity: number;
  speedup: number;
  efficiency: number;
  vs_critical_path: number;
};

type Scheduling = {
  critical_path: { path: string[]; seconds: number; task_seconds: Record<string, number> };
  sequential_seconds: number;
  theoretical_max_speedup: number;
  schedules: Schedule[];
  best_observed_speedup: number;
  workers_for_best: number;
  diminishing_returns_at: number;
  barrier_cost_seconds: number;
  dominant_task: { name: string; seconds: number; share_of_critical_path: number };
  interpretation: string;
};

type Engine = {
  cycle_detection: { detected: boolean; cycle: string[]; message: string; note: string };
  caching: {
    fingerprints: Record<string, string>;
    perturbation: string;
    n_invalidated: number;
    n_total: number;
    invalidated: string[];
    transitive: boolean;
    note: string;
  };
};

export default function DagEngine() {
  const state = useArtifacts<{ graph: Graph; scheduling: Scheduling; engine: Engine }>(
    "11_pipeline_dag",
    ["graph", "scheduling", "engine"]
  );
  return (
    <Resolved state={state} what="DAG engine artifacts">
      {(d) => <Body g={d.graph} s={d.scheduling} e={d.engine} />}
    </Resolved>
  );
}

function Body({ g, s, e }: { g: Graph; s: Scheduling; e: Engine }) {
  const [workers, setWorkers] = useState(s.workers_for_best);
  const schedule = s.schedules.find((x) => x.workers === workers) ?? s.schedules[0];
  const onCriticalPath = new Set(s.critical_path.path);
  const byName = Object.fromEntries(g.nodes.map((n) => [n.name, n]));
  const slowest = [...g.nodes].sort((a, b) => b.seconds - a.seconds).slice(0, 8);

  return (
    <>
      <div className="grid grid-4" style={{ marginBottom: 8 }}>
        <Stat label="Graph" value={`${g.n_tasks} tasks`}
              sub={`${g.n_edges} edges · ${g.n_levels} levels · max width ${g.max_width}`} />
        <Stat label="Sequential" value={`${fmt(s.sequential_seconds, 1)}s`}
              sub="every task, one after another" />
        <Stat label="Critical path" value={`${fmt(s.critical_path.seconds, 1)}s`} tone="warn"
              sub={`${s.critical_path.path.length} tasks — the floor`} />
        <Stat label="Speedup ceiling" value={`${fmt(s.theoretical_max_speedup, 2)}×`} tone="accent"
              sub={`best observed ${fmt(s.best_observed_speedup, 2)}× at ${s.workers_for_best} workers`} />
      </div>

      <Callout kind="warn" title="The interesting result here is a disappointing one">
        {s.interpretation}
      </Callout>

      <Section title="Does adding workers help?" note="measured on this repository's real task durations">
        <div className="card">
          <div className="row" style={{ marginBottom: 14 }}>
            {s.schedules.map((sc) => (
              <button key={sc.workers} className={`chip${workers === sc.workers ? " active" : ""}`}
                      onClick={() => setWorkers(sc.workers)}>
                {sc.workers} worker{sc.workers === 1 ? "" : "s"}
              </button>
            ))}
          </div>
          <LineChart
            height={260}
            xLabel="workers"
            yLabel="speedup over sequential"
            yMin={0}
            series={[
              {
                name: "achieved speedup",
                points: s.schedules.map((sc) => ({ x: sc.workers, y: sc.speedup })),
              },
              {
                name: "linear (ideal)",
                dashed: true,
                points: s.schedules.map((sc) => ({ x: sc.workers, y: sc.workers })),
              },
            ]}
            markers={[{
              y: s.theoretical_max_speedup,
              label: `critical-path ceiling ${fmt(s.theoretical_max_speedup, 2)}×`,
              color: "var(--warn)",
            }]}
          />
          <p className="tiny dim" style={{ marginTop: 8 }}>
            The ideal line is what a naive capacity plan assumes. The achieved curve leaves it
            almost immediately and flattens against the ceiling — because a dependency chain
            cannot be parallelised no matter how much hardware is pointed at it.
          </p>
        </div>

        <MetricTable
          rows={s.schedules}
          highlight={(r) => r.workers === workers}
          columns={[
            { key: "workers", label: "Workers", num: true },
            { key: "makespan_seconds", label: "Makespan", num: true, render: (r) => `${fmt(r.makespan_seconds, 1)}s` },
            {
              key: "speedup", label: "Speedup", num: true,
              render: (r) => <strong className={r.speedup > 1.4 ? "good-text" : "warn-text"}>{fmt(r.speedup, 2)}×</strong>,
            },
            {
              key: "efficiency", label: "Efficiency", num: true,
              render: (r) => (
                <span className={r.efficiency < 0.3 ? "bad-text" : undefined}>
                  {(r.efficiency * 100).toFixed(0)}%
                </span>
              ),
            },
            {
              key: "idle_fraction_of_capacity", label: "Idle capacity", num: true,
              render: (r) => `${(r.idle_fraction_of_capacity * 100).toFixed(0)}%`,
            },
            {
              key: "idle_worker_seconds", label: "Idle worker-seconds", num: true,
              render: (r) => fmt(r.idle_worker_seconds, 0),
            },
            {
              key: "vs_critical_path", label: "vs critical path", num: true,
              render: (r) => `${fmt(r.vs_critical_path, 3)}×`,
            },
          ]}
        />
        <p className="small dim" style={{ marginTop: 12 }}>
          "Idle capacity" is worker-seconds unused divided by worker-seconds available — at{" "}
          {s.schedules[s.schedules.length - 1].workers} workers,{" "}
          {(s.schedules[s.schedules.length - 1].idle_fraction_of_capacity * 100).toFixed(0)}% of
          the fleet is doing nothing, and the makespan is within{" "}
          {fmt(Math.abs(s.schedules[s.schedules.length - 1].makespan_seconds - s.schedules[1].makespan_seconds), 1)}s
          of the two-worker result. This is the number to take to a discussion about buying
          more runners.
        </p>
      </Section>

      <Section title={`Schedule at ${schedule.workers} worker${schedule.workers === 1 ? "" : "s"}`}
               note="level-synchronous execution — a barrier between levels">
        <div className="card">
          <MetricTable
            rows={schedule.timeline}
            columns={[
              { key: "level", label: "Level", num: true },
              {
                key: "tasks", label: "Tasks in this level",
                render: (r) => (
                  <span className="tiny mono">
                    {r.tasks.map((t) => (
                      <span key={t} className={onCriticalPath.has(t) ? "warn-text" : undefined}>
                        {t}{" "}
                      </span>
                    ))}
                  </span>
                ),
              },
              { key: "started_at", label: "Starts at", num: true, render: (r) => `${fmt(r.started_at, 1)}s` },
              { key: "level_seconds", label: "Level takes", num: true, render: (r) => `${fmt(r.level_seconds, 1)}s` },
              {
                key: "idle_worker_seconds", label: "Idle worker-sec", num: true,
                render: (r) => fmt(r.idle_worker_seconds, 1),
              },
            ]}
          />
          <p className="tiny dim" style={{ marginTop: 8 }}>
            Tasks shown in amber lie on the critical path. A level takes as long as its slowest
            member, so one long task inside a wide level makes every other worker in that level
            wait — visible above as idle worker-seconds concentrated in a single row.
          </p>
        </div>
      </Section>

      <Section title="Where the time actually goes" note="one task dominates everything">
        <div className="grid grid-2">
          <div className="card">
            <h3>Critical path</h3>
            <BarChart
              horizontal
              height={200}
              xLabel="seconds (log-scale would hide the point, so this is linear)"
              data={s.critical_path.path.map((t) => ({
                label: t,
                value: s.critical_path.task_seconds[t],
              }))}
              valueFormat={(v) => `${fmt(v, 1)}s`}
            />
            <Callout kind="info" title={`${s.dominant_task.name} is ${(s.dominant_task.share_of_critical_path * 100).toFixed(1)}% of the path`}>
              Optimising anything else on this graph is measurable in seconds. Making that one
              task faster — or caching it — is the only change that moves the wall-clock number,
              and no scheduler can discover that for you.
            </Callout>
          </div>
          <div className="card">
            <h3>Slowest tasks overall</h3>
            <BarChart
              horizontal
              height={260}
              data={slowest.map((n) => ({
                label: n.name,
                value: n.seconds,
                color: onCriticalPath.has(n.name) ? "var(--warn)" : undefined,
              }))}
              valueFormat={(v) => `${fmt(v, 1)}s`}
            />
            <p className="tiny dim" style={{ marginTop: 6 }}>
              Amber bars sit on the critical path. Note that the second-slowest task does{" "}
              <em>not</em> — shortening it changes nothing at all, which is the practical use of
              computing the path in the first place.
            </p>
          </div>
        </div>
      </Section>

      <Section title="Topological structure" note="Kahn's algorithm — levels, not a flat order">
        <div className="card">
          <div className="stack">
            {g.levels.map((lv) => (
              <div key={lv.level} className="row" style={{ alignItems: "flex-start", gap: 12 }}>
                <span className="badge" style={{ minWidth: 76 }}>Level {lv.level}</span>
                <div style={{ flex: 1 }}>
                  {lv.tasks.map((t) => (
                    <span key={t} className="chip" style={{ marginRight: 6, marginBottom: 6 }}>
                      <span className={onCriticalPath.has(t) ? "warn-text" : undefined}>{t}</span>
                      <span className="faint"> {fmt(byName[t]?.seconds ?? 0, 1)}s</span>
                    </span>
                  ))}
                </div>
                <span className="tiny faint">width {lv.width}</span>
              </div>
            ))}
          </div>
          <p className="small dim" style={{ marginTop: 12, marginBottom: 0 }}>
            A depth-first topological sort would produce a single valid ordering and throw away
            the fact that {g.max_width} of these tasks can run at once. Kahn's algorithm peels
            the graph off in layers of zero in-degree, so the width of each layer{" "}
            <em>is</em> the available parallelism — the information a scheduler needs.
          </p>
        </div>
      </Section>

      <Section title="Cycle detection" note="what the engine says when the graph is not a DAG">
        <div className="card">
          <div className="row">
            <span className="badge badge-sim">✕ cycle detected</span>
            <code style={{ fontSize: "1.02rem" }}>{e.cycle_detection.message}</code>
          </div>
          <p className="small" style={{ margin: "12px 0 0" }}>{e.cycle_detection.note}</p>
        </div>
      </Section>

      <Section title="Content-addressed caching" note="fingerprints chain through dependencies">
        <div className="grid grid-2">
          <div className="card">
            <h3>Perturbation test</h3>
            <p className="small" style={{ marginTop: 0 }}>
              <code>{e.caching.perturbation}</code> →{" "}
              <strong className="warn-text">
                {e.caching.n_invalidated} of {e.caching.n_total} tasks invalidated
              </strong>
              {e.caching.transitive && " (transitively)"}
            </p>
            <p className="small dim">{e.caching.note}</p>
            <p className="tiny faint" style={{ marginBottom: 0 }}>
              A fingerprint is the hash of a task's own content <em>plus</em> the fingerprints of
              its dependencies. That chaining is what makes invalidation transitive without a
              separate graph walk: change one byte in the shared library and every downstream
              hash changes by construction.
            </p>
          </div>
          <div className="card">
            <h3>Fingerprints</h3>
            <div className="tiny mono" style={{ lineHeight: 1.9 }}>
              {Object.entries(e.caching.fingerprints).map(([k, v]) => (
                <div key={k} className="row" style={{ gap: 10 }}>
                  <span className="faint" style={{ minWidth: 190 }}>{k}</span>
                  <span>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Section>

      <Section title="What these numbers are not">
        <Caveat>
          <strong>The scheduler is level-synchronous, and that is pessimistic.</strong> Tasks
          are released in layers with a barrier between them, costing{" "}
          {fmt(s.barrier_cost_seconds, 1)}s per boundary in the model. A true work-stealing DAG
          scheduler would start a task the instant its own dependencies finish rather than
          waiting for its whole level, so the speedups above are a lower bound on what an
          Airflow- or Dagster-style executor would achieve. The critical-path ceiling of{" "}
          {fmt(s.theoretical_max_speedup, 2)}×, however, binds both of them equally.
        </Caveat>
        <Caveat>
          <strong>Durations come from one machine, one run.</strong> Task times were measured
          on this container's CPU with no repetition and no variance estimate. On a machine with
          a GPU the dominant task collapses and the entire shape of this analysis changes — the
          critical path would be recomputed and would likely run through different tasks. The
          method transfers; these particular seconds do not.
        </Caveat>
        <Caveat>
          <strong>Fingerprints cover code and declared inputs, not the world.</strong> A task
          that reads an environment variable, a clock, or a network resource can produce a
          different result under an unchanged fingerprint, and this engine would serve the
          stale cache entry. That is the standard failure mode of content-addressed caching and
          it is not defended against here.
        </Caveat>
      </Section>
    </>
  );
}
