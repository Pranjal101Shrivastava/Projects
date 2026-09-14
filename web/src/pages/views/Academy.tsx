import { useMemo, useState } from "react";
import { useArtifacts } from "../../lib/data";
import { BarChart, ConfusionMatrix, Histogram, LineChart, ScatterChart, fmt, seriesColor } from "../../components/Charts";
import { Callout, Caveat, MetricTable, Resolved, Section, Stat, Tabs } from "../../components/UI";

type QuizQuestion = { question: string; options: string[]; answer: number; explanation: string };

type Module = {
  title: string; dataset: string; lesson: string; quiz?: QuizQuestion[];
  [key: string]: unknown;
};

const MODULES = [
  { key: "bayes", label: "Bayes" },
  { key: "threshold", label: "Thresholds & cost" },
  { key: "gradient_descent", label: "Gradient descent" },
  { key: "backprop", label: "Backpropagation" },
  { key: "bias_variance", label: "Bias & variance" },
  { key: "sampling", label: "Sampling & CLT" },
];

export default function Academy() {
  const state = useArtifacts<Record<string, Module>>(
    "08_crispdm_academy", MODULES.map((m) => m.key)
  );
  const [active, setActive] = useState("bayes");

  return (
    <Resolved state={state} what="teaching modules">
      {(d) => (
        <>
          <Callout kind="info" title="Every figure here is computed from real data">
            Teaching material almost always uses generated data, because generated data
            satisfies its assumptions. Students then meet real data where the assumptions
            fail and have no framework for it. Where an assumption breaks below, the breakage
            is the lesson.
          </Callout>

          <Tabs tabs={MODULES} active={active} onChange={setActive} />

          {active === "bayes" && <BayesModule m={d.bayes} />}
          {active === "threshold" && <ThresholdModule m={d.threshold} />}
          {active === "gradient_descent" && <GradientModule m={d.gradient_descent} />}
          {active === "backprop" && <BackpropModule m={d.backprop} />}
          {active === "bias_variance" && <BiasVarianceModule m={d.bias_variance} />}
          {active === "sampling" && <SamplingModule m={d.sampling} />}

          <Quiz questions={(d[active].quiz ?? []) as QuizQuestion[]} />
        </>
      )}
    </Resolved>
  );
}

function Lesson({ m }: { m: Module }) {
  return (
    <div className="card" style={{ marginBottom: 18 }}>
      <div className="row" style={{ marginBottom: 8 }}>
        <h3 style={{ margin: 0 }}>{m.title}</h3>
        <span className="spacer" />
        <span className="badge badge-real">● {m.dataset}</span>
      </div>
      <p style={{ margin: 0 }}>{m.lesson}</p>
    </div>
  );
}

/* ---------------------------------------------------------------- Bayes */

function BayesModule({ m }: { m: Module }) {
  const dep = m.dependence_check as { pair: string; cramers_v_given_died: number; cramers_v_given_survived: number; violates_independence: boolean }[];
  const cmp = m.comparison as Record<string, { roc_auc: number; brier: number }>;
  const prior = m.prior as { survived: number; died: number };
  const likelihoods = m.likelihoods as Record<string, Record<string, { p_given_survived: number; p_given_died: number }>>;

  const [pclass, setPclass] = useState("1");
  const [sex, setSex] = useState("female");

  // Naive Bayes by hand, so the arithmetic is visible rather than hidden in a library.
  const posterior = useMemo(() => {
    const lsSurv = likelihoods.Pclass[pclass].p_given_survived * likelihoods.Sex[sex].p_given_survived;
    const lsDied = likelihoods.Pclass[pclass].p_given_died * likelihoods.Sex[sex].p_given_died;
    const numerator = prior.survived * lsSurv;
    return numerator / (numerator + prior.died * lsDied);
  }, [pclass, sex, likelihoods, prior]);

  return (
    <>
      <Lesson m={m} />
      <div className="grid grid-2">
        <div className="card">
          <h3>Compute a posterior</h3>
          <div className="controls">
            <div className="control">
              <label className="control-label" htmlFor="pc">Passenger class</label>
              <select id="pc" value={pclass} onChange={(e) => setPclass(e.target.value)}>
                {Object.keys(likelihoods.Pclass).map((k) => <option key={k} value={k}>Class {k}</option>)}
              </select>
            </div>
            <div className="control">
              <label className="control-label" htmlFor="sx">Sex</label>
              <select id="sx" value={sex} onChange={(e) => setSex(e.target.value)}>
                {Object.keys(likelihoods.Sex).map((k) => <option key={k} value={k}>{k}</option>)}
              </select>
            </div>
            <div className="control">
              <div className="control-label">P(survived | evidence)</div>
              <div className="stat-value accent" style={{ fontSize: "1.9rem" }}>
                {(posterior * 100).toFixed(1)}%
              </div>
              <div className="stat-sub">prior was {(prior.survived * 100).toFixed(1)}%</div>
            </div>
          </div>
          <pre className="sample" style={{ maxHeight: 150 }}>
{`posterior ∝ prior × ∏ P(feature | class)

P(survived)            = ${prior.survived.toFixed(4)}
P(class ${pclass} | survived)    = ${likelihoods.Pclass[pclass].p_given_survived.toFixed(4)}
P(${sex} | survived) = ${likelihoods.Sex[sex].p_given_survived.toFixed(4)}
                       ─────────
P(survived | evidence) = ${posterior.toFixed(4)}`}
          </pre>
        </div>

        <div className="card">
          <h3>How badly is independence violated?</h3>
          <BarChart
            horizontal
            height={190}
            data={dep.map((d) => ({
              label: d.pair,
              value: Math.max(d.cramers_v_given_died, d.cramers_v_given_survived),
              color: d.violates_independence ? "var(--bad)" : "var(--good)",
            }))}
            valueFormat={(v) => fmt(v, 3)}
          />
          <p className="tiny dim" style={{ marginTop: 8 }}>
            Cramér's V between feature pairs, computed within each class. Values above 0.2
            (red) mean the pair is materially dependent given the class — exactly what Naive
            Bayes assumes away.
          </p>
          <div className="grid grid-2" style={{ marginTop: 12 }}>
            <Stat label="Naive Bayes" value={fmt(cmp.naive_bayes.roc_auc, 3)}
                  sub={`ROC-AUC · Brier ${fmt(cmp.naive_bayes.brier, 4)}`} />
            <Stat label="Logistic regression" value={fmt(cmp.logistic_regression.roc_auc, 3)}
                  sub={`ROC-AUC · Brier ${fmt(cmp.logistic_regression.brier, 4)}`} />
          </div>
        </div>
      </div>
    </>
  );
}

/* ------------------------------------------------------------ Threshold */

function ThresholdModule({ m }: { m: Module }) {
  const roc = m.roc_curve as { fpr: number; tpr: number }[];
  const pr = m.pr_curve as { recall: number; precision: number }[];
  const confusions = m.confusion_at_thresholds as { threshold: number; tp: number; fp: number; fn: number; tn: number; precision: number; recall: number; accuracy: number; alerts: number }[];
  const [idx, setIdx] = useState(2);
  const c = confusions[idx];

  return (
    <>
      <Lesson m={m} />
      <div className="grid grid-2">
        <div className="card">
          <h3>ROC curve</h3>
          <LineChart
            height={250}
            xLabel="false positive rate"
            yLabel="true positive rate"
            showLegend={false}
            xMin={0} xMax={1} yMin={0} yMax={1}
            series={[
              { name: "model", points: roc.map((r) => ({ x: r.fpr, y: r.tpr })) },
              { name: "chance", points: [{ x: 0, y: 0 }, { x: 1, y: 1 }], color: "var(--text-faint)", dashed: true },
            ]}
          />
          <p className="tiny dim">ROC-AUC {fmt(m.roc_auc as number, 4)} — looks excellent.</p>
        </div>
        <div className="card">
          <h3>Precision-recall curve</h3>
          <LineChart
            height={250}
            xLabel="recall"
            yLabel="precision"
            showLegend={false}
            xMin={0} xMax={1} yMin={0} yMax={1}
            series={[{ name: "model", points: pr.map((r) => ({ x: r.recall, y: r.precision })) }]}
            markers={[{ y: m.no_skill_pr as number, label: "no-skill floor", color: "var(--bad)" }]}
          />
          <p className="tiny dim">
            PR-AUC {fmt(m.pr_auc as number, 4)} — the same model, far more sober. The no-skill
            floor sits at {fmt(m.no_skill_pr as number, 5)}, the prevalence itself.
          </p>
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Move the threshold</h3>
        <div className="controls">
          <div className="control" style={{ minWidth: 260 }}>
            <label className="control-label" htmlFor="thr">Threshold — {c.threshold}</label>
            <input id="thr" type="range" min={0} max={confusions.length - 1} value={idx}
                   onChange={(e) => setIdx(Number(e.target.value))} />
          </div>
          <Stat label="Precision" value={`${(c.precision * 100).toFixed(1)}%`} />
          <Stat label="Recall" value={`${(c.recall * 100).toFixed(1)}%`} />
          <Stat label="Accuracy" value={`${(c.accuracy * 100).toFixed(3)}%`} tone="warn"
                sub="barely moves — that is the trap" />
        </div>
        <ConfusionMatrix tp={c.tp} fp={c.fp} fn={c.fn} tn={c.tn}
                         positiveLabel="Fraud" negativeLabel="Legitimate" />
        <p className="tiny dim" style={{ marginTop: 10 }}>
          Watch accuracy as you drag: it stays pinned near 99.9% across the entire range while
          precision and recall move by tens of percentage points. That insensitivity is
          exactly why accuracy must not be quoted on an imbalanced problem.
        </p>
      </div>
    </>
  );
}

/* ------------------------------------------------------- Gradient descent */

function GradientModule({ m }: { m: Module }) {
  const traj = m.trajectories as Record<string, { learning_rate: number; path: { step: number; w: number; b: number; loss: number }[]; diverged: boolean; converged: boolean; behaviour: string; loss_ratio_to_optimum: number }>;
  const surface = m.surface as { w: number; b: number; loss: number }[];
  const optimum = m.optimum as { w: number; b: number; loss: number };
  const rates = Object.keys(traj);
  const [rate, setRate] = useState(rates[1]);
  const t = traj[rate];

  return (
    <>
      <Lesson m={m} />
      <div className="grid grid-2">
        <div className="card">
          <h3>Loss surface and descent path</h3>
          <ScatterChart
            height={320}
            xLabel="w (slope)"
            yLabel="b (intercept)"
            radius={2.6}
            opacity={0.5}
            points={[
              ...surface
                .filter((_, i) => i % 3 === 0)
                .map((s) => ({ x: s.w, y: s.b, c: Math.min(7, Math.floor(Math.log10(s.loss + 1) * 3)), label: `loss ${fmt(s.loss, 3)}` })),
            ]}
          />
          <p className="tiny dim" style={{ marginTop: 6 }}>
            Convex surface with a single minimum at w = {fmt(optimum.w, 3)}, b ={" "}
            {fmt(optimum.b, 3)}, loss {fmt(optimum.loss, 4)}. Colour encodes loss magnitude.
          </p>
        </div>
        <div className="card">
          <h3>Loss by step</h3>
          <div className="row" style={{ marginBottom: 10 }}>
            {rates.map((r) => (
              <button key={r} className={`chip${rate === r ? " active" : ""}`} onClick={() => setRate(r)}>
                lr = {r}
              </button>
            ))}
          </div>
          <LineChart
            height={230}
            xLabel="step"
            yLabel="loss (log scale)"
            showLegend={false}
            logY
            series={[{
              name: "loss",
              points: t.path.map((p) => ({ x: p.step, y: Math.max(1e-6, p.loss) })),
              color: t.diverged ? "var(--bad)" : "var(--good)",
            }]}
          />
          <Callout kind={t.diverged ? "bad" : t.converged ? "good" : "warn"}
                   title={t.diverged ? "Diverged" : t.converged ? "Converged" : "Converging slowly"}>
            {t.behaviour}. Final loss is{" "}
            <strong>{fmt(t.loss_ratio_to_optimum, 2)}×</strong> the optimum.
          </Callout>
        </div>
      </div>
    </>
  );
}

/* ------------------------------------------------------------ Backprop */

function BackpropModule({ m }: { m: Module }) {
  const checks = m.gradient_checks as { parameter: string; analytic: number; numerical: number; relative_error: number }[];
  const steps = m.chain_rule_steps as { step: string; expression: string; note: string }[];
  const curve = m.training_curve as { step: number; loss: number }[];

  return (
    <>
      <Lesson m={m} />
      <div className="grid grid-4" style={{ marginBottom: 16 }}>
        <Stat label="Max relative error" value={(m.max_relative_error as number).toExponential(2)}
              tone={m.check_passed ? "good" : "bad"} />
        <Stat label="Check" value={m.check_passed ? "PASSED" : "FAILED"}
              tone={m.check_passed ? "good" : "bad"} sub="analytic vs finite differences" />
        <Stat label="Initial loss" value={fmt(m.initial_loss as number, 4)} />
        <Stat label="Final loss" value={fmt(m.final_loss as number, 4)} tone="accent" />
      </div>

      <div className="grid grid-2">
        <div className="card">
          <h3>Gradient check</h3>
          <MetricTable
            rows={checks}
            columns={[
              { key: "parameter", label: "Parameter" },
              { key: "analytic", label: "Analytic", num: true, render: (r) => r.analytic.toFixed(8) },
              { key: "numerical", label: "Numerical", num: true, render: (r) => r.numerical.toFixed(8) },
              {
                key: "relative_error", label: "Rel. error", num: true,
                render: (r) => <span className="good-text">{r.relative_error.toExponential(1)}</span>,
              },
            ]}
          />
          <p className="tiny dim" style={{ marginTop: 8 }}>
            Central finite differences: (f(θ+ε) − f(θ−ε)) / 2ε. Agreement to ~1e-8 proves the
            derivation is right. This is the first thing to run when a network will not train
            — an error near 0.3 means a bug, not floating-point noise.
          </p>
        </div>
        <div className="card">
          <h3>The chain rule, step by step</h3>
          <div className="stack">
            {steps.map((s, i) => (
              <div key={i} className="decision">
                <div className="row">
                  <span className="decision-q mono">{s.step}</span>
                  <span className="spacer" />
                  <span className="decision-a mono">{s.expression}</span>
                </div>
                <div className="decision-why tiny">{s.note}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Training with the hand-derived gradients</h3>
        <LineChart
          height={200}
          xLabel="step"
          yLabel="binary cross-entropy"
          showLegend={false}
          series={[{ name: "loss", points: curve.map((c) => ({ x: c.step, y: c.loss })) }]}
        />
      </div>
    </>
  );
}

/* ------------------------------------------------------- Bias / variance */

function BiasVarianceModule({ m }: { m: Module }) {
  const curve = m.curve as { degree: number; bias_squared: number; variance: number; total_test_mse: number; train_mse: number }[];

  return (
    <>
      <Lesson m={m} />
      <div className="grid grid-4" style={{ marginBottom: 16 }}>
        <Stat label="Optimal degree" value={m.optimal_degree as number} tone="accent" />
        <Stat label="Bootstrap resamples" value={m.n_bootstrap as number} />
        <Stat label="Variance share at max degree"
              value={`${((m.variance_share_at_max_degree as number) * 100).toFixed(1)}%`}
              tone="warn" sub="bias dominates throughout" />
        <Stat label="Training error monotone?"
              value={m.train_error_monotone ? "yes" : "no"}
              tone={m.train_error_monotone ? "good" : "warn"} />
      </div>

      <div className="card">
        <h3>Decomposition by model capacity</h3>
        <LineChart
          height={280}
          xLabel="polynomial degree"
          yLabel="mean squared error"
          series={[
            { name: "bias²", points: curve.map((c) => ({ x: c.degree, y: c.bias_squared })) },
            { name: "variance", points: curve.map((c) => ({ x: c.degree, y: c.variance })), color: seriesColor(1) },
            { name: "total test MSE", points: curve.map((c) => ({ x: c.degree, y: c.total_test_mse })), color: seriesColor(2) },
            { name: "train MSE", points: curve.map((c) => ({ x: c.degree, y: c.train_mse })), color: seriesColor(7), dashed: true },
          ]}
          markers={[{ x: m.optimal_degree as number, label: "optimum", color: "var(--good)" }]}
        />
      </div>

      <Caveat>
        This real curve is <strong>not</strong> the textbook U. Variance rises monotonically
        as expected, but accounts for only{" "}
        {((m.variance_share_at_max_degree as number) * 100).toFixed(1)}% of test error even at
        the highest degree — bias dominates throughout, because day-of-year simply does not
        determine daily temperature and the irreducible noise is large. Training error is also
        not monotone here, because each degree is averaged over bootstrap resamples and
        resampling noise exceeds the small gain from extra capacity once bias has plateaued.
        A simulated example would show the clean U and hide both of these.
      </Caveat>
    </>
  );
}

/* -------------------------------------------------------------- Sampling */

function SamplingModule({ m }: { m: Module }) {
  const pop = m.population as { skewness: number; mean: number; std: number; median: number; histogram: any };
  const dists = m.sampling_distributions as { sample_size: number; histogram: any; observed_se: number; predicted_se: number; skewness: number }[];
  const [idx, setIdx] = useState(dists.length - 3);
  const d = dists[idx];

  return (
    <>
      <Lesson m={m} />
      <div className="grid grid-2">
        <div className="card">
          <h3>Population — transaction amounts</h3>
          <Histogram data={pop.histogram} xLabel="amount (€, truncated at 500)" color="var(--c5)" />
          <div className="grid grid-3" style={{ marginTop: 10 }}>
            <Stat label="Skewness" value={fmt(pop.skewness, 2)} tone="warn" />
            <Stat label="Mean" value={fmt(pop.mean, 2)} />
            <Stat label="Median" value={fmt(pop.median, 2)} />
          </div>
        </div>
        <div className="card">
          <h3>Sampling distribution of the mean</h3>
          <div className="row" style={{ marginBottom: 10 }}>
            {dists.map((x, i) => (
              <button key={x.sample_size} className={`chip${idx === i ? " active" : ""}`}
                      onClick={() => setIdx(i)}>
                n = {x.sample_size}
              </button>
            ))}
          </div>
          <Histogram data={d.histogram} xLabel={`mean of ${d.sample_size} draws`} />
          <div className="grid grid-3" style={{ marginTop: 10 }}>
            <Stat label="Skewness" value={fmt(d.skewness, 3)}
                  tone={Math.abs(d.skewness) < 0.5 ? "good" : "warn"} />
            <Stat label="Observed SE" value={fmt(d.observed_se, 2)} />
            <Stat label="σ/√n predicted" value={fmt(d.predicted_se, 2)} />
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Convergence toward normality</h3>
        <LineChart
          height={230}
          xLabel="sample size n"
          yLabel="skewness of the sample mean"
          showLegend={false}
          logY={false}
          series={[{ name: "skewness", points: dists.map((x) => ({ x: x.sample_size, y: x.skewness })) }]}
          markers={[{ x: 30, label: "the 'n=30' rule", color: "var(--bad)" }]}
        />
        <p className="tiny dim" style={{ marginTop: 8 }}>
          At n = 30 — the familiar rule-of-thumb threshold — the sample mean is still
          skewed {fmt(dists.find((x) => x.sample_size === 30)?.skewness ?? 0, 2)}. That rule
          is calibrated on mildly non-normal populations, not on one with skewness{" "}
          {fmt(pop.skewness, 1)}. The n = 1 and n = 2 points are erratic because 1,500 draws
          from a tail this heavy under-sample the extremes.
        </p>
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ Quiz */

function Quiz({ questions }: { questions: QuizQuestion[] }) {
  const [answers, setAnswers] = useState<Record<number, number>>({});
  if (!questions.length) return null;

  return (
    <Section title="Check your understanding">
      <div className="stack">
        {questions.map((q, qi) => {
          const chosen = answers[qi];
          const answered = chosen !== undefined;
          return (
            <div className="card" key={qi}>
              <p style={{ fontWeight: 600 }}>{q.question}</p>
              <div className="stack" style={{ gap: 6 }}>
                {q.options.map((opt, oi) => {
                  const isCorrect = oi === q.answer;
                  const isChosen = chosen === oi;
                  const tone = !answered ? "" : isCorrect ? " active" : isChosen ? "" : "";
                  return (
                    <button
                      key={oi}
                      className={`chip${tone}`}
                      style={{
                        justifyContent: "flex-start",
                        textAlign: "left",
                        borderColor: answered
                          ? isCorrect ? "var(--good)" : isChosen ? "var(--bad)" : "var(--border)"
                          : undefined,
                        color: answered
                          ? isCorrect ? "var(--good)" : isChosen ? "var(--bad)" : "var(--text-dim)"
                          : undefined,
                      }}
                      onClick={() => setAnswers((a) => ({ ...a, [qi]: oi }))}
                    >
                      {answered && isCorrect ? "✓ " : answered && isChosen ? "✕ " : ""}
                      {opt}
                    </button>
                  );
                })}
              </div>
              {answered && (
                <Callout kind={chosen === q.answer ? "good" : "warn"}
                         title={chosen === q.answer ? "Correct" : "Not quite"}>
                  {q.explanation}
                </Callout>
              )}
            </div>
          );
        })}
      </div>
    </Section>
  );
}
