import { Callout, Section } from "../components/UI";

/**
 * The standing argument of the portfolio.
 *
 * This page exists because the individual projects each demonstrate one or two of these
 * rules, and the rules only add up to a position when stated together.
 */
export default function Methodology() {
  return (
    <div className="wrap">
      <header className="hero">
        <h1>Methodology</h1>
        <p className="lead">
          Eight projects sharing one library, one set of rules, and one commitment: every
          number on this site is read from a JSON artifact written by a pipeline in the
          repository, so the prose and the measurements cannot drift apart.
        </p>
      </header>

      <Section title="Five rules, enforced in code">
        <div className="stack">
          <Rule
            n="01"
            title="No undeclared data, and no relabelled simulations"
            code="lib/dsx/data.py"
          >
            Every dataset is declared once in a registry with its true origin, licence and an
            explicit <code>REAL</code> or <code>SIMULATED</code> marker. A model script
            cannot fetch an arbitrary URL; it asks the registry for a declared id, and an
            unknown id raises. Downloads are cached and pinned by SHA-256, so an upstream
            file that changes underneath the work fails loudly rather than quietly shifting
            every metric downstream. All eleven sources in this portfolio are{" "}
            <strong className="good-text">REAL</strong>.
          </Rule>

          <Rule n="02" title="Splitting that resists the three leakage modes" code="lib/dsx/splits.py">
            <strong>Preprocessing leakage</strong> is prevented structurally:{" "}
            <code>assert_pipeline_safe()</code> refuses any estimator not wrapped in a
            scikit-learn <code>Pipeline</code>, so transforms are refitted inside each fold.{" "}
            <strong>Temporal leakage</strong> is prevented by <code>temporal_split()</code>,
            which never shuffles and supports an embargo band at the boundary for
            rolling-window features. <strong>Group leakage</strong> is prevented by{" "}
            <code>grouped_split()</code>, which keeps entities whole. Every split returns a
            serialisable report that ships in the artifact for auditing.
          </Rule>

          <Rule n="03" title="Every metric next to its no-skill baseline" code="lib/dsx/metrics.py">
            A score without a reference point is not a result.{" "}
            <code>classification_report()</code> always emits prevalence and the
            always-negative accuracy beside the model's own, so accuracy cannot be quoted as
            skill on an imbalanced problem. <code>regression_report()</code> takes an explicit
            baseline and reports skill score against it.{" "}
            <code>interval_coverage()</code> reports realised coverage next to the nominal
            level, which is the only way a prediction interval can be falsified.
          </Rule>

          <Rule n="04" title="Decisions recorded with their rejected alternatives" code="lib/dsx/crispdm.py">
            CRISP-DM is usually applied as six headings in a document, which is not
            verifiable. Here each phase is a structured record carrying the decisions taken,
            the evidence produced, and — the field that does the real work — the alternatives
            rejected and why. A write-up listing only what was chosen is unfalsifiable; one
            that says "accuracy was not used because prevalence is 0.0017" can be argued with.
          </Rule>

          <Rule n="05" title="No hand-typed numbers anywhere" code="lib/dsx/artifacts.py">
            Every figure in every README, paper and dashboard is written by a pipeline into
            JSON and read back. Each artifact carries a <code>_run</code> block recording the
            git commit, library versions, random seed and wall-clock duration that produced
            it. If a claim in prose disagrees with its artifact, the artifact is correct and
            the prose is a bug.
          </Rule>
        </div>
      </Section>

      <Section title="What was deliberately not done">
        <div className="grid grid-2">
          <Callout kind="warn" title="No SMOTE on the fraud data">
            Interpolating between 492 positives in 28-dimensional PCA space places synthetic
            points in regions where no real fraud has ever been observed, and the model then
            learns a boundary around fabricated data. Class weighting achieves the same
            rebalancing without inventing observations.
          </Callout>
          <Callout kind="warn" title="No accuracy reported as a headline">
            On the fraud data a model that never fires scores 99.87%. Accuracy appears only
            beside that baseline, never alone.
          </Callout>
          <Callout kind="warn" title="No target invented where the data lacks one">
            The NYC TLC release has no trip duration column. Rather than synthesise one — as
            the reference portfolio this work responds to does — the target became hourly
            demand, which the data actually supports.
          </Callout>
          <Callout kind="warn" title="No k chosen for interpretability alone">
            Segmentation k was selected by bootstrap reproducibility, not by picking four
            because four personas are easy to name. Where the validity indices disagreed with
            the stability test, the disagreement is reported rather than resolved silently.
          </Callout>
        </div>
      </Section>

      <Section title="Negative results, kept">
        <p className="prose dim">
          A portfolio in which everything worked is not evidence of skill; it is evidence of
          selective reporting. These are the results that did not go the way they were
          supposed to, each reported on its own project page with its cause.
        </p>
        <div className="stack">
          <Negative
            claim="A conformal prediction interval that under-covered"
            where="Project 01"
            detail="Split-conformal intervals held at 90% and 95% but realised only 71.2% coverage at the nominal 80% level. Demand grew 82.9% across the window, so calibration and test residuals are not exchangeable and the tightest band cannot absorb the drift."
          />
          <Negative
            claim="Clustering algorithms that only half-agreed"
            where="Project 02"
            detail="K-means agrees with the Gaussian mixture at ARI 0.49 and with Ward linkage at 0.50. Roughly half the partition structure is imposed by the spherical assumption rather than present in the data, so the segments are one defensible partition among several."
          />
          <Negative
            claim="A model that lost to a zero-parameter baseline"
            where="Project 05"
            detail="On the sunspot series no learned forecaster beat seasonal naive. The ~11-year cycle is not calendar-anchored, so every seasonal method mis-specifies it."
          />
          <Negative
            claim="Standard advice that made a model 82× worse"
            where="Project 04"
            detail="Setting scale_pos_weight to the negative/positive ratio — the conventional recommendation for imbalanced boosting — collapsed LightGBM PR-AUC from 0.736 to 0.009. All four variants are published."
          />
          <Negative
            claim="A bias-variance curve that refused to be a U"
            where="Project 08"
            detail="Variance accounts for only 3.6% of test error even at degree 15, and training error is not monotone under bootstrap resampling. Both departures from the textbook figure are detected in code and stated."
          />
        </div>
      </Section>

      <Section title="Reproducing this">
        <div className="card">
          <pre className="sample">
{`git clone https://github.com/Pranjal101Shrivastava/Projects
cd Projects
pip install -r requirements.txt
export PYTHONPATH=lib

# Run any pipeline; each writes its own artifacts/ directory.
python3 projects/01_nyc_mobility/pipeline/build.py
python3 projects/04_fraud_detection/pipeline/build.py

# Verify no leakage across every pipeline in the repository.
python3 tools/audit.py

# Rebuild this site from the artifacts.
python3 tools/sync_artifacts.py
cd web && npm install && npm run build`}
          </pre>
          <p className="small dim" style={{ marginTop: 12, marginBottom: 0 }}>
            Every pipeline is seeded and each artifact records the commit that produced it, so
            a rerun at the same commit reproduces the same numbers. Datasets download on first
            use and are verified against their recorded hash thereafter.
          </p>
        </div>
      </Section>
    </div>
  );
}

function Rule({ n, title, code, children }: { n: string; title: string; code: string; children: React.ReactNode }) {
  return (
    <div className="card">
      <div className="row" style={{ marginBottom: 8 }}>
        <span className="project-num">{n}</span>
        <strong style={{ fontSize: "1.02rem" }}>{title}</strong>
        <span className="spacer" />
        <code className="tiny">{code}</code>
      </div>
      <p className="small dim" style={{ margin: 0 }}>{children}</p>
    </div>
  );
}

function Negative({ claim, where, detail }: { claim: string; where: string; detail: string }) {
  return (
    <div className="card" style={{ borderLeft: "3px solid var(--warn)" }}>
      <div className="row">
        <strong>{claim}</strong>
        <span className="spacer" />
        <span className="badge badge-neutral">{where}</span>
      </div>
      <p className="small dim" style={{ margin: "6px 0 0" }}>{detail}</p>
    </div>
  );
}
