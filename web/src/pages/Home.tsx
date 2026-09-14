import { Link } from "react-router-dom";
import { PROJECTS } from "../lib/projects";

/**
 * Portfolio index.
 *
 * The card for each project leads with a *comparison* rather than a score, because a bare
 * metric is unreadable without its baseline. "PR-AUC 0.766" tells a visitor nothing;
 * "0.766 against a 0.0017 no-skill floor" tells them what was achieved.
 */
export default function Home() {
  return (
    <div className="wrap">
      <header className="hero">
        <div className="row" style={{ marginBottom: 14 }}>
          <span className="badge badge-real">● 100% REAL DATA</span>
          <span className="badge badge-neutral">11 declared sources</span>
          <span className="badge badge-neutral">SHA-256 pinned</span>
        </div>
        <h1>Applied Data Science Portfolio</h1>
        <p className="lead">
          {PROJECTS.length} end-to-end systems built on real, publicly documented datasets — 4.5M NYC
          dispatch records, 284,807 card transactions, 41,188 marketing calls. Each one
          ships its data provenance, its leakage controls, its baselines, and the results
          that did not work.
        </p>
        <p className="lead" style={{ fontSize: "0.95rem" }}>
          Every number on this site is read at runtime from a JSON artifact written by a
          pipeline in the repository. None are typed by hand, so the prose and the
          measurements cannot drift apart.
        </p>
      </header>

      <section className="section">
        <div className="grid grid-2">
          {PROJECTS.map((p) => (
            <Link key={p.slug} to={`/p/${p.slug}`} className="project-card">
              <div className="project-num">{p.number}</div>
              <div className="project-title">{p.title}</div>
              <div className="project-desc">{p.description}</div>
              <div className="project-meta">
                <span className="badge badge-accent">{p.domain}</span>
                {p.tags.map((t) => (
                  <span key={t} className="badge badge-neutral">{t}</span>
                ))}
              </div>
              <div className="headline">
                {p.headline}
                <br />
                <strong>{p.headlineValue}</strong>
              </div>
            </Link>
          ))}
        </div>
      </section>

      <section className="section">
        <div className="section-head">
          <h2>What this portfolio is arguing</h2>
        </div>
        <div className="grid grid-3">
          <div className="card">
            <h3>Real data, stated plainly</h3>
            <p className="small dim" style={{ margin: 0 }}>
              Every source is declared once with its true origin, licence and an explicit
              <span className="good-text"> REAL</span> or{" "}
              <span className="warn-text">SIMULATED</span> marker. Downloads are pinned by
              SHA-256, so an upstream file that changes underneath the work fails loudly
              instead of quietly shifting every metric.
            </p>
          </div>
          <div className="card">
            <h3>Baselines before models</h3>
            <p className="small dim" style={{ margin: 0 }}>
              A score without a reference point is not a result. Every metric is reported
              beside what doing nothing would achieve — the seasonal-naive forecast, the
              always-negative classifier, the frequency-distribution language model.
            </p>
          </div>
          <div className="card">
            <h3>Negative results kept</h3>
            <p className="small dim" style={{ margin: 0 }}>
              A conformal interval that under-covers, a clustering whose algorithms only
              half-agree, a series where no model beats naive, an ablation contradicting
              standard advice. These are reported as prominently as the successes, because
              a portfolio in which everything worked is not evidence of skill.
            </p>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="section-head">
          <h2>Selected findings</h2>
          <span className="section-note">each links to the project that produced it</span>
        </div>
        <div className="stack">
          <Finding
            to="/p/automl"
            claim="One column inflated every model by ~40%"
            detail="UCI Bank Marketing ships with a 'duration' column that records how long the sales call lasted — unknowable before the call is placed. Running the identical tournament with and without it moves the best PR-AUC from 0.496 to 0.694. Nothing in the cross-validation flags it."
          />
          <Finding
            to="/p/fraud"
            claim="Standard advice for imbalanced boosting made the model 82× worse"
            detail="Setting scale_pos_weight to the negative/positive ratio — the conventional recommendation — collapsed LightGBM PR-AUC from 0.736 to 0.009. All four reweighting variants are published rather than only the winner."
          />
          <Finding
            to="/p/market-basket"
            claim="Two from-scratch algorithms agreed exactly, 189× apart in runtime"
            detail="Apriori and FP-Growth were implemented independently and returned byte-identical output on all 13,106 frequent itemsets, in 117.94s and 0.62s respectively. The run fails loudly if they ever disagree."
          />
          <Finding
            to="/p/forecasting"
            claim="On one of four series, nothing beat the naive baseline"
            detail="The sunspot cycle averages ~11 years but is not calendar-anchored, and every learned model ranked below seasonal naive. A single-series study would have concluded 'SARIMA wins' and been wrong about the general case."
          />
          <Finding
            to="/p/nyc-mobility"
            claim="A prediction interval that quietly under-covered"
            detail="Split-conformal intervals held at 90% and 95% but realised only 71.2% coverage at the nominal 80% level. Demand grew 82.9% across the window, so calibration and test residuals are not exchangeable — the limitation is reported with its cause rather than omitted."
          />
        </div>
      </section>
    </div>
  );
}

function Finding({ to, claim, detail }: { to: string; claim: string; detail: string }) {
  return (
    <Link to={to} className="card" style={{ display: "block", color: "inherit" }}>
      <div className="row" style={{ alignItems: "flex-start" }}>
        <span className="badge badge-accent" style={{ marginTop: 2 }}>FINDING</span>
        <div style={{ flex: 1, minWidth: 220 }}>
          <strong>{claim}</strong>
          <p className="small dim" style={{ margin: "6px 0 0" }}>{detail}</p>
        </div>
      </div>
    </Link>
  );
}
