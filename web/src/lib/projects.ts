/**
 * Project registry.
 *
 * One entry per system: route, title, and the single headline figure worth remembering.
 * `headline` is deliberately phrased as a *comparison* rather than a bare score — "PR-AUC
 * 0.77" means nothing on its own, "0.77 against a 0.0017 no-skill floor" means something.
 */

export type ProjectMeta = {
  id: string;
  slug: string;
  number: string;
  title: string;
  description: string;
  domain: string;
  dataset: string;
  rows: string;
  tags: string[];
  headline: string;
  headlineValue: string;
};

export const PROJECTS: ProjectMeta[] = [
  {
    id: "01_nyc_mobility",
    slug: "nyc-mobility",
    number: "01",
    title: "NYC Ride-Hail Demand",
    description:
      "Hourly pickup demand per zone from 4.5M real TLC dispatch records, with conformal prediction intervals and a documented coverage failure.",
    domain: "Spatio-temporal regression",
    dataset: "NYC TLC FOIL response, Apr–Sep 2014",
    rows: "4,534,327 pickups",
    tags: ["LightGBM", "Conformal intervals", "Temporal split"],
    headline: "MAE vs seasonal-naive baseline",
    headlineValue: "15.65 vs 25.70 — 39.1% skill",
  },
  {
    id: "02_customer_segmentation",
    slug: "segmentation",
    number: "02",
    title: "Customer Segmentation",
    description:
      "Bank marketing segments validated by bootstrap stability and an outcome the clustering never saw — where geometry and reproducibility disagree.",
    domain: "Unsupervised clustering",
    dataset: "UCI Bank Marketing, 2008–2010",
    rows: "41,188 contacts",
    tags: ["K-means", "Bootstrap ARI", "External validation"],
    headline: "Conversion spread across segments",
    headlineValue: "4.15% → 63.83% (χ²=4786)",
  },
  {
    id: "03_market_basket",
    slug: "market-basket",
    number: "03",
    title: "Market Basket Mining",
    description:
      "Apriori and FP-Growth implemented from scratch, cross-validated against each other, with Benjamini-Hochberg FDR control over 52,285 candidate rules.",
    domain: "Association rule mining",
    dataset: "arules Groceries transactions",
    rows: "9,835 baskets",
    tags: ["Apriori", "FP-Growth", "FDR control"],
    headline: "FP-Growth speedup, identical output",
    headlineValue: "189× faster on 13,106 itemsets",
  },
  {
    id: "04_fraud_detection",
    slug: "fraud",
    number: "04",
    title: "Fraud Detection",
    description:
      "Card fraud at 0.17% prevalence, with a reweighting ablation that contradicts the standard advice and an explicit demonstration of why ROC misleads here.",
    domain: "Imbalanced classification",
    dataset: "ULB credit card transactions, 2013",
    rows: "284,807 transactions",
    tags: ["PR-AUC", "Cost curve", "Ablation"],
    headline: "PR-AUC against no-skill floor",
    headlineValue: "0.766 vs 0.0017 — 580× lift",
  },
  {
    id: "05_timeseries_forecasting",
    slug: "forecasting",
    number: "05",
    title: "Forecasting Tournament",
    description:
      "Six forecasters across four structurally different real series under rolling-origin backtests — including one series where nothing beats the naive baseline.",
    domain: "Time series",
    dataset: "Airline, temperature, drug sales, sunspots",
    rows: "605 observations across 4 series",
    tags: ["SARIMA", "Rolling origin", "MASE"],
    headline: "Series where naive wins outright",
    headlineValue: "1 of 4 — sunspots",
  },
  {
    id: "06_automl_tournament",
    slug: "automl",
    number: "06",
    title: "AutoML & the Leak",
    description:
      "An identical model tournament run twice — with and without a column that cannot exist at scoring time — to price one documented target leak.",
    domain: "AutoML / data leakage",
    dataset: "UCI Bank Marketing, 2008–2010",
    rows: "41,188 contacts",
    tags: ["Leakage", "HPO", "Stacking"],
    headline: "PR-AUC inflation from one column",
    headlineValue: "0.496 → 0.694 — +39.9%",
  },
  {
    id: "07_nano_transformer",
    slug: "transformer",
    number: "07",
    title: "Nano Transformer",
    description:
      "A character-level decoder written from tensor operations — RoPE, SwiGLU, pre-norm, weight tying — trained on CPU and evaluated against real baselines.",
    domain: "Deep learning",
    dataset: "Tiny Shakespeare",
    rows: "1,115,394 characters",
    tags: ["RoPE", "SwiGLU", "From scratch"],
    headline: "Held-out perplexity vs unigram",
    headlineValue: "see live artifact",
  },
  {
    id: "08_crispdm_academy",
    slug: "academy",
    number: "08",
    title: "CRISP-DM Academy",
    description:
      "Six interactive teaching modules computed from real data — including a bias-variance curve that refuses to be the textbook U, and a CLT that converges far slower than the rule of thumb.",
    domain: "Statistical education",
    dataset: "Titanic, ULB fraud, Melbourne temperature",
    rows: "5 real datasets",
    tags: ["Interactive", "Gradient check", "Quizzes"],
    headline: "Gradient check max relative error",
    headlineValue: "1.4 × 10⁻⁸",
  },
];

export const bySlug = (slug: string) => PROJECTS.find((p) => p.slug === slug);
