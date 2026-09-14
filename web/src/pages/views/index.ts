/**
 * Per-project view registry.
 *
 * The project page shell is identical everywhere; only the Findings tab differs, and this
 * map is where that difference is declared. Keys are the pipeline project ids, so an
 * artifact directory and its view stay named the same thing.
 */

import type { ComponentType } from "react";
import NycMobility from "./NycMobility";
import Segmentation from "./Segmentation";
import MarketBasket from "./MarketBasket";
import Fraud from "./Fraud";
import Forecasting from "./Forecasting";
import AutoML from "./AutoML";
import Transformer from "./Transformer";
import Academy from "./Academy";
import SimilaritySearch from "./SimilaritySearch";
import Fairness from "./Fairness";
import DagEngine from "./DagEngine";
import Backtest from "./Backtest";

export const PROJECT_VIEWS: Record<string, ComponentType> = {
  "01_nyc_mobility": NycMobility,
  "02_customer_segmentation": Segmentation,
  "03_market_basket": MarketBasket,
  "04_fraud_detection": Fraud,
  "05_timeseries_forecasting": Forecasting,
  "06_automl_tournament": AutoML,
  "07_nano_transformer": Transformer,
  "08_crispdm_academy": Academy,
  "09_similarity_search": SimilaritySearch,
  "10_fairness_audit": Fairness,
  "11_pipeline_dag": DagEngine,
  "12_market_backtest": Backtest,
};
