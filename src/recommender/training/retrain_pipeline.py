"""Full retraining orchestrator: embeddings + LightGBM ranker with MLflow tracking.

Run order:
1. unified_pipeline  — fetches repo data, trains embeddings, uploads to Qdrant
2. lgbm_ranker       — builds query groups, trains LambdaRank, logs run to MLflow

Environment variables:
    API_BASE_URL              Gateway URL for fetching repo data
    APIKEY_MONGODB            API key for the gateway
    MLFLOW_TRACKING_URI       MLflow server URI (default: file:///app/mlruns)
    MLFLOW_EXPERIMENT_NAME    MLflow experiment name (default: git-query-retrain)
    ENABLE_LGBM_TRAINING      Set to "false" to skip LightGBM step (default: true)
    MAX_REPOS                 Max repos to fetch (optional)
    LGBM_TOP_N_QUERIES        Number of query groups for LightGBM (default: 100)
    LGBM_CANDIDATES_PER_QUERY Candidates per query group (default: 100)
    LGBM_NUM_BOOST_ROUNDS     LightGBM boosting rounds (default: 300)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure /app is on the path when running inside Docker
_app_root = Path(__file__).resolve().parents[2]
if str(_app_root) not in sys.path:
    sys.path.insert(0, str(_app_root))


def run_embedding_pipeline() -> None:
    """Run the unified embedding training pipeline."""
    logger.info("=== Step 1: Embedding pipeline ===")
    import runpy
    runpy.run_module("training.unified_pipeline", run_name="__main__", alter_sys=True)
    logger.info("Embedding pipeline complete.")


def run_lgbm_pipeline() -> None:
    """Fetch repo data and train the LightGBM ranker with MLflow tracking."""
    logger.info("=== Step 2: LightGBM ranker ===")

    import pandas as pd

    from training.lgbm_ranker import LGBMRanker, build_query_groups

    # -- load repo data --
    api_url = os.environ["API_BASE_URL"]
    api_key = os.environ["APIKEY_MONGODB"]
    max_repos = int(os.getenv("MAX_REPOS", "0")) or None

    # Import dataset loader (available via PYTHONPATH=/app in Docker)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.dataset import RepoDataset

    logger.info(f"Fetching repos from {api_url} (max={max_repos or 'all'})...")
    ds = RepoDataset.from_gateway(url=api_url, api_key=api_key, max_repos=max_repos)
    df = ds.to_dataframe()
    logger.info(f"Loaded {len(df)} repos.")

    if df.empty:
        logger.warning("No repos loaded — skipping LightGBM training.")
        return

    # -- dataset versioning --
    from mlops.dataset_versioner import DatasetVersioner

    versioner = DatasetVersioner()
    dataset_version = versioner.compute_version(df)
    logger.info(f"Dataset version: {dataset_version}")

    meta_path = Path("/app/models/metadata/dataset_version.json")
    versioner.save_version_metadata(dataset_version, df, meta_path)

    # -- build query groups --
    top_n = int(os.getenv("LGBM_TOP_N_QUERIES", "100"))
    candidates = int(os.getenv("LGBM_CANDIDATES_PER_QUERY", "100"))
    grouped = build_query_groups(df, top_n_queries=top_n, candidates_per_query=candidates)
    logger.info(f"Query groups: {grouped['query_id'].nunique()}, rows: {len(grouped)}")

    # -- MLflow tracker --
    from mlops.mlflow_tracker import MLflowTracker

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "git-query-retrain")
    tracker = MLflowTracker(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )

    # -- train --
    num_boost_rounds = int(os.getenv("LGBM_NUM_BOOST_ROUNDS", "300"))
    ranker = LGBMRanker()
    with tracker.start_run(run_name=f"lgbm-{dataset_version}"):
        metrics = ranker.train(
            grouped,
            tracker=tracker,
            dataset_version=dataset_version,
            num_boost_rounds=num_boost_rounds,
        )

    logger.info(f"Mean NDCG@10: {metrics['mean_ndcg_at_10']:.4f}")
    logger.info(f"Std  NDCG@10: {metrics['std_ndcg_at_10']:.4f}")

    # -- save model --
    model_path = Path("/app/models/lgbm_ranker_latest.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    ranker.save(model_path)
    ranker.save_registry_entry(model_path, Path("/app/models/metadata"))
    logger.info(f"Model saved to {model_path}")

    # -- save reference snapshot for drift monitoring --
    reference_path = Path("/app/models/metadata/reference_data.parquet")
    df.to_parquet(reference_path, index=False)
    logger.info(f"Reference snapshot saved to {reference_path} ({len(df)} rows)")


def main() -> None:
    enable_lgbm = os.getenv("ENABLE_LGBM_TRAINING", "true").lower() != "false"

    try:
        run_embedding_pipeline()
    except Exception as exc:
        logger.error(f"Embedding pipeline failed: {exc}")
        sys.exit(1)

    if enable_lgbm:
        try:
            run_lgbm_pipeline()
        except Exception as exc:
            logger.error(f"LightGBM pipeline failed: {exc}")
            sys.exit(1)
    else:
        logger.info("ENABLE_LGBM_TRAINING=false — skipping LightGBM step.")

    logger.info("Full retraining pipeline complete.")


if __name__ == "__main__":
    main()
