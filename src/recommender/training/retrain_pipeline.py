"""Full retraining orchestrator: embedding indexing + LightGBM reranker.

Run order:
1. EmbeddingIndexingPipeline  — fetches all repos, encodes with pretrained model,
                                uploads vectors to Qdrant
2. RerankerLGBMPipeline       — builds query groups, trains LambdaRank, registers model

Environment variables:
    API_BASE_URL                  Gateway URL for fetching repo data (required)
    APIKEY_MONGODB                API key for the gateway (required)
    EMBEDDING_MODEL               Pretrained model name (default: all-MiniLM-L6-v2)
    BATCH_SIZE                    Encoding batch size (default: 32)
    FETCH_BATCH_SIZE              API fetch page size (default: 500)
    MAX_REPOS                     Max repos to fetch (default: all)
    SKIP_IF_NO_NEW_DATA           Skip if no repo set change detected (default: true)
    USE_CHUNKED_PIPELINE          Use streaming chunked mode (default: true)
    CHUNK_SIZE                    Repos per chunk in chunked mode (default: 100000)
    N_WORKERS                     CPU workers for multi-process embedding (default: cpu_count)
    MODELS_DIR                    Model output directory (default: /app/models)
    DATA_CACHE_DIR                Data cache directory (default: /app/training_data)
    ENABLE_LGBM_TRAINING          Set to "false" to skip LightGBM step (default: true)
    LGBM_TOP_N_QUERIES            Number of query groups for LightGBM (default: 100)
    LGBM_CANDIDATES_PER_QUERY     Candidates per query group (default: 100)
    LGBM_NUM_BOOST_ROUNDS         LightGBM boosting rounds (default: 300)
    MLFLOW_TRACKING_URI           MLflow server URI (default: file:///app/mlruns)
    MLFLOW_EXPERIMENT_NAME        MLflow experiment name (default: git-query-retrain)
    RECOMMENDER_URL               Recommender service URL for post-training hooks (default: http://git-query-recommender:8095)
    PURGE_QDRANT_BEFORE_RETRAIN   Delete Qdrant collection before re-indexing (default: false)
    QDRANT_COLLECTION             Qdrant collection name to purge (default: repositories_embeddings)
    APIKEY_QDRANT                 Qdrant API key (default: APIKEY_MONGODB value)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure /app is on the path when running inside Docker
_app_root = Path(__file__).resolve().parents[2]
if str(_app_root) not in sys.path:
    sys.path.insert(0, str(_app_root))


def _post_non_fatal(url: str) -> bool:
    """Best-effort POST helper used for post-training model lifecycle hooks."""
    try:
        response = requests.post(url, timeout=10)
        response.raise_for_status()
        return True
    except requests.RequestException as exc:
        logger.warning("Post-training call failed for %s: %s", url, exc)
        return False


def _purge_qdrant_collection(api_url: str, api_key: str, collection: str) -> None:
    """Purge Qdrant collection through gateway API if requested."""
    endpoint = f"{api_url.rstrip('/')}/api/qdrant/collections/{collection}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.delete(endpoint, headers=headers, timeout=30)
        if resp.status_code in (200, 204, 404):
            logger.info(
                "Qdrant purge request completed for collection '%s' (status=%s)",
                collection,
                resp.status_code,
            )
            return
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Qdrant purge step failed for '%s': %s", collection, exc)


async def main() -> None:
    api_url = os.environ["API_BASE_URL"]
    api_key = os.environ["APIKEY_MONGODB"]
    recommender_url = os.getenv("RECOMMENDER_URL", "http://git-query-recommender:8095").rstrip("/")
    qdrant_api_key = os.getenv("APIKEY_QDRANT", api_key)
    models_dir = os.getenv("MODELS_DIR", "/app/models")
    # Keep config-driven trainers aligned with orchestrator model directory.
    os.environ["MODEL_PATH"] = models_dir
    max_repos_env = os.getenv("MAX_REPOS")
    max_repos = int(max_repos_env) if max_repos_env else None

    # Optional maintenance step: clear Qdrant collection before full rebuild.
    if os.getenv("PURGE_QDRANT_BEFORE_RETRAIN", "false").lower() == "true":
        qdrant_collection = os.getenv("QDRANT_COLLECTION", "repositories_embeddings")
        _purge_qdrant_collection(
            api_url=api_url,
            api_key=qdrant_api_key,
            collection=qdrant_collection,
        )

    # --- Step 1: Embedding indexing ---
    logger.info("=== Step 1: Embedding indexing ===")

    from training.pipelines.embedding_indexing_pipeline import EmbeddingIndexingPipeline

    n_workers_env = os.getenv("N_WORKERS")
    n_workers = int(n_workers_env) if n_workers_env else (os.cpu_count() or 1)

    await EmbeddingIndexingPipeline(
        api_base_url=api_url,
        api_key=api_key,
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        models_dir=models_dir,
        data_cache_dir=os.getenv("DATA_CACHE_DIR", "/app/training_data"),
        batch_size=int(os.getenv("BATCH_SIZE", "32")),
        fetch_batch_size=int(os.getenv("FETCH_BATCH_SIZE", "500")),
        max_repos=max_repos,
        skip_if_no_new_data=os.getenv("SKIP_IF_NO_NEW_DATA", "true").lower() == "true",
        use_chunked=os.getenv("USE_CHUNKED_PIPELINE", "true").lower() == "true",
        chunk_size=int(os.getenv("CHUNK_SIZE", "100000")),
        n_workers=n_workers,
    ).run()

    logger.info("Embedding indexing complete.")

    # --- Step 2: LightGBM reranker ---
    enable_lgbm = os.getenv("ENABLE_LGBM_TRAINING", "true").lower() != "false"
    if not enable_lgbm:
        logger.info("ENABLE_LGBM_TRAINING=false — skipping LightGBM step.")
        return

    logger.info("=== Step 2: LightGBM reranker ===")

    from mlops.mlflow_tracker import MLflowTracker
    from training.pipelines.reranker_lgbm_pipeline import RerankerLGBMPipeline

    tracker = MLflowTracker(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns"),
        experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "git-query-retrain"),
    )

    pipeline = RerankerLGBMPipeline(
        api_base_url=api_url,
        api_key=api_key,
        variant="default",
        models_dir=models_dir,
        max_repos=max_repos,
        num_boost_rounds=int(os.getenv("LGBM_NUM_BOOST_ROUNDS", "300")),
        top_n_queries=int(os.getenv("LGBM_TOP_N_QUERIES", "100")),
        candidates_per_query=int(os.getenv("LGBM_CANDIDATES_PER_QUERY", "100")),
        tracker=tracker,
    )
    from mlops.model_promoter import ModelPromoter

    with tracker.start_run(run_name="lgbm-retrain"):
        lgbm_result = await pipeline.run()

        model_id = lgbm_result.get("model_id") if isinstance(lgbm_result, dict) else None
        model_path = lgbm_result.get("model_path") if isinstance(lgbm_result, dict) else None

        if model_id:
            run_id = tracker.get_run_id()

            # Log training metrics from the main thread — LGBMRanker.train() runs in
            # a thread-pool executor where MLflow's thread-local run context is absent,
            # so metrics logged there are silently dropped. Re-logging here guarantees
            # they appear in the UI.
            candidate_metrics = {k: float(v) for k, v in lgbm_result.items() if isinstance(v, (int, float))}
            tracker.log_metrics(candidate_metrics)
            tracker.log_params({"model_id": model_id, "variant": "default"})

            # Log model artifact and register in MLflow Model Registry as Staging
            mlflow_version = None
            if run_id and model_path and Path(model_path).exists():
                artifact_subpath = "models"
                tracker.log_artifact(model_path, artifact_subpath)
                mlflow_version = tracker.register_model_version(
                    run_id=run_id,
                    model_name="git-query-lgbm-reranker",
                    artifact_path=f"{artifact_subpath}/{Path(model_path).name}",
                )

            # Champion/challenger: promote only if better than current production
            promoter = ModelPromoter(
                recommender_url=recommender_url,
                mlflow_tracker=tracker,
            )
            promoted = promoter.promote_if_better(
                candidate_model_id=model_id,
                candidate_metrics=candidate_metrics,
                candidate_mlflow_version=mlflow_version,
            )
            tracker.set_tag("promoted", str(promoted))
            tracker.set_tag("mlflow_version", str(mlflow_version) if mlflow_version else "none")
            if promoted:
                logger.info("Model %s promoted to production", model_id)
            else:
                logger.info("Model %s did not improve on production — kept as candidate", model_id)
        else:
            logger.warning("No model_id returned by RerankerLGBMPipeline; skipping promote/reload")

    logger.info("LightGBM training complete.")
    logger.info("Full retraining pipeline complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("\n⚠ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("\n❌ Training failed: %s", e, exc_info=True)
        sys.exit(1)
