"""LightGBM LambdaRank trainer — async-friendly wrapper around LGBMRanker."""

import logging
import os
from datetime import UTC, datetime
from typing import Any

from ..lgbm_ranker import LGBMRanker

logger = logging.getLogger(__name__)


class RerankerLGBMTrainer:
    """Async-friendly wrapper around LGBMRanker for pipeline integration.

    Bridges the pipeline interface (async train(training_data, variant))
    with LGBMRanker's synchronous API.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        model_dir: str | None = None,
    ):
        self.params = params
        self.model_dir = model_dir

    async def train(
        self,
        training_data: dict[str, Any],
        variant: str,
        num_boost_rounds: int = 500,
        early_stopping_rounds: int = 50,
        tracker=None,
    ) -> dict[str, Any]:
        """Train LightGBM LambdaRank and register the model.

        Args:
            training_data: Must contain "grouped_df" key with the grouped
                DataFrame produced by build_query_groups().
            variant: Model variant name used in the registry entry.
            num_boost_rounds: LightGBM boosting rounds.
            early_stopping_rounds: Early-stopping patience.
            tracker: Optional MLflowTracker instance.

        Returns:
            Training metrics dict from LGBMRanker.train().
        """
        import asyncio

        grouped_df = training_data.get("grouped_df")
        if grouped_df is None:
            raise ValueError("training_data must contain 'grouped_df' key")

        dataset_version = training_data.get("dataset_version")

        logger.info("Training LightGBM LambdaRank for variant=%s", variant)

        ranker = LGBMRanker(params=self.params)

        # Run blocking training in thread pool to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        metrics = await loop.run_in_executor(
            None,
            lambda: ranker.train(
                grouped_df,
                tracker=tracker,
                dataset_version=dataset_version,
                num_boost_rounds=num_boost_rounds,
                early_stopping_rounds=early_stopping_rounds,
            ),
        )

        # Save model artifact
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        model_filename = f"lgbm_{variant}_{timestamp}.pkl"
        # Prefer explicit pipeline model_dir; fall back to env var; then /app/models.
        # settings.model_path is intentionally not used here — the recommender config
        # package is not available in the training Docker container (PYTHONPATH=/app).
        resolved_model_dir = self.model_dir or os.getenv("MODEL_PATH") or "/app/models"
        model_path = os.path.join(resolved_model_dir, model_filename)
        await loop.run_in_executor(None, ranker.save, model_path)
        logger.info("LightGBM model saved to: %s", model_path)

        # Save reference snapshot for drift monitoring (latest + versioned for rollback)
        grouped_df = training_data.get("grouped_df")
        if grouped_df is not None:
            reference_dir = os.path.join(resolved_model_dir, "metadata")
            os.makedirs(reference_dir, exist_ok=True)
            reference_path = os.path.join(reference_dir, "reference_data.parquet")
            reference_path_versioned = os.path.join(reference_dir, f"reference_data_{timestamp}.parquet")
            await loop.run_in_executor(None, lambda: grouped_df.to_parquet(reference_path, index=False))
            await loop.run_in_executor(None, lambda: grouped_df.to_parquet(reference_path_versioned, index=False))
            logger.info("Reference snapshot saved to %s (%d rows)", reference_path, len(grouped_df))

            # Save reference prediction scores for prediction drift monitoring.
            # Use the most frequent query_text in the training set as the representative
            # query — this mirrors how the recommender computes scores against a user
            # interaction profile derived from historical positively-interacted repos.
            try:
                import numpy as np

                rep_query = (
                    grouped_df["query_text"].value_counts().index[0]
                    if "query_text" in grouped_df.columns and len(grouped_df) > 0
                    else ""
                )
                ref_scores = ranker.predict(grouped_df, query_text=rep_query)
                scores_path = os.path.join(reference_dir, "reference_scores.npy")
                scores_path_versioned = os.path.join(reference_dir, f"reference_scores_{timestamp}.npy")
                await loop.run_in_executor(None, lambda: np.save(scores_path, ref_scores))
                await loop.run_in_executor(None, lambda: np.save(scores_path_versioned, ref_scores))
                logger.info(
                    "Reference scores saved to %s (%d rows, query=%r)",
                    scores_path,
                    len(ref_scores),
                    rep_query,
                )
            except Exception as e:
                logger.warning("Could not save reference scores for drift monitoring: %s", e)

        # Register in model registry (only available when running inside the full
        # recommender package — skipped gracefully in the training Docker container).
        model_id = f"lgbm_{variant}_{datetime.now(UTC).timestamp()}"
        try:
            from ...models import ModelMetadata
            from ...services.registry_service import ModelRegistryService

            metadata = ModelMetadata(
                model_id=model_id,
                model_type="reranker",
                variant=variant,
                version="1.0.0",
                path=os.path.relpath(model_path, resolved_model_dir),
                hyperparameters=ranker.params,
                metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                trained_at=datetime.now(UTC),
                is_active=False,
                status="candidate",
            )
            registry = ModelRegistryService()
            model_id = await registry.register_model(metadata)
        except ImportError:
            logger.info(
                "ModelRegistryService not available in this context — "
                "model_id set to %s, skipping registry registration",
                model_id,
            )

        result = dict(metrics)
        result["model_id"] = model_id
        result["model_path"] = model_path
        return result
