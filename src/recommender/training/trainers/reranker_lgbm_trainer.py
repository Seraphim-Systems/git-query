"""LightGBM LambdaRank trainer — async-friendly wrapper around LGBMRanker."""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..lgbm_ranker import LGBMRanker
from ...config import settings
from ...models import ModelMetadata
from ...services.registry_service import ModelRegistryService

logger = logging.getLogger(__name__)


class RerankerLGBMTrainer:
    """Async-friendly wrapper around LGBMRanker for pipeline integration.

    Bridges the pipeline interface (async train(training_data, variant))
    with LGBMRanker's synchronous API.
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        model_dir: Optional[str] = None,
    ):
        self.params = params
        self.model_dir = model_dir

    async def train(
        self,
        training_data: Dict[str, Any],
        variant: str,
        num_boost_rounds: int = 500,
        early_stopping_rounds: int = 50,
        tracker=None,
    ) -> Dict[str, Any]:
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
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_filename = f"lgbm_{variant}_{timestamp}.pkl"
        # Prefer explicit pipeline model_dir; fall back to env / static settings.
        resolved_model_dir = (
            self.model_dir or os.getenv("MODEL_PATH") or settings.model_path
        )
        model_path = os.path.join(resolved_model_dir, model_filename)
        await loop.run_in_executor(None, ranker.save, model_path)
        logger.info("LightGBM model saved to: %s", model_path)

        # Save reference snapshot for drift monitoring
        grouped_df = training_data.get("grouped_df")
        if grouped_df is not None:
            reference_path = os.path.join(resolved_model_dir, "metadata", "reference_data.parquet")
            os.makedirs(os.path.dirname(reference_path), exist_ok=True)
            await loop.run_in_executor(None, lambda: grouped_df.to_parquet(reference_path, index=False))
            logger.info("Reference snapshot saved to %s (%d rows)", reference_path, len(grouped_df))

        # Register in model registry
        metadata = ModelMetadata(
            model_id=f"lgbm_{variant}_{datetime.now(timezone.utc).timestamp()}",
            model_type="reranker",
            variant=variant,
            version="1.0.0",
            path=os.path.relpath(model_path, resolved_model_dir),
            hyperparameters=ranker.params,
            metrics={
                k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))
            },
            trained_at=datetime.now(timezone.utc),
            is_active=False,
            status="candidate",
        )

        registry = ModelRegistryService()
        model_id = await registry.register_model(metadata)

        result = dict(metrics)
        result["model_id"] = model_id
        return result
