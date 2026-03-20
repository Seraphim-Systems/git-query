"""Cross-encoder reranker training pipeline."""

import logging
from typing import Any, Dict, Optional

from .base_pipeline import BasePipeline

logger = logging.getLogger(__name__)


class RerankerCrossEncoderPipeline(BasePipeline):
    """End-to-end pipeline for training a cross-encoder reranker."""

    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        variant: str = "default",
        models_dir: str = "/app/models",
        max_repos: Optional[int] = None,
        epochs: int = 3,
        batch_size: int = 16,
    ):
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.variant = variant
        self.models_dir = models_dir
        self.max_repos = max_repos
        self.epochs = epochs
        self.batch_size = batch_size

    async def fetch(self) -> Dict[str, Any]:
        from ..data.mongo_data_fetcher import MongoDataFetcher

        fetcher = MongoDataFetcher(
            api_base_url=self.api_base_url,
            api_key=self.api_key,
            models_dir=self.models_dir,
        )
        return fetcher.fetch_training_pairs(max_repos=self.max_repos)

    async def train(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        from ..trainers.reranker_cross_encoder_trainer import RerankerCrossEncoderTrainer

        trainer = RerankerCrossEncoderTrainer()
        return await trainer.train(
            training_data=training_data,
            variant=self.variant,
            epochs=self.epochs,
            batch_size=self.batch_size,
        )

    async def evaluate(self, training_data: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluation is performed inside the cross-encoder trainer."""
        return metrics

    async def register(self, metrics: Dict[str, Any]) -> None:
        """Registration is handled inside RerankerCrossEncoderTrainer.train()."""
