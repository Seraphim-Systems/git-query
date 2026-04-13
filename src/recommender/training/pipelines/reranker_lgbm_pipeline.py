"""LightGBM LambdaRank training pipeline."""

import logging
from typing import Any, Dict, Optional

from .base_pipeline import BasePipeline

logger = logging.getLogger(__name__)


class RerankerLGBMPipeline(BasePipeline):
    """End-to-end pipeline for training a LightGBM LambdaRank reranker."""

    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        variant: str = "default",
        models_dir: str = "/app/models",
        max_repos: Optional[int] = None,
        num_boost_rounds: int = 500,
        early_stopping_rounds: int = 50,
        top_n_queries: int = 100,
        candidates_per_query: int = 100,
        tracker=None,
    ):
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.variant = variant
        self.models_dir = models_dir
        self.max_repos = max_repos
        self.num_boost_rounds = num_boost_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.top_n_queries = top_n_queries
        self.candidates_per_query = candidates_per_query
        self.tracker = tracker

    async def fetch(self) -> Dict[str, Any]:
        from ..data.mongo_data_fetcher import MongoDataFetcher

        fetcher = MongoDataFetcher(
            api_base_url=self.api_base_url,
            api_key=self.api_key,
            models_dir=self.models_dir,
        )
        return fetcher.fetch_training_pairs(
            max_repos=self.max_repos,
            top_n_queries=self.top_n_queries,
            candidates_per_query=self.candidates_per_query,
        )

    async def train(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        from ..trainers.reranker_lgbm_trainer import RerankerLGBMTrainer

        trainer = RerankerLGBMTrainer(model_dir=self.models_dir)
        return await trainer.train(
            training_data=training_data,
            variant=self.variant,
            num_boost_rounds=self.num_boost_rounds,
            early_stopping_rounds=self.early_stopping_rounds,
            tracker=self.tracker,
        )

    async def evaluate(self, training_data: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """NDCG evaluation is performed inside LGBMRanker.train() with multi-seed eval."""
        return metrics

    async def register(self, metrics: Dict[str, Any]) -> None:
        """Registration is handled inside RerankerLGBMTrainer.train()."""
