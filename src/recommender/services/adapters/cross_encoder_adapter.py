"""CrossEncoder adapter for the reranker service."""

import logging
from typing import List

from sentence_transformers import CrossEncoder

from .base_adapter import BaseRerankerAdapter
from ...training.utils import prepare_repo_text

logger = logging.getLogger(__name__)


class CrossEncoderAdapter(BaseRerankerAdapter):
    """Wraps a sentence-transformers CrossEncoder model."""

    def __init__(self, model_path: str):
        logger.info("Loading CrossEncoder from: %s", model_path)
        self._model = CrossEncoder(model_path)

    def score(self, query: str, candidates: list) -> List[float]:
        pairs = [
            [
                query,
                prepare_repo_text({
                    "name": getattr(c, "name", None),
                    "description": getattr(c, "description", None),
                    "language": getattr(c, "language", None),
                }),
            ]
            for c in candidates
        ]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]
