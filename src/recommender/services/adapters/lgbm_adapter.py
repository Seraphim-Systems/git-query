"""LightGBM adapter for the reranker service."""

import logging
from typing import List

import joblib
import pandas as pd

from .base_adapter import BaseRerankerAdapter
from ...data.features import FeatureExtractor

logger = logging.getLogger(__name__)


class LGBMAdapter(BaseRerankerAdapter):
    """Wraps a LightGBM LambdaRank model for use as a reranker.

    Loaded via joblib from a .pkl file produced by LGBMRanker.save().
    Feature extraction is performed inline using FeatureExtractor.
    """

    def __init__(self, model_path: str):
        payload = joblib.load(model_path)
        # Support both raw model objects and payload dicts from LGBMRanker.save()
        if isinstance(payload, dict):
            self._model = payload["model"]
            self._feature_cols = payload.get("feature_cols")
        else:
            self._model = payload
            self._feature_cols = getattr(payload, "feature_cols", None)
        self._fe = FeatureExtractor()

    def score(self, query: str, candidates: list) -> List[float]:
        rows = []
        for c in candidates:
            rows.append(
                {
                    "name": getattr(c, "name", None),
                    "description": getattr(c, "description", None),
                    "stars": getattr(c, "stars", None),
                    "forks": getattr(c, "forks", None),
                    "language": getattr(c, "language", None),
                    "license": getattr(c, "license", None),
                    "topics": getattr(c, "topics", None),
                    "readme": getattr(c, "readme", None),
                    "updated_at": getattr(c, "updated_at", None),
                    "pushed_at": getattr(c, "pushed_at", None),
                }
            )
        df = pd.DataFrame(rows)
        X = self._fe.extract_all(df, query=query)
        if self._feature_cols is not None:
            try:
                X = X[self._feature_cols]
            except KeyError as e:
                logger.warning("Feature column filter failed, using all features: %s", e)
        return self._model.predict(X.values).tolist()
