"""Reranker adapter package — Strategy pattern for hot-swappable reranker backends."""

from .base_adapter import BaseRerankerAdapter
from .lgbm_adapter import LGBMAdapter
from .cross_encoder_adapter import CrossEncoderAdapter
from .adapter_factory import AdapterFactory

__all__ = ["BaseRerankerAdapter", "LGBMAdapter", "CrossEncoderAdapter", "AdapterFactory"]
