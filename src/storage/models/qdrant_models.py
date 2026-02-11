"""Compatibility shim: re-export Qdrant models from `src.db.models`."""

from db.models import QdrantQuery, QdrantInsert

__all__ = ["QdrantQuery", "QdrantInsert"]
