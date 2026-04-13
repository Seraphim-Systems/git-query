"""Compatibility shim: re-export Batch models from `src.db.models`."""

from src.db.models import BatchInsert

__all__ = ["BatchInsert"]
