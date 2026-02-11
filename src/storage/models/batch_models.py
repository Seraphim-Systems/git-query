"""Compatibility shim: re-export Batch models from `src.db.models`."""

from db.models import BatchInsert

__all__ = ["BatchInsert"]
