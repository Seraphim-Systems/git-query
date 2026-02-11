"""Compatibility shim: re-export Mongo models from `src.db.models`."""

from db.models import MongoQuery, MongoInsert

__all__ = ["MongoQuery", "MongoInsert"]
