"""Compatibility shim: re-export models from `src.db.models`.

Models have been centralized under `src/db/models`. This module keeps the
historical `src.storage.models` import path working by re-exporting the
classes from the canonical location.
"""

from db.models import (
    MongoQuery,
    MongoInsert,
    QdrantQuery,
    QdrantInsert,
    BatchInsert,
)

__all__ = [
    "MongoQuery",
    "MongoInsert",
    "QdrantQuery",
    "QdrantInsert",
    "BatchInsert",
]
