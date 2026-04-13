"""
Batch operation Pydantic models (migrated from src/storage/models).
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from .mongodb_models import MongoInsert
from .qdrant_models import QdrantInsert


class BatchInsert(BaseModel):
    mongodb_data: Optional[List[MongoInsert]] = None
    qdrant_data: Optional[List[QdrantInsert]] = None
    redis_data: Optional[List[Dict[str, Any]]] = None
