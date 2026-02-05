"""
Qdrant Pydantic models
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QdrantQuery(BaseModel):
    collection: str = Field(..., description="Collection name")
    vector: List[float] = Field(..., description="Query vector")
    limit: int = Field(default=10, le=100, description="Number of results")
    score_threshold: Optional[float] = Field(
        default=None, description="Minimum score threshold"
    )


class QdrantInsert(BaseModel):
    collection: str = Field(..., description="Collection name")
    points: List[Dict[str, Any]] = Field(
        ..., description="Points to insert (id, vector, payload)"
    )
