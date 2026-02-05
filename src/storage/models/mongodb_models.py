"""
MongoDB Pydantic models
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MongoQuery(BaseModel):
    database: str = Field(default="gitquery", description="Database name")
    collection: str = Field(..., description="Collection name")
    filter: Dict[str, Any] = Field(default={}, description="Query filter")
    projection: Optional[Dict[str, int]] = Field(
        default=None, description="Fields to return"
    )
    limit: int = Field(default=100, le=1000, description="Maximum number of documents")
    skip: int = Field(default=0, description="Number of documents to skip")
    sort: Optional[Dict[str, int]] = Field(default=None, description="Sort criteria")


class MongoInsert(BaseModel):
    database: str = Field(default="gitquery", description="Database name")
    collection: str = Field(..., description="Collection name")
    documents: List[Dict[str, Any]] = Field(..., description="Documents to insert")
