"""Shared data models"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class RepositoryBase(BaseModel):
    """Base repository model"""

    repo_id: str
    full_name: str
    name: str
    owner: str
    description: str
    language: Optional[str] = None
    stars: int = 0
    forks: int = 0
    watchers: int = 0


class RawRepository(RepositoryBase):
    """Raw repository data from scraper"""

    processing_status: Optional[str] = "pending"
    scraped_at: Optional[datetime] = None


class CleanedRepository(RepositoryBase):
    """Cleaned repository data"""

    topics: List[str] = Field(default_factory=list)
    is_fork: bool = False
    is_archived: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    search_text: str = ""
    cleaned_at: Optional[datetime] = None


class ProcessingStats(BaseModel):
    """Processing statistics"""

    total_records: int
    pending: int
    completed: int
    failed: int
    last_run: Optional[datetime] = None
