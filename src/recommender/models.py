"""Pydantic models for the recommendation system."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from enum import Enum


class InteractionType(str, Enum):
    """Types of user interactions."""

    CLICK = "click"
    SAVE = "save"
    DISMISS = "dismiss"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    VIEW = "view"


class RecommendationRequest(BaseModel):
    """Request for repository recommendations."""

    query: str = Field(..., description="User search query")
    user_id: Optional[str] = Field(None, description="User identifier for personalization")
    language: Optional[str] = Field(None, description="Filter by programming language")
    min_stars: Optional[int] = Field(None, description="Minimum number of stars")
    license: Optional[str] = Field(None, description="Filter by license type")
    max_age_days: Optional[int] = Field(None, description="Maximum age in days")
    activity_threshold: Optional[str] = Field(None, description="Minimum activity level")
    top_k: int = Field(10, description="Number of results to return", ge=1, le=50)
    enable_personalization: bool = Field(True, description="Apply personalization if available")
    variant: Optional[str] = Field(None, description="A/B test variant to use")


class RepositoryResult(BaseModel):
    """A single repository result."""

    repo_id: str
    name: str
    full_name: str
    description: Optional[str]
    language: Optional[str]
    stars: int
    forks: int
    url: str
    license: Optional[str]
    last_updated: Optional[datetime]
    score: float = Field(..., description="Relevance score")
    rank: int = Field(..., description="Position in results")
    explanation: Optional[Dict[str, Any]] = Field(
        None, description="Why this was recommended"
    )


class RecommendationResponse(BaseModel):
    """Response containing recommendations."""

    query: str
    user_id: Optional[str]
    results: List[RepositoryResult]
    total_candidates: int = Field(..., description="Total repos considered")
    processing_time_ms: float
    variant: str = Field("baseline", description="Which recommendation variant was used")
    personalized: bool = Field(False, description="Whether personalization was applied")
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UserInteraction(BaseModel):
    """User interaction event."""

    user_id: str
    query: str
    repo_id: str
    interaction_type: InteractionType
    position_in_results: Optional[int] = None
    variant: str = Field("baseline", description="Which variant was shown")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UserPreferences(BaseModel):
    """Learned user preferences."""

    user_id: str
    language_preferences: Dict[str, float] = Field(
        default_factory=dict, description="Language -> preference score"
    )
    topic_preferences: Dict[str, float] = Field(
        default_factory=dict, description="Topic -> preference score"
    )
    explicit_languages: List[str] = Field(
        default_factory=list,
        description="Languages the user explicitly said they use (e.g. during onboarding)",
    )
    total_interactions: int = 0
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvaluationMetrics(BaseModel):
    """Evaluation metrics for a model variant."""

    variant: str
    precision_at_k: Dict[int, float] = Field(
        default_factory=dict, description="Precision@K for different K values"
    )
    recall_at_k: Dict[int, float] = Field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = Field(default_factory=dict)
    mrr: float = Field(..., description="Mean Reciprocal Rank")
    click_through_rate: float = Field(..., description="CTR across all queries")
    avg_response_time_ms: float
    total_queries: int
    total_interactions: int
    evaluation_period_start: datetime
    evaluation_period_end: datetime
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ABTestConfig(BaseModel):
    """Configuration for an A/B test."""

    test_id: str
    name: str
    description: str
    variants: List[str] = Field(..., description="List of variant names")
    traffic_split: Dict[str, float] = Field(
        ..., description="Variant -> traffic percentage (0-1)"
    )
    start_date: datetime
    end_date: Optional[datetime] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ModelMetadata(BaseModel):
    """Metadata for a trained model."""

    model_id: str
    model_type: Literal["embedding", "cross_encoder", "personalization", "reranker"]
    variant: str = Field(..., description="Variant name for A/B testing")
    version: str
    path: str = Field(..., description="Relative path within the model volume")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(
        default_factory=dict, description="Training evaluation metrics"
    )
    trained_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = False
    status: Literal["candidate", "active", "archived"] = "candidate"
    notes: Optional[str] = None

