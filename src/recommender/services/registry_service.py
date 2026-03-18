"""Service for model registration, discovery, and lifecycle management."""

import logging
from typing import Optional, List, Literal

from ..models import ModelMetadata
from ..database import db_manager

logger = logging.getLogger(__name__)


class ModelRegistryService:
    """
    Registry service for managing ML model artifacts and metadata.

    Provides a unified interface for:
    - Registering new models after training
    - Promoting models to 'active' status
    - Discovering active models for inference
    - Archiving old models
    """

    async def register_model(self, metadata: ModelMetadata) -> str:
        """Register a new model in the registry."""
        logger.info(
            "Registering new model: %s (type: %s)", metadata.model_id, metadata.model_type
        )
        await db_manager.save_model_metadata(metadata)
        return metadata.model_id

    async def get_active_model(
        self,
        model_type: Literal["embedding", "cross_encoder", "personalization"],
        variant: str = "default",
    ) -> Optional[ModelMetadata]:
        """Get the currently active model for a specific type and variant."""
        return await db_manager.get_active_model(model_type, variant)

    async def promote_model(self, model_id: str) -> bool:
        """Promote a model to 'active' status.

        Steps:
        1. Look up the model to determine its type/variant.
        2. Archive all currently active models of that type/variant.
        3. Activate the specified model.

        NOTE: these two writes are not atomic. On failure between them,
        call promote_model again to recover (deactivate is idempotent).
        """
        try:
            model = await db_manager.get_model_by_id(model_id)
            if not model:
                logger.error("Model not found: %s", model_id)
                return False

            await db_manager.deactivate_models(model.model_type, model.variant)
            await db_manager.activate_model(model_id)

            logger.info("Successfully promoted model %s to active", model_id)
            return True

        except Exception as e:
            logger.error("Failed to promote model %s: %s", model_id, e, exc_info=True)
            return False

    async def list_models(
        self,
        model_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[ModelMetadata]:
        """List models with optional filtering."""
        return await db_manager.list_models_query(model_type, status)
