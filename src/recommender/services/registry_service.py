"""Service for model registration, discovery, and lifecycle management."""

import logging
from typing import Optional, List, Literal, Any, Dict
from datetime import datetime

from ..models import ModelMetadata
from ..database import db_manager
from ..config import settings

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
        """
        Register a new model in the registry.

        Args:
            metadata: Metadata for the new model

        Returns:
            The registered model_id
        """
        logger.info(f"Registering new model: {metadata.model_id} (type: {metadata.model_type})")
        
        # Save to database
        await db_manager.save_model_metadata(metadata)
        
        return metadata.model_id

    async def get_active_model(
        self, 
        model_type: Literal["embedding", "cross_encoder", "personalization"],
        variant: str = "default"
    ) -> Optional[ModelMetadata]:
        """
        Get the currently active model for a specific type and variant.

        Args:
            model_type: Type of model to retrieve
            variant: A/B test variant

        Returns:
            Active ModelMetadata or None if not found
        """
        return await db_manager.get_active_model(model_type, variant)

    async def promote_model(self, model_id: str) -> bool:
        """
        Promote a model to 'active' status.
        
        This is an atomic operation:
        1. Deactivate current active model of the same type/variant
        2. Activate the specified model

        Args:
            model_id: ID of the model to promote

        Returns:
            True if successful, False otherwise
        """
        try:
            # 1. Get the model metadata
            doc = await db_manager.db[settings.models_collection].find_one({"model_id": model_id})
            if not doc:
                logger.error(f"Model not found: {model_id}")
                return False
                
            model = ModelMetadata(**doc)
            
            # 2. Deactivate others of same type/variant
            await db_manager.db[settings.models_collection].update_many(
                {
                    "model_type": model.model_type, 
                    "variant": model.variant, 
                    "is_active": True
                },
                {"$set": {"is_active": False, "status": "archived"}}
            )
            
            # 3. Activate this one
            await db_manager.db[settings.models_collection].update_one(
                {"model_id": model_id},
                {"$set": {"is_active": True, "status": "active"}}
            )
            
            logger.info(f"Successfully promoted model {model_id} to active")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model {model_id}: {e}")
            return False

    async def list_models(
        self, 
        model_type: Optional[str] = None, 
        status: Optional[str] = None
    ) -> List[ModelMetadata]:
        """
        List models with optional filtering.

        Args:
            model_type: Filter by model type
            status: Filter by status

        Returns:
            List of matching ModelMetadata
        """
        query = {}
        if model_type:
            query["model_type"] = model_type
        if status:
            query["status"] = status
            
        cursor = db_manager.db[settings.models_collection].find(query).sort("trained_at", -1)
        
        models = []
        async for doc in cursor:
            doc.pop("_id", None)
            models.append(ModelMetadata(**doc))
            
        return models
