import pytest
from datetime import datetime
from src.recommender.models import ModelMetadata
from src.recommender.services.registry_service import ModelRegistryService
from unittest.mock import MagicMock, AsyncMock

@pytest.fixture
def registry_service():
    return ModelRegistryService()

@pytest.fixture
def sample_metadata():
    return ModelMetadata(
        model_id="test_model_123",
        model_type="embedding",
        variant="default",
        version="1.0.0",
        path="embedding/test_v1",
        metrics={"accuracy": 0.95},
        trained_at=datetime.utcnow(),
        is_active=False,
        status="candidate"
    )

@pytest.mark.asyncio
async def test_register_model(registry_service, sample_metadata, mocker):
    # Mock db_manager
    mock_db = mocker.patch("src.recommender.database.db_manager.save_model_metadata", new_callable=AsyncMock)
    
    model_id = await registry_service.register_model(sample_metadata)
    
    assert model_id == "test_model_123"
    mock_db.assert_called_once_with(sample_metadata)

@pytest.mark.asyncio
async def test_get_active_model(registry_service, sample_metadata, mocker):
    # Mock db_manager
    mock_get = mocker.patch("src.recommender.database.db_manager.get_active_model", new_callable=AsyncMock)
    mock_get.return_value = sample_metadata
    
    active_model = await registry_service.get_active_model("embedding", "default")
    
    assert active_model.model_id == "test_model_123"
    mock_get.assert_called_once_with("embedding", "default")

@pytest.mark.asyncio
async def test_promote_model(registry_service, sample_metadata, mocker):
    # Mock db_manager.db
    mock_db_instance = MagicMock()
    mocker.patch("src.recommender.database.db_manager.db", mock_db_instance)
    
    # Mock find_one
    mock_db_instance.__getitem__.return_value.find_one = AsyncMock(return_value=sample_metadata.model_dump())
    # Mock update_many and update_one
    mock_db_instance.__getitem__.return_value.update_many = AsyncMock()
    mock_db_instance.__getitem__.return_value.update_one = AsyncMock()
    
    success = await registry_service.promote_model("test_model_123")
    
    assert success is True
    # Verify update_many was called to deactivate others
    mock_db_instance.__getitem__.return_value.update_many.assert_called_once()
    # Verify update_one was called to activate this one
    mock_db_instance.__getitem__.return_value.update_one.assert_called_once()
