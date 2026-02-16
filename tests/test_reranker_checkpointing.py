import pytest
from unittest.mock import MagicMock, patch, ANY, AsyncMock
import os
import shutil

# We need to mock settings before importing RerankerTrainer if it uses settings at module level,
# but it uses it inside methods or init, so patching where it is imported is enough.
# However, RerankerTrainer imports settings from ..config.

from src.recommender.training.reranker_trainer import RerankerTrainer

@pytest.mark.asyncio
async def test_reranker_checkpointing():
    # Mock settings
    with patch("src.recommender.training.reranker_trainer.settings") as mock_settings:
        mock_settings.model_path = "/tmp/models"
        mock_settings.checkpoint_path = "/tmp/models/checkpoints"
        mock_settings.cross_encoder_model_name = "test-model"

        # Mock CrossEncoder
        with patch("src.recommender.training.reranker_trainer.CrossEncoder") as MockCrossEncoder:
            mock_model = MagicMock()
            MockCrossEncoder.return_value = mock_model
            
            # Mock ModelRegistryService
            with patch("src.recommender.services.registry_service.ModelRegistryService") as MockRegistry:
                mock_registry = AsyncMock()
                MockRegistry.return_value = mock_registry
                
                # Mock os and shutil
                # We need to be careful not to mock os.path completely as it breaks logic
                # So we patch specific functions or let them be if we use real paths in tests (but we want to avoid disk IO)
                # But here we are using side_effect to use real logic for path manipulation
                
                with patch("src.recommender.training.reranker_trainer.os.makedirs") as mock_makedirs:
                    with patch("src.recommender.training.reranker_trainer.os.remove") as mock_remove:
                        with patch("src.recommender.training.reranker_trainer.os.path.isdir") as mock_isdir:
                            mock_isdir.return_value = True # Assume directories
                            
                            with patch("src.recommender.training.reranker_trainer.shutil.rmtree") as mock_rmtree:
                                with patch("src.recommender.training.reranker_trainer.glob.glob") as mock_glob:
                                    
                                    # Instantiate trainer
                                    trainer = RerankerTrainer(base_model="test-base", checkpoint_save_total_limit=2)
                                    
                                    # Create dummy data
                                    training_data = {
                                        "queries": ["q1"],
                                        "positive_repos": [{"name": "p1", "description": "d1", "language": "l1"}],
                                        "negative_repos": [{"name": "n1", "description": "d2", "language": "l2"}]
                                    }
                                    
                                    # Run train
                                    await trainer.train(training_data, variant="test_v1", epochs=2)
                                    
                                    # Verify fit called
                                    assert mock_model.fit.called
                                    args, kwargs = mock_model.fit.call_args
                                    assert "callback" in kwargs
                                    callback = kwargs["callback"]
                                    
                                    # Test callback logic
                                    # 1. Save checkpoint
                                    # callback(score, epoch, steps)
                                    # epoch is 0-indexed in callback
                                    callback(0.5, 0, 100) 
                                    
                                    # Expect save to be called with epoch_1
                                    # path should be .../epoch_1
                                    # Verify save called
                                    # The path construction depends on os.path.join
                                    # Since we didn't mock os.path.join, it uses real one.
                                    # But settings.checkpoint_path is /tmp/models/checkpoints (mocked)
                                    
                                    # We can inspect the call
                                    save_call_args = mock_model.save.call_args
                                    assert save_call_args is not None
                                    saved_path = save_call_args[0][0]
                                    assert "epoch_1" in saved_path
                                    assert "test_v1" in saved_path
                                    
                                    # 2. Pruning logic
                                    # Mock glob to return 3 checkpoints: epoch_1, epoch_2, epoch_3
                                    # We need to return sorted or unsorted list. glob returns unsorted.
                                    mock_glob.return_value = [
                                        "/tmp/models/checkpoints/test_v1/epoch_1",
                                        "/tmp/models/checkpoints/test_v1/epoch_3",
                                        "/tmp/models/checkpoints/test_v1/epoch_2"
                                    ]
                                    
                                    # Trigger another callback (epoch 2 -> epoch_3)
                                    callback(0.6, 2, 300)
                                    
                                    # Expect pruning of epoch_1 (oldest)
                                    # Limit is 2. We have 3.
                                    # epoch_1 is the oldest (1 < 2 < 3)
                                    
                                    mock_rmtree.assert_called_with("/tmp/models/checkpoints/test_v1/epoch_1")

