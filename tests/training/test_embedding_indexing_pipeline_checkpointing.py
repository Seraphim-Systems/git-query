import json
import os
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.recommender.training.pipelines.embedding_indexing_pipeline import _EmbeddingIndexer


@pytest.fixture
def indexer(tmp_path):
    """Provides an isolated _EmbeddingIndexer with a temporary directory."""
    return _EmbeddingIndexer(
        api_base_url="http://fake-api.local",
        api_key="fake-key",
        models_dir=str(tmp_path / "models"),
        data_cache_dir=str(tmp_path / "cache"),
    )


def make_repos(start: int, count: int):
    """Generate dummy repository dictionaries."""
    return [
        {
            "repo_id": f"repo_{i}",
            "name": f"name_{i}",
            "owner": "test",
            "description": f"desc_{i}",
            "language": "Python"
        }
        for i in range(start, start + count)
    ]


@pytest.fixture
def mocks():
    """Mock the heavy external dependencies and return the control primitives."""
    with patch("src.recommender.training.pipelines.embedding_indexing_pipeline.SentenceTransformer") as mock_st, \
         patch("src.recommender.training.pipelines.embedding_indexing_pipeline.EmbeddingUploader") as mock_eu, \
         patch.object(_EmbeddingIndexer, "_get_total_count") as mock_count, \
         patch.object(_EmbeddingIndexer, "_fetch_batch") as mock_fetch:

        # Let the total count be 10 for all tests unless overridden
        mock_count.return_value = 10

        # Model mock
        mock_model_instance = MagicMock()
        mock_model_instance.encode.side_effect = lambda texts, **kwargs: np.ones((len(texts), 384), dtype=np.float32)
        mock_st.return_value = mock_model_instance

        # Uploader mock
        mock_uploader_instance = MagicMock()
        mock_uploader_instance.upload_batch.return_value = {"status": "success"}
        mock_eu.return_value = mock_uploader_instance

        yield mock_fetch, mock_model_instance, mock_uploader_instance


def test_creates_checkpoint_after_chunk(indexer, mocks):
    mock_fetch, mock_model_instance, _ = mocks

    # We mock fetch_batch to progressively return data based on skip
    def side_effect(skip, limit, **kwargs):
        if skip >= 10:
            return []
        return make_repos(skip, min(limit, 10 - skip))
    mock_fetch.side_effect = side_effect

    # Make encode fail on the second call (chunk 2)
    call_count = [0]

    def mock_encode(texts, **kwargs):
        call_count[0] += 1
        if call_count[0] == 2:
            raise RuntimeError("Intentional failure on chunk 2")
        return np.ones((len(texts), 384), dtype=np.float32)
        
    mock_model_instance.encode.side_effect = mock_encode

    # Run shouldd raise the exception after processing chunk 1
    with pytest.raises(RuntimeError, match="Intentional failure on chunk 2"):
        indexer.run_chunked(chunk_size=5, fetch_batch_size=5, skip_if_no_new_data=False)

    # 1. Verify chunked_progress. json exists
    checkpoint_path = indexer.models_dir / "checkpoints" / "chunked_progress.json"
    assert checkpoint_path.exists(), "Checkpoint file was not created"

    with open(checkpoint_path) as f:
        ckpt = json.load(f)

    # Offset should be 5 since chunk 1 processed 5 items successfully
    assert ckpt["offset"] == 5
    assert ckpt["total_repos"] == 10
    assert "run_timestamp" in ckpt

    # 2. Verify mapping was explicitly saved for chunk 1
    mapping_path = indexer.models_dir / "metadata" / f"repo_mapping_{ckpt['run_timestamp']}.json"
    assert mapping_path.exists(), "Repository mapping was not persisted for the checkpoint"
    
    with open(mapping_path) as f:
        mapping = json.load(f)
    assert len(mapping) == 5, "Mapping should only contain the 5 repos from chunk 1"


def test_resumes_from_checkpoint(indexer, mocks):
    mock_fetch, _, _ = mocks

    # Setup a manual checkpoint step
    ts = "20260101_120000"
    checkpoint_dir = indexer.models_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "chunked_progress.json"

    with open(checkpoint_path, "w") as f:
        json.dump({
            "offset": 5,
            "run_timestamp": ts,
            "total_repos": 10
        }, f)

    # Setup matching maping
    mapping_dir = indexer.models_dir / "metadata"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = mapping_dir / f"repo_mapping_{ts}.json"
    
    # Simulating 5 repos already indexed
    mapping = [{"repo_id": f"repo_{i}", "index": i, "hash": ""} for i in range(5)]
    with open(mapping_path, "w") as f:
        json.dump(mapping, f)

    # Enforce that fetch begins at skip=5 (resuming correctly)
    def side_effect(skip, limit, **kwargs):
        assert skip >= 5, f"Expected fetch to resume at offset 5, but got {skip}"
        if skip >= 10:
            return []
        return make_repos(skip, min(limit, 10 - skip))
    mock_fetch.side_effect = side_effect

    # Run the pipeline
    indexer.run_chunked(chunk_size=5, fetch_batch_size=5, skip_if_no_new_data=False)

    # Verify fetch was called with skip=5
    assert mock_fetch.call_count > 0

    # The checkpoint should be cleanly unlinked after complete success
    assert not checkpoint_path.exists(), "Checkpoint should be removed after full success"

    # Verify final accumulated mapping includes all 10 entries
    latest_mapping_path = mapping_dir / "repo_mapping_latest.json"
    with open(latest_mapping_path) as f:
        final_mapping = json.load(f)
    assert len(final_mapping) == 10, "Final mapping should accumulate all 10 repositories"
    
    
def test_removes_checkpoint_on_success(indexer, mocks):
    mock_fetch, _, _ = mocks
    
    def side_effect(skip, limit, **kwargs):
        if skip >= 10:
            return []
        return make_repos(skip, min(limit, 10 - skip))
    mock_fetch.side_effect = side_effect

    # Ensure clean state
    checkpoint_path = indexer.models_dir / "checkpoints" / "chunked_progress.json"
    assert not checkpoint_path.exists()

    # Run successfully
    indexer.run_chunked(chunk_size=10, fetch_batch_size=10, skip_if_no_new_data=False)

    # Ensure checkpoint was created midway but ultimately cleaned up
    assert not checkpoint_path.exists()
    
    latest_mapping_path = indexer.models_dir / "metadata" / "repo_mapping_latest.json"
    assert latest_mapping_path.exists()


def test_handles_corrupted_checkpoint(indexer, mocks):
    mock_fetch, _, _ = mocks
    
    def side_effect(skip, limit, **kwargs):
        if skip >= 10:
            return []
        return make_repos(skip, min(limit, 10 - skip))
    mock_fetch.side_effect = side_effect

    # Create a completely corrupted checkpoint file
    checkpoint_dir = indexer.models_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "chunked_progress.json"

    with open(checkpoint_path, "w") as f:
        f.write("{invalid_json: true, trailing_comma: ],}")

    assert checkpoint_path.exists()

    # Run pipeline. It should catch the decoding error, unlink the corrupted file, and start fresh
    indexer.run_chunked(chunk_size=5, fetch_batch_size=5, skip_if_no_new_data=False)

    # File should be removed via the cleanup logic
    assert not checkpoint_path.exists()

    # We expect fetch to start from 0 because it started fresh
    skips = [call.kwargs.get("skip", call.args[0] if call.args else 0) for call in mock_fetch.call_args_list]
    assert 0 in skips, "Should have fetched from the very beginning"


def test_inchunk_deduplication(indexer, mocks):
    mock_fetch, mock_model_instance, _ = mocks

    # Simulate fetch returning 5 repos, where 2 are exact copies of the first and second
    def side_effect(skip, limit, **kwargs):
        if skip >= 5:
            return []
        repos = make_repos(0, 3)
        repos.append(repos[0].copy())
        repos.append(repos[1].copy())
        return repos
    mock_fetch.side_effect = side_effect

    indexer.run_chunked(chunk_size=10, fetch_batch_size=10, skip_if_no_new_data=False)

    # Encode should only be called once, for the 3 unique items
    assert mock_model_instance.encode.call_count == 1
    encoded_texts = mock_model_instance.encode.call_args[0][0]
    assert len(encoded_texts) == 3


def test_cross_run_deduplication(indexer, mocks):
    mock_fetch, mock_model_instance, _ = mocks

    # Simulate 3 repos already indexed
    mapping_dir = indexer.models_dir / "metadata"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    latest_mapping_path = mapping_dir / "repo_mapping_latest.json"
    
    mapping = [{"repo_id": f"repo_{i}", "index": i, "hash": ""} for i in range(3)]
    with open(latest_mapping_path, "w") as f:
        json.dump(mapping, f)

    # Simulate fetch returning 5 repos (repo_0 through repo_4)
    def side_effect(skip, limit, **kwargs):
        if skip >= 5:
            return []
        return make_repos(0, 5)
    mock_fetch.side_effect = side_effect

    # Run pipeline with cross-run deduplication enabled
    indexer.run_chunked(chunk_size=10, fetch_batch_size=10, skip_if_no_new_data=True)

    # It should skip repo_0, repo_1, and repo_2. Only repo_3 and repo_4 get encoded.
    assert mock_model_instance.encode.call_count == 1
    encoded_texts = mock_model_instance.encode.call_args[0][0]
    assert len(encoded_texts) == 2


def test_stable_id_uses_mongo_id(indexer):
    assert indexer._stable_id({"_id": "mongo123"}) == "mongo123"
    # Should shadow other keys
    assert indexer._stable_id({"_id": "mongo", "id": "123"}) == "mongo"


def test_stable_id_uses_direct_keys(indexer):
    assert indexer._stable_id({"nameWithOwner": "user/repo"}) == "user/repo"
    assert indexer._stable_id({"full_name": "user/repo"}) == "user/repo"
    assert indexer._stable_id({"repo_id": 42}) == "42"
    assert indexer._stable_id({"id": 100}) == "100"


def test_stable_id_uses_owner_and_name(indexer):
    assert indexer._stable_id({"owner": "alpha", "name": "beta"}) == "alpha/beta"
    assert indexer._stable_id({"owner_login": "gamma", "name": "delta"}) == "gamma/delta"


def test_stable_id_fallback_hash(indexer):
    import re
    doc = {"some_random_key": "data", "stars": 50}
    hash_id = indexer._stable_id(doc)
    
    assert len(hash_id) == 32
    assert re.match(r'^[a-f0-9]{32}$', hash_id)
    
    # Must be perfectly deterministic
    doc2 = {"stars": 50, "some_random_key": "data"} # keys out of order
    assert indexer._stable_id(doc2) == hash_id


def test_stable_id_empty_doc(indexer):
    assert indexer._stable_id(None) == ""
    assert indexer._stable_id({}) == ""

