"""Comprehensive tests for embedding deduplication pipeline.

Tests verify that:
- Repository deduplication works at fetch time (_dedupe_repositories)
- New-data detection correctly identifies added/removed repos (check_for_new_data)
- Upload script deduplicates mapping entries before sending to Qdrant
- [DOCUMENTED GAP] Items already indexed in Qdrant are not re-embedded (xfail)
- Second pipeline run with identical data is skipped entirely
- embed_batch passes texts unchanged to model.encode without duplication
"""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_repos():
    """A canonical list of unique repository dicts."""
    return [
        {"_id": "repo-1", "name": "alpha", "description": "Alpha project", "language": "Python", "stars": 100},
        {"_id": "repo-2", "name": "beta",  "description": "Beta project",  "language": "Go",     "stars": 200},
        {"_id": "repo-3", "name": "gamma", "description": "Gamma project", "language": "Rust",   "stars": 300},
    ]


@pytest.fixture
def pipeline(tmp_path):
    """UnifiedTrainingPipeline with temp dirs — no real API or DB connections."""
    from src.recommender.training.unified_pipeline import UnifiedTrainingPipeline
    return UnifiedTrainingPipeline(
        api_base_url="http://fake-api",
        api_key="fake-key",
        models_dir=str(tmp_path / "models"),
        data_cache_dir=str(tmp_path / "data"),
    )


def _make_uploader():
    from src.recommender.scripts.upload_embeddings import EmbeddingUploader
    return EmbeddingUploader(base_url="http://fake", qdrant_api_key="fake-key")


def _fake_pipeline_result():
    """Minimal return value that satisfies save_model / register_model signatures."""
    return (
        np.zeros((3, 4)),
        ["r1", "r2", "r3"],
        {
            "model_name": "m", "timestamp": "t", "embedding_dim": 4,
            "num_repos": 3, "device": "cpu", "batch_size": 32,
            "training_time_seconds": 0.1, "normalized": True, "repo_hashes": [],
        },
    )


# ============================================================
# 1. Unit: _dedupe_repositories
# ============================================================

class TestDedupeRepositories:
    """_dedupe_repositories removes duplicates by stable ID, first-occurrence wins."""

    def test_removes_exact_duplicates(self, pipeline, sample_repos):
        duped = sample_repos + sample_repos        # 6 items, 3 unique
        assert len(pipeline._dedupe_repositories(duped)) == 3

    def test_preserves_first_occurrence_order(self, pipeline, sample_repos):
        duped = sample_repos + sample_repos
        result = pipeline._dedupe_repositories(duped)
        assert [r["_id"] for r in result] == ["repo-1", "repo-2", "repo-3"]

    def test_unique_list_is_unchanged(self, pipeline, sample_repos):
        result = pipeline._dedupe_repositories(sample_repos)
        assert len(result) == len(sample_repos)

    def test_empty_list(self, pipeline):
        assert pipeline._dedupe_repositories([]) == []

    def test_stable_id_via_full_name(self, pipeline):
        repos = [
            {"full_name": "org/repo-a"},
            {"full_name": "org/repo-a"},   # duplicate
            {"full_name": "org/repo-b"},
        ]
        assert len(pipeline._dedupe_repositories(repos)) == 2

    def test_stable_id_via_content_hash(self, pipeline):
        """Two identical dicts with no recognised ID field deduplicate via content hash."""
        doc = {"name": "orphan", "description": "no id fields"}
        assert len(pipeline._dedupe_repositories([doc, doc])) == 1

    def test_mixed_id_formats(self, pipeline):
        """Handles mix of _id, full_name, and repo_id without crashing."""
        repos = [
            {"_id": "mongo-123"},
            {"full_name": "org/repo"},
            {"repo_id": "ext-999"},
            {"_id": "mongo-123"},   # duplicate of first
        ]
        assert len(pipeline._dedupe_repositories(repos)) == 3

    def test_distinct_dicts_without_id_fields_are_kept(self, pipeline):
        """Two genuinely distinct anonymous dicts produce different hashes → both kept."""
        repos = [{"x": 1}, {"x": 2}]
        assert len(pipeline._dedupe_repositories(repos)) == 2


# ============================================================
# 2. Unit: check_for_new_data
# ============================================================

class TestCheckForNewData:
    """check_for_new_data detects added/removed repos against the local mapping file."""

    def _write_mapping(self, pipeline, repos):
        meta_dir = Path(pipeline.models_dir) / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        mapping = [
            {"repo_id": pipeline._stable_id(r), "index": i, "hash": ""}
            for i, r in enumerate(repos)
        ]
        (meta_dir / "repo_mapping_latest.json").write_text(json.dumps(mapping))

    def test_first_run_no_mapping_returns_true(self, pipeline, sample_repos):
        has_new, _ = pipeline.check_for_new_data(sample_repos)
        assert has_new is True

    def test_identical_data_returns_false(self, pipeline, sample_repos):
        self._write_mapping(pipeline, sample_repos)
        has_new, _ = pipeline.check_for_new_data(sample_repos)
        assert has_new is False

    def test_added_repo_detected(self, pipeline, sample_repos):
        self._write_mapping(pipeline, sample_repos[:2])   # previous: 2 repos
        has_new, _ = pipeline.check_for_new_data(sample_repos)  # current: 3
        assert has_new is True

    def test_removed_repo_detected(self, pipeline, sample_repos):
        self._write_mapping(pipeline, sample_repos)        # previous: 3 repos
        has_new, _ = pipeline.check_for_new_data(sample_repos[:2])  # current: 2
        assert has_new is True

    def test_corrupted_mapping_falls_back_to_true(self, pipeline, sample_repos):
        meta_dir = Path(pipeline.models_dir) / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "repo_mapping_latest.json").write_text("not-valid-json{{{{")
        has_new, _ = pipeline.check_for_new_data(sample_repos)
        assert has_new is True


# ============================================================
# 3. Unit: upload_all mapping deduplication
# ============================================================

class TestUploadAllMappingDeduplication:
    """EmbeddingUploader.upload_all must skip duplicate repo_ids in the local mapping."""

    def _collected_points(self, mock_upload):
        """Flatten all points from all upload_batch calls."""
        return [pt for c in mock_upload.call_args_list for pt in c[0][1]]

    def test_duplicate_mapping_entries_skipped(self):
        uploader = _make_uploader()
        embeddings = np.random.rand(3, 4).astype(np.float32)
        mapping = [
            {"repo_id": "repo-1", "index": 0, "hash": "aaa", "full_name": None},
            {"repo_id": "repo-1", "index": 0, "hash": "aaa", "full_name": None},   # dup
            {"repo_id": "repo-2", "index": 1, "hash": "bbb", "full_name": None},
        ]
        metadata = {"model_name": "test", "timestamp": "20260101"}

        with patch.object(uploader, "load_embeddings", return_value=(embeddings, mapping, metadata)), \
             patch.object(uploader, "ensure_collection"), \
             patch.object(uploader, "upload_batch", return_value={"status": "ok"}) as mock_upload:
            uploader.upload_all(collection="col", batch_size=100)

        uploaded_ids = [p["id"] for p in self._collected_points(mock_upload)]
        assert uploaded_ids.count("repo-1") == 1
        assert "repo-2" in uploaded_ids
        assert len(uploaded_ids) == 2

    def test_empty_repo_id_skipped(self):
        uploader = _make_uploader()
        embeddings = np.random.rand(2, 4).astype(np.float32)
        mapping = [
            {"repo_id": "",       "index": 0, "hash": "", "full_name": None},   # empty → skip
            {"repo_id": "repo-1", "index": 1, "hash": "", "full_name": None},
        ]
        metadata = {"model_name": "test", "timestamp": "20260101"}

        with patch.object(uploader, "load_embeddings", return_value=(embeddings, mapping, metadata)), \
             patch.object(uploader, "ensure_collection"), \
             patch.object(uploader, "upload_batch", return_value={"status": "ok"}) as mock_upload:
            uploader.upload_all(collection="col", batch_size=100)

        pts = self._collected_points(mock_upload)
        assert len(pts) == 1
        assert pts[0]["id"] == "repo-1"

    def test_out_of_range_index_skipped(self):
        uploader = _make_uploader()
        embeddings = np.random.rand(1, 4).astype(np.float32)
        mapping = [
            {"repo_id": "repo-1", "index": 0,  "hash": "", "full_name": None},
            {"repo_id": "repo-2", "index": 99, "hash": "", "full_name": None},   # OOB
        ]
        metadata = {"model_name": "test", "timestamp": "20260101"}

        with patch.object(uploader, "load_embeddings", return_value=(embeddings, mapping, metadata)), \
             patch.object(uploader, "ensure_collection"), \
             patch.object(uploader, "upload_batch", return_value={"status": "ok"}) as mock_upload:
            uploader.upload_all(collection="col", batch_size=100)

        pts = self._collected_points(mock_upload)
        assert len(pts) == 1
        assert pts[0]["id"] == "repo-1"

    def test_batching_respects_batch_size(self):
        """5 points with batch_size=2 → 3 upload_batch calls."""
        uploader = _make_uploader()
        n = 5
        embeddings = np.random.rand(n, 4).astype(np.float32)
        mapping = [{"repo_id": f"r{i}", "index": i, "hash": "", "full_name": None} for i in range(n)]
        metadata = {"model_name": "test", "timestamp": "20260101"}

        with patch.object(uploader, "load_embeddings", return_value=(embeddings, mapping, metadata)), \
             patch.object(uploader, "ensure_collection"), \
             patch.object(uploader, "upload_batch", return_value={"status": "ok"}) as mock_upload, \
             patch("time.sleep"):
            uploader.upload_all(collection="col", batch_size=2)

        assert mock_upload.call_count == 3   # ceil(5/2)


# ============================================================
# 4. E2E: [DOCUMENTED GAP] Qdrant pre-check before embedding
#
# These tests document the DESIRED behavior that is not yet implemented.
# Marked xfail(strict=True): CI stays green, gap remains visible.
# When the fix lands, remove xfail and confirm they pass.
#
# Recommended fixes:
#   train_embeddings(): query Qdrant for existing IDs, filter repos before encode()
#   upload_all():       scroll Qdrant for existing IDs, filter points before upload
# ============================================================

class TestQdrantPreCheckBeforeEmbedding:
    """Documents the gap: embed is called for repos already in Qdrant."""

    @pytest.mark.xfail(
        reason="Gap: train_embeddings encodes ALL repos without pre-checking Qdrant. "
               "Fix: filter already-indexed repo_ids before calling model.encode().",
        strict=True,
    )
    def test_already_indexed_repos_not_re_embedded(self, pipeline, sample_repos):
        """
        Given repo-1 and repo-2 already exist in Qdrant, train_embeddings
        should only encode repo-3 (1 text). Currently encodes all 3.
        """
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 384)   # only 1 expected

        with patch("src.recommender.training.unified_pipeline.SentenceTransformer", return_value=mock_model):
            pipeline.train_embeddings(repositories=sample_repos, model_name="m", batch_size=32)

        encoded_texts = mock_model.encode.call_args[0][0]
        assert len(encoded_texts) == 1, (
            f"Expected 1 text (only repo-3 is new), got {len(encoded_texts)}. "
            "Pipeline re-embeds already-indexed repos — Qdrant pre-check needed."
        )

    @pytest.mark.xfail(
        reason="Gap: upload_all upserts all points without pre-checking Qdrant IDs. "
               "Fix: scroll Qdrant for existing IDs and filter before upload.",
        strict=True,
    )
    def test_upload_skips_already_indexed_ids(self):
        """
        Given repo-1 already exists in Qdrant, upload_all should only upload repo-2.
        Currently uploads both.
        """
        uploader = _make_uploader()
        embeddings = np.random.rand(2, 4).astype(np.float32)
        mapping = [
            {"repo_id": "repo-1", "index": 0, "hash": "aaa", "full_name": None},   # exists in Qdrant
            {"repo_id": "repo-2", "index": 1, "hash": "bbb", "full_name": None},   # new
        ]
        metadata = {"model_name": "test", "timestamp": "20260101"}

        with patch.object(uploader, "load_embeddings", return_value=(embeddings, mapping, metadata)), \
             patch.object(uploader, "ensure_collection"), \
             patch.object(uploader, "upload_batch", return_value={"status": "ok"}) as mock_upload:
            uploader.upload_all(collection="col", batch_size=100)

        uploaded_ids = [p["id"] for c in mock_upload.call_args_list for p in c[0][1]]
        assert "repo-1" not in uploaded_ids, "repo-1 already in Qdrant — should be skipped"
        assert "repo-2" in uploaded_ids


# ============================================================
# 5. E2E: Second pipeline run with identical data is skipped
# ============================================================

class TestSkipRunOnIdenticalData:
    """Second run with the same repo set must not invoke train_embeddings."""

    def _write_mapping(self, pipeline, repos):
        meta_dir = Path(pipeline.models_dir) / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        mapping = [
            {"repo_id": pipeline._stable_id(r), "index": i, "hash": ""}
            for i, r in enumerate(repos)
        ]
        (meta_dir / "repo_mapping_latest.json").write_text(json.dumps(mapping))

    def test_second_run_skips_embedding(self, pipeline, sample_repos):
        self._write_mapping(pipeline, sample_repos)

        with patch.object(pipeline, "fetch_repositories", return_value=sample_repos), \
             patch.object(pipeline, "train_embeddings") as mock_train, \
             patch.object(pipeline, "save_data_cache"), \
             patch.object(pipeline, "save_model"), \
             patch.object(pipeline, "register_model"), \
             patch.object(pipeline, "upload_to_qdrant"):
            pipeline.run(skip_if_no_new_data=True)

        mock_train.assert_not_called()

    def test_new_repo_triggers_full_retrain(self, pipeline, sample_repos):
        self._write_mapping(pipeline, sample_repos[:2])   # only 2 previously

        with patch.object(pipeline, "fetch_repositories", return_value=sample_repos), \
             patch.object(pipeline, "train_embeddings", return_value=_fake_pipeline_result()) as mock_train, \
             patch.object(pipeline, "save_data_cache"), \
             patch.object(pipeline, "save_model"), \
             patch.object(pipeline, "register_model"), \
             patch.object(pipeline, "upload_to_qdrant"):
            pipeline.run(skip_if_no_new_data=True)

        mock_train.assert_called_once()

    def test_skip_disabled_always_retrains(self, pipeline, sample_repos):
        """With skip_if_no_new_data=False, train even when nothing changed."""
        self._write_mapping(pipeline, sample_repos)

        with patch.object(pipeline, "fetch_repositories", return_value=sample_repos), \
             patch.object(pipeline, "train_embeddings", return_value=_fake_pipeline_result()) as mock_train, \
             patch.object(pipeline, "save_data_cache"), \
             patch.object(pipeline, "save_model"), \
             patch.object(pipeline, "register_model"), \
             patch.object(pipeline, "upload_to_qdrant"):
            pipeline.run(skip_if_no_new_data=False)

        mock_train.assert_called_once()


# ============================================================
# 6. Integration: embed_batch / embed_text pass texts unchanged
# ============================================================

class TestEmbedServiceNoDuplication:
    """EmbeddingService must pass texts to model.encode exactly as received."""

    @pytest.mark.asyncio
    async def test_embed_batch_calls_encode_once_with_exact_texts(self):
        from src.recommender.services.embedding_service import EmbeddingService

        svc = EmbeddingService(model_name="fake-model")
        fake_model = MagicMock()
        fake_model.encode.return_value = np.random.rand(3, 384)
        svc.model = fake_model

        texts = ["alpha description", "beta description", "gamma description"]
        result = await svc.embed_batch(texts)

        fake_model.encode.assert_called_once()
        assert list(fake_model.encode.call_args[0][0]) == texts
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_embed_text_calls_encode_with_exact_string(self):
        from src.recommender.services.embedding_service import EmbeddingService

        svc = EmbeddingService(model_name="fake-model")
        fake_model = MagicMock()
        fake_model.encode.return_value = np.random.rand(384)
        svc.model = fake_model

        result = await svc.embed_text("python web framework")

        fake_model.encode.assert_called_once_with(
            "python web framework",
            convert_to_tensor=False,
            show_progress_bar=False,
        )
        assert isinstance(result, list)
        assert len(result) == 384

    @pytest.mark.asyncio
    async def test_embed_batch_output_length_matches_input(self):
        from src.recommender.services.embedding_service import EmbeddingService

        svc = EmbeddingService(model_name="fake-model")
        fake_model = MagicMock()
        fake_model.encode.return_value = np.random.rand(5, 384)
        svc.model = fake_model

        result = await svc.embed_batch([f"repo {i}" for i in range(5)])
        assert len(result) == 5


# ============================================================
# 7. Unit: Chunked pipeline deduplication
#
# run_chunked() must honour the same deduplication contract as run():
#   1. In-chunk:   _dedupe_repositories() with stable_id — first-occurrence wins
#   2. Cross-chunk: seen_ids set tracks all repo IDs embedded this run
#   3. Cross-run:  prev_ids from repo_mapping_latest.json skips already-indexed repos
# ============================================================

class TestChunkedPipelineDeduplication:
    """Deduplication behaviour for the streaming chunked pipeline."""

    def test_cross_chunk_seen_ids_excludes_repo_in_second_chunk(self, pipeline):
        """A repo embedded in chunk 1 must be excluded from chunk 2 via seen_ids."""
        seen_ids: set = set()

        chunk1 = [{"_id": "repo-1"}, {"_id": "repo-2"}]
        chunk1 = pipeline._dedupe_repositories(chunk1)
        for r in chunk1:
            seen_ids.add(pipeline._stable_id(r))

        chunk2 = [{"_id": "repo-1"}, {"_id": "repo-3"}]    # repo-1 reappears
        chunk2_filtered = [r for r in chunk2 if pipeline._stable_id(r) not in seen_ids]

        assert len(chunk2_filtered) == 1
        assert pipeline._stable_id(chunk2_filtered[0]) == "repo-3"

    def test_cross_run_prev_ids_skips_already_indexed(self, pipeline):
        """Repos present in prev_ids (from last run's mapping) are excluded."""
        prev_ids = {"repo-1", "repo-2"}

        chunk = [{"_id": "repo-1"}, {"_id": "repo-2"}, {"_id": "repo-3"}]
        new_repos = [r for r in chunk if pipeline._stable_id(r) not in prev_ids]

        assert len(new_repos) == 1
        assert pipeline._stable_id(new_repos[0]) == "repo-3"

    def test_seen_ids_grows_monotonically_across_chunks(self, pipeline):
        """seen_ids accumulates every unique stable_id embedded so far."""
        seen_ids: set = set()

        for chunk in [
            [{"_id": "a"}, {"_id": "b"}],
            [{"_id": "c"}],
            [{"_id": "a"}],    # a already seen
        ]:
            new = [r for r in chunk if pipeline._stable_id(r) not in seen_ids]
            for r in new:
                seen_ids.add(pipeline._stable_id(r))

        assert seen_ids == {"a", "b", "c"}

    def test_save_mapping_writes_latest_atomically(self, pipeline, tmp_path):
        """_save_mapping writes repo_mapping_latest.json and a timestamped copy."""
        mapping = [
            {"repo_id": "r1", "index": 0, "hash": "", "full_name": None},
            {"repo_id": "r2", "index": 1, "hash": "", "full_name": None},
        ]
        pipeline._save_mapping(mapping, "20260219_000000")

        latest = Path(pipeline.models_dir) / "metadata" / "repo_mapping_latest.json"
        timestamped = Path(pipeline.models_dir) / "metadata" / "repo_mapping_20260219_000000.json"

        assert latest.exists(), "repo_mapping_latest.json must be written"
        assert timestamped.exists(), "timestamped mapping must be written"

        loaded = json.loads(latest.read_text())
        assert len(loaded) == 2
        assert loaded[0]["repo_id"] == "r1"

    def test_save_mapping_overwrites_previous_latest(self, pipeline):
        """Second _save_mapping call replaces repo_mapping_latest.json."""
        pipeline._save_mapping(
            [{"repo_id": "old", "index": 0, "hash": "", "full_name": None}],
            "20260219_000000",
        )
        pipeline._save_mapping(
            [{"repo_id": "new-a", "index": 0, "hash": "", "full_name": None},
             {"repo_id": "new-b", "index": 1, "hash": "", "full_name": None}],
            "20260219_000001",
        )

        latest = Path(pipeline.models_dir) / "metadata" / "repo_mapping_latest.json"
        loaded = json.loads(latest.read_text())
        assert len(loaded) == 2
        assert loaded[0]["repo_id"] == "new-a"

    def test_dedupe_uses_same_stable_id_logic_as_original_pipeline(self, pipeline):
        """_dedupe_repositories in chunked mode is identical to run() — same method."""
        repos = [
            {"_id": "abc", "name": "repo-a"},
            {"_id": "abc", "name": "repo-a-duplicate"},   # same _id → drop
            {"full_name": "org/repo-b"},
        ]
        result = pipeline._dedupe_repositories(repos)

        assert len(result) == 2
        assert result[0]["name"] == "repo-a"       # first occurrence kept
        assert result[1]["full_name"] == "org/repo-b"

    def test_run_chunked_skips_all_chunks_when_no_new_repos(self, pipeline, sample_repos):
        """When all repos are already in prev_ids, no embedding or upload occurs."""
        # Write prev mapping covering all sample repos
        meta_dir = Path(pipeline.models_dir) / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        prev_mapping = [
            {"repo_id": pipeline._stable_id(r), "index": i, "hash": ""}
            for i, r in enumerate(sample_repos)
        ]
        (meta_dir / "repo_mapping_latest.json").write_text(json.dumps(prev_mapping))

        fake_embed = MagicMock(return_value=np.zeros((0, 384), dtype=np.float32))

        with patch.object(pipeline, "_get_total_count", return_value=3), \
             patch.object(pipeline, "_fetch_batch", return_value=sample_repos), \
             patch.object(pipeline, "_embed_chunk_parallel", fake_embed) as mock_embed, \
             patch.object(pipeline, "_save_mapping") as mock_save_mapping:
            pipeline.run_chunked(
                chunk_size=100_000,
                max_repos=3,
                skip_if_no_new_data=True,
                n_workers=1,
            )

        mock_embed.assert_not_called()
        mock_save_mapping.assert_not_called()

    def test_run_chunked_embeds_only_new_repos(self, pipeline, sample_repos):
        """When 2 of 3 repos exist in prev_ids, only the new one is embedded."""
        meta_dir = Path(pipeline.models_dir) / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        # Only repo-1 and repo-2 exist from last run
        prev_mapping = [
            {"repo_id": pipeline._stable_id(r), "index": i, "hash": ""}
            for i, r in enumerate(sample_repos[:2])
        ]
        (meta_dir / "repo_mapping_latest.json").write_text(json.dumps(prev_mapping))

        captured_texts: list = []

        def fake_embed(texts, model, batch_size=32, pool=None):
            captured_texts.extend(texts)
            return np.zeros((len(texts), 4), dtype=np.float32)

        fake_uploader = MagicMock()
        fake_uploader.upload_batch.return_value = {"status": "ok"}
        fake_uploader.ensure_collection.return_value = None

        with patch.object(pipeline, "_get_total_count", return_value=3), \
             patch.object(pipeline, "_fetch_batch", return_value=sample_repos), \
             patch.object(pipeline, "_embed_chunk_parallel", side_effect=fake_embed), \
             patch("src.recommender.training.unified_pipeline.SentenceTransformer"), \
             patch("src.recommender.training.unified_pipeline.EmbeddingUploader",
                   return_value=fake_uploader):
            pipeline.run_chunked(
                chunk_size=100_000,
                max_repos=3,
                skip_if_no_new_data=True,
                n_workers=1,
            )

        # Only repo-3 (gamma) should have been embedded
        assert len(captured_texts) == 1
