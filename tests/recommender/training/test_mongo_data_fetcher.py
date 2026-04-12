"""Unit tests for MongoDataFetcher — London School TDD (mock-first)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_fetcher(api_base_url="http://api.test", api_key="key", models_dir="/tmp/models"):
    from src.recommender.training.data.mongo_data_fetcher import MongoDataFetcher

    return MongoDataFetcher(api_base_url=api_base_url, api_key=api_key, models_dir=models_dir)


def _mock_post(documents=None, count=None):
    """Return a mock requests.post that yields a standard API response."""
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {
        "count": count if count is not None else len(documents or []),
        "documents": documents or [],
    }
    return resp


# ===========================================================================
# _stable_id
# ===========================================================================


class TestStableId:
    def test_uses_repo_id_field(self):
        fetcher = _make_fetcher()
        assert fetcher._stable_id({"repo_id": "foo/bar"}) == "foo/bar"

    def test_uses_full_name_field(self):
        fetcher = _make_fetcher()
        assert fetcher._stable_id({"full_name": "alice/myrepo"}) == "alice/myrepo"

    def test_falls_back_to_owner_name(self):
        fetcher = _make_fetcher()
        assert fetcher._stable_id({"owner": "alice", "name": "myrepo"}) == "alice/myrepo"

    def test_returns_empty_for_empty_doc(self):
        fetcher = _make_fetcher()
        assert fetcher._stable_id({}) == ""

    def test_uses_md5_fallback_for_unknown_doc(self):
        fetcher = _make_fetcher()
        sid = fetcher._stable_id({"unknown_field": "value"})
        assert len(sid) == 32  # MD5 hex digest


# ===========================================================================
# _dedupe_repositories
# ===========================================================================


class TestDedupeRepositories:
    def test_removes_duplicates_by_stable_id(self):
        fetcher = _make_fetcher()
        repos = [{"repo_id": "r1", "name": "a"}, {"repo_id": "r1", "name": "b"}, {"repo_id": "r2"}]
        result = fetcher._dedupe_repositories(repos)
        assert len(result) == 2

    def test_preserves_first_occurrence(self):
        fetcher = _make_fetcher()
        repos = [{"repo_id": "r1", "name": "first"}, {"repo_id": "r1", "name": "second"}]
        result = fetcher._dedupe_repositories(repos)
        assert result[0]["name"] == "first"

    def test_returns_all_unique(self):
        fetcher = _make_fetcher()
        repos = [{"repo_id": "r1"}, {"repo_id": "r2"}, {"repo_id": "r3"}]
        assert len(fetcher._dedupe_repositories(repos)) == 3

    def test_empty_input(self):
        fetcher = _make_fetcher()
        assert fetcher._dedupe_repositories([]) == []


# ===========================================================================
# check_for_new_data
# ===========================================================================


class TestCheckForNewData:
    def test_returns_true_when_no_mapping_file(self):
        fetcher = _make_fetcher(models_dir="/nonexistent/path")
        repos = [{"repo_id": "r1"}]
        has_new, _ = fetcher.check_for_new_data(repos)
        assert has_new is True

    def test_returns_repos_when_no_mapping_file(self):
        fetcher = _make_fetcher(models_dir="/nonexistent/path")
        repos = [{"repo_id": "r1"}]
        _, returned = fetcher.check_for_new_data(repos)
        assert returned == repos

    def test_returns_false_when_ids_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_dir = Path(tmpdir) / "metadata"
            meta_dir.mkdir()
            mapping = [{"repo_id": "r1"}, {"repo_id": "r2"}]
            (meta_dir / "repo_mapping_latest.json").write_text(json.dumps(mapping))

            fetcher = _make_fetcher(models_dir=tmpdir)
            repos = [{"repo_id": "r1"}, {"repo_id": "r2"}]
            has_new, _ = fetcher.check_for_new_data(repos)

        assert has_new is False

    def test_returns_true_when_new_repo_added(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_dir = Path(tmpdir) / "metadata"
            meta_dir.mkdir()
            mapping = [{"repo_id": "r1"}]
            (meta_dir / "repo_mapping_latest.json").write_text(json.dumps(mapping))

            fetcher = _make_fetcher(models_dir=tmpdir)
            repos = [{"repo_id": "r1"}, {"repo_id": "r2"}]
            has_new, _ = fetcher.check_for_new_data(repos)

        assert has_new is True

    def test_returns_true_when_mapping_file_is_corrupted(self):
        """Corrupt JSON in the mapping file triggers the fallback: treat as new data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_dir = Path(tmpdir) / "metadata"
            meta_dir.mkdir()
            (meta_dir / "repo_mapping_latest.json").write_text("not valid json {{{")

            fetcher = _make_fetcher(models_dir=tmpdir)
            repos = [{"repo_id": "r1"}]
            has_new, returned = fetcher.check_for_new_data(repos)

        assert has_new is True
        assert returned == repos


# ===========================================================================
# fetch_repositories
# ===========================================================================


class TestFetchRepositories:
    def test_returns_empty_when_count_is_zero(self):
        fetcher = _make_fetcher()
        with patch("src.recommender.training.data.mongo_data_fetcher.requests.post") as mock_post:
            mock_post.return_value = _mock_post(documents=[], count=0)
            result = fetcher.fetch_repositories()
        assert result == []

    def test_returns_list_of_dicts(self):
        fetcher = _make_fetcher()
        docs = [{"repo_id": "r1", "name": "foo"}, {"repo_id": "r2", "name": "bar"}]
        with patch("src.recommender.training.data.mongo_data_fetcher.requests.post") as mock_post:
            mock_post.return_value = _mock_post(documents=docs, count=2)
            result = fetcher.fetch_repositories(batch_size=10, max_repos=10)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_returns_empty_list_on_connection_error(self):
        """When the API is unreachable, _fetch_batch swallows the error and returns []."""
        import requests as req

        fetcher = _make_fetcher()
        # _get_total_count will also raise — patch it to return a known value
        with patch("src.recommender.training.data.mongo_data_fetcher.requests.post") as mock_post:
            mock_post.side_effect = req.exceptions.ConnectionError("unreachable")
            result = fetcher.fetch_repositories(batch_size=10, max_repos=10)

        assert result == []

    def test_paginates_when_count_exceeds_batch_size(self):
        """When total repos > batch_size, multiple _fetch_batch calls are made."""
        fetcher = _make_fetcher()

        def side_effect(url, **kwargs):
            body = kwargs.get("json", {})
            limit = body.get("limit", 0)
            skip = body.get("skip", 0)
            if limit == 1:
                # _get_total_count probe — return count=10
                return _mock_post(documents=[], count=10)
            # _fetch_batch call — return docs for the requested slice
            docs = [{"repo_id": f"r{skip + i}"} for i in range(limit)]
            return _mock_post(documents=docs, count=10)

        with patch("src.recommender.training.data.mongo_data_fetcher.requests.post", side_effect=side_effect):
            result = fetcher.fetch_repositories(batch_size=5, max_repos=10)

        assert len(result) == 10
        assert result[0]["repo_id"] == "r0"
        assert result[5]["repo_id"] == "r5"


# ===========================================================================
# fetch_training_pairs
# ===========================================================================


class TestFetchTrainingPairs:
    def test_returns_required_keys(self):
        fetcher = _make_fetcher()
        # Enough repos with distinct languages so build_query_groups works
        docs = [
            {"repo_id": f"py{i}", "name": f"pyrepo{i}", "language": "Python", "stars": i * 10}
            for i in range(10)
        ] + [
            {"repo_id": f"js{i}", "name": f"jsrepo{i}", "language": "JavaScript", "stars": i * 5}
            for i in range(10)
        ]
        with patch("src.recommender.training.data.mongo_data_fetcher.requests.post") as mock_post:
            mock_post.return_value = _mock_post(documents=docs, count=len(docs))
            result = fetcher.fetch_training_pairs(max_repos=100)

        assert "queries" in result
        assert "positive_repos" in result
        assert "negative_repos" in result
        assert "grouped_df" in result
        assert "dataset_version" in result

    def test_grouped_df_has_interaction_score_column(self):
        """grouped_df must always carry an interaction_score column."""
        fetcher = _make_fetcher()
        docs = [
            {"repo_id": f"py{i}", "name": f"pyrepo{i}", "language": "Python", "stars": i * 10}
            for i in range(10)
        ] + [
            {"repo_id": f"js{i}", "name": f"jsrepo{i}", "language": "JavaScript", "stars": i * 5}
            for i in range(10)
        ]
        with patch("src.recommender.training.data.mongo_data_fetcher.requests.post") as mock_post:
            mock_post.return_value = _mock_post(documents=docs, count=len(docs))
            result = fetcher.fetch_training_pairs(max_repos=100)

        assert "interaction_score" in result["grouped_df"].columns

    def test_interaction_score_zero_when_no_interactions(self):
        """When fetch_interactions returns {}, all interaction_score values are 0.0."""
        fetcher = _make_fetcher()
        docs = [
            {"repo_id": f"py{i}", "name": f"r{i}", "language": "Python", "stars": i * 10}
            for i in range(10)
        ] + [
            {"repo_id": f"js{i}", "name": f"r{i}", "language": "JavaScript", "stars": i * 5}
            for i in range(10)
        ]
        with patch("src.recommender.training.data.mongo_data_fetcher.requests.post") as mock_post:
            mock_post.return_value = _mock_post(documents=docs, count=len(docs))
            result = fetcher.fetch_training_pairs(max_repos=100)

        assert (result["grouped_df"]["interaction_score"] == 0.0).all()

    def test_interaction_scores_joined_correctly(self):
        """Repos with matching IDs get their weighted interaction score."""
        fetcher = _make_fetcher()
        docs = [
            {"repo_id": f"py{i}", "name": f"r{i}", "language": "Python", "stars": i * 10}
            for i in range(10)
        ] + [
            {"repo_id": f"js{i}", "name": f"r{i}", "language": "JavaScript", "stars": i * 5}
            for i in range(10)
        ]
        # py0 has a save (+3) and a click (+1) = 4.0
        interaction_docs = [
            {"repo_id": "py0", "interaction_type": "save"},
            {"repo_id": "py0", "interaction_type": "click"},
            {"repo_id": "py1", "interaction_type": "thumbs_down"},
        ]

        def side_effect(url, **kwargs):
            body = kwargs.get("json", {})
            if body.get("collection") == "user_interactions":
                return _mock_post(documents=interaction_docs)
            return _mock_post(documents=docs, count=len(docs))

        with patch("src.recommender.training.data.mongo_data_fetcher.requests.post", side_effect=side_effect):
            result = fetcher.fetch_training_pairs(max_repos=100)

        gdf = result["grouped_df"]
        py0_score = gdf.loc[gdf["repo_id"] == "py0", "interaction_score"]
        assert not py0_score.empty
        assert py0_score.iloc[0] == pytest.approx(4.0)


# ===========================================================================
# fetch_interactions
# ===========================================================================


class TestFetchInteractions:
    def test_returns_empty_dict_on_api_failure(self):
        import requests as req

        fetcher = _make_fetcher()
        with patch("src.recommender.training.data.mongo_data_fetcher.requests.post") as mock_post:
            mock_post.side_effect = req.exceptions.ConnectionError("unreachable")
            result = fetcher.fetch_interactions()

        assert result == {}

    def test_aggregates_weights_per_repo(self):
        """Multiple interactions on the same repo are summed."""
        fetcher = _make_fetcher()
        docs = [
            {"repo_id": "r1", "interaction_type": "save"},    # +3
            {"repo_id": "r1", "interaction_type": "click"},   # +1 → total 4
            {"repo_id": "r2", "interaction_type": "thumbs_up"},  # +5
            {"repo_id": "r2", "interaction_type": "dismiss"},    # -2 → total 3
        ]
        with patch("src.recommender.training.data.mongo_data_fetcher.requests.post") as mock_post:
            mock_post.return_value = _mock_post(documents=docs)
            result = fetcher.fetch_interactions()

        assert result["r1"] == pytest.approx(4.0)
        assert result["r2"] == pytest.approx(3.0)

    def test_zero_weight_interactions_excluded(self):
        """Unknown interaction types (weight=0) are not added to the dict."""
        fetcher = _make_fetcher()
        docs = [{"repo_id": "r1", "interaction_type": "unknown_type"}]
        with patch("src.recommender.training.data.mongo_data_fetcher.requests.post") as mock_post:
            mock_post.return_value = _mock_post(documents=docs)
            result = fetcher.fetch_interactions()

        assert "r1" not in result

    def test_negative_only_interactions_included(self):
        """Repos with only negative signals still appear (negative score)."""
        fetcher = _make_fetcher()
        docs = [{"repo_id": "r1", "interaction_type": "thumbs_down"}]  # -5
        with patch("src.recommender.training.data.mongo_data_fetcher.requests.post") as mock_post:
            mock_post.return_value = _mock_post(documents=docs)
            result = fetcher.fetch_interactions()

        assert result["r1"] == pytest.approx(-5.0)

    def test_empty_collection_returns_empty_dict(self):
        fetcher = _make_fetcher()
        with patch("src.recommender.training.data.mongo_data_fetcher.requests.post") as mock_post:
            mock_post.return_value = _mock_post(documents=[])
            result = fetcher.fetch_interactions()

        assert result == {}
