"""Unit tests for offline_evaluator — London School TDD (mock-first)."""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    def test_all_relevant(self):
        from src.recommender.mlops.offline_evaluator import precision_at_k

        assert precision_at_k({"a", "b", "c"}, ["a", "b", "c"], k=3) == pytest.approx(1.0)

    def test_none_relevant(self):
        from src.recommender.mlops.offline_evaluator import precision_at_k

        assert precision_at_k({"x"}, ["a", "b", "c"], k=3) == pytest.approx(0.0)

    def test_partial_hit(self):
        from src.recommender.mlops.offline_evaluator import precision_at_k

        # 1 hit in top 2
        assert precision_at_k({"a"}, ["a", "b"], k=2) == pytest.approx(0.5)

    def test_k_larger_than_list(self):
        """k larger than ranked list — denominator is still k."""
        from src.recommender.mlops.offline_evaluator import precision_at_k

        assert precision_at_k({"a"}, ["a"], k=5) == pytest.approx(0.2)

    def test_empty_ranked_list(self):
        from src.recommender.mlops.offline_evaluator import precision_at_k

        assert precision_at_k({"a"}, [], k=5) == pytest.approx(0.0)

    def test_empty_relevant_set(self):
        from src.recommender.mlops.offline_evaluator import precision_at_k

        assert precision_at_k(set(), ["a", "b"], k=2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


class TestNdcgAtK:
    def test_perfect_ranking(self):
        """Relevant items ranked first → NDCG = 1.0."""
        from src.recommender.mlops.offline_evaluator import ndcg_at_k

        assert ndcg_at_k({"a", "b"}, ["a", "b", "c", "d"], k=2) == pytest.approx(1.0)

    def test_worst_ranking(self):
        """Relevant items ranked last → NDCG < 1.0."""
        from src.recommender.mlops.offline_evaluator import ndcg_at_k

        score = ndcg_at_k({"c", "d"}, ["a", "b", "c", "d"], k=4)
        assert 0.0 <= score < 1.0

    def test_no_relevant_in_top_k(self):
        from src.recommender.mlops.offline_evaluator import ndcg_at_k

        assert ndcg_at_k({"z"}, ["a", "b", "c"], k=3) == pytest.approx(0.0)

    def test_empty_ranked_list(self):
        from src.recommender.mlops.offline_evaluator import ndcg_at_k

        assert ndcg_at_k({"a"}, [], k=5) == pytest.approx(0.0)

    def test_empty_relevant_set(self):
        from src.recommender.mlops.offline_evaluator import ndcg_at_k

        assert ndcg_at_k(set(), ["a", "b"], k=2) == pytest.approx(0.0)

    def test_single_relevant_item(self):
        """One relevant item at rank 1 → DCG = 1 / log2(2) = 1.0 = IDCG → NDCG = 1.0."""
        from src.recommender.mlops.offline_evaluator import ndcg_at_k

        assert ndcg_at_k({"a"}, ["a", "b", "c"], k=3) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _build_sessions
# ---------------------------------------------------------------------------


class TestBuildSessions:
    def test_positive_interaction_marks_repo_relevant(self):
        from src.recommender.mlops.offline_evaluator import _build_sessions

        docs = [{"query": "python", "user_id": "u1", "repo_id": "r1", "interaction_type": "save"}]
        sessions = _build_sessions(docs)
        assert sessions[("python", "u1")]["r1"] is True

    def test_negative_interaction_marks_repo_not_relevant(self):
        from src.recommender.mlops.offline_evaluator import _build_sessions

        docs = [{"query": "python", "user_id": "u1", "repo_id": "r1", "interaction_type": "dismiss"}]
        sessions = _build_sessions(docs)
        assert sessions[("python", "u1")]["r1"] is False

    def test_positive_overrides_prior_negative(self):
        """A save after a dismiss counts the repo as relevant."""
        from src.recommender.mlops.offline_evaluator import _build_sessions

        docs = [
            {"query": "q", "user_id": "u", "repo_id": "r1", "interaction_type": "dismiss"},
            {"query": "q", "user_id": "u", "repo_id": "r1", "interaction_type": "save"},
        ]
        sessions = _build_sessions(docs)
        assert sessions[("q", "u")]["r1"] is True

    def test_negative_does_not_override_positive(self):
        """A dismiss after a save does NOT flip relevance back to False."""
        from src.recommender.mlops.offline_evaluator import _build_sessions

        docs = [
            {"query": "q", "user_id": "u", "repo_id": "r1", "interaction_type": "save"},
            {"query": "q", "user_id": "u", "repo_id": "r1", "interaction_type": "dismiss"},
        ]
        sessions = _build_sessions(docs)
        assert sessions[("q", "u")]["r1"] is True

    def test_docs_missing_required_fields_are_skipped(self):
        from src.recommender.mlops.offline_evaluator import _build_sessions

        docs = [
            {"query": "", "user_id": "u1", "repo_id": "r1", "interaction_type": "save"},  # empty query
            {"query": "q", "user_id": "", "repo_id": "r1", "interaction_type": "save"},  # empty user
            {"query": "q", "user_id": "u1", "repo_id": "", "interaction_type": "save"},  # empty repo
        ]
        sessions = _build_sessions(docs)
        assert len(sessions) == 0

    def test_multiple_users_same_query_are_separate_sessions(self):
        from src.recommender.mlops.offline_evaluator import _build_sessions

        docs = [
            {"query": "python", "user_id": "u1", "repo_id": "r1", "interaction_type": "click"},
            {"query": "python", "user_id": "u2", "repo_id": "r2", "interaction_type": "click"},
        ]
        sessions = _build_sessions(docs)
        assert ("python", "u1") in sessions
        assert ("python", "u2") in sessions


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def _make_sessions(self, n_repos: int = 5, n_relevant: int = 2):
        """Helper: one session with n_repos total, first n_relevant are positive."""
        from collections import defaultdict

        sessions = defaultdict(dict)
        for i in range(n_repos):
            sessions[("q", "u")][f"r{i}"] = i < n_relevant
        return sessions

    def test_returns_expected_metric_keys(self):
        from src.recommender.mlops.offline_evaluator import evaluate

        sessions = self._make_sessions(n_repos=5, n_relevant=2)
        result = evaluate(sessions, k_values=(5, 10))
        assert "precision_at_5" in result
        assert "precision_at_10" in result
        assert "ndcg_at_5" in result
        assert "ndcg_at_10" in result

    def test_returns_empty_dict_when_no_qualifying_sessions(self):
        """Sessions with <MIN_SESSION_SIZE repos are skipped; empty dict returned."""
        from src.recommender.mlops.offline_evaluator import evaluate

        # Only 2 repos — below MIN_SESSION_SIZE=3
        sessions = {("q", "u"): {"r1": True, "r2": False}}
        result = evaluate(sessions, k_values=(5,))
        assert result == {}

    def test_returns_empty_dict_when_no_positive_interactions(self):
        """Sessions with no positive repos are skipped."""
        from src.recommender.mlops.offline_evaluator import evaluate

        sessions = {("q", "u"): {"r1": False, "r2": False, "r3": False}}
        result = evaluate(sessions, k_values=(5,))
        assert result == {}

    def test_metrics_are_floats_between_0_and_1(self):
        from src.recommender.mlops.offline_evaluator import evaluate

        sessions = self._make_sessions(n_repos=5, n_relevant=2)
        result = evaluate(sessions, k_values=(5,))
        for val in result.values():
            assert 0.0 <= val <= 1.0

    def test_perfect_ranking_gives_precision_1(self):
        """Positives ranked first → Precision@k = 1 when k ≤ n_relevant."""
        from src.recommender.mlops.offline_evaluator import evaluate

        # 5 repos: r0, r1, r2 relevant; r3, r4 not
        # evaluate() ranks: positives (True) first → ranked = [r0,r1,r2,r3,r4]
        sessions = {("q", "u"): {"r0": True, "r1": True, "r2": True, "r3": False, "r4": False}}
        result = evaluate(sessions, k_values=(3,))
        assert result["precision_at_3"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# check_regression
# ---------------------------------------------------------------------------


class TestCheckRegression:
    def test_no_regression_when_metrics_improve(self):
        from src.recommender.mlops.offline_evaluator import check_regression

        current = {"precision_at_5": 0.80, "ndcg_at_5": 0.85}
        previous = {"precision_at_5": 0.75, "ndcg_at_5": 0.80}
        assert check_regression(current, previous, threshold=0.10) is False

    def test_regression_detected_when_metric_drops(self):
        from src.recommender.mlops.offline_evaluator import check_regression

        current = {"precision_at_5": 0.60}
        previous = {"precision_at_5": 0.75}  # drop = 20% > 10% threshold
        assert check_regression(current, previous, threshold=0.10) is True

    def test_no_regression_within_threshold(self):
        from src.recommender.mlops.offline_evaluator import check_regression

        current = {"precision_at_5": 0.68}
        previous = {"precision_at_5": 0.75}  # drop = ~9.3% < 10%
        assert check_regression(current, previous, threshold=0.10) is False

    def test_missing_previous_key_skipped(self):
        """Metric not in previous dict is ignored (first run scenario)."""
        from src.recommender.mlops.offline_evaluator import check_regression

        current = {"ndcg_at_10": 0.50}
        previous = {}
        assert check_regression(current, previous, threshold=0.10) is False

    def test_zero_previous_value_skipped(self):
        """Previous value of 0 cannot compute drop percentage — skip."""
        from src.recommender.mlops.offline_evaluator import check_regression

        current = {"precision_at_5": 0.0}
        previous = {"precision_at_5": 0.0}
        assert check_regression(current, previous, threshold=0.10) is False

    def test_custom_threshold(self):
        """A tighter threshold (5%) catches smaller drops."""
        from src.recommender.mlops.offline_evaluator import check_regression

        current = {"precision_at_5": 0.72}
        previous = {"precision_at_5": 0.75}  # drop = 4% — within 10% but exceeds 3%
        assert check_regression(current, previous, threshold=0.03) is True
        assert check_regression(current, previous, threshold=0.10) is False
