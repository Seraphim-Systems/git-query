"""Evaluation metrics and offline evaluation."""

from typing import List, Dict, Any
from collections import defaultdict
import numpy as np


class RecommenderEvaluator:
    """
    Evaluator for recommendation systems.

    Computes standard IR metrics:
    - Precision@K
    - Recall@K
    - NDCG@K
    - MRR (Mean Reciprocal Rank)
    - CTR (Click-Through Rate)
    """

    def __init__(self):
        pass

    def evaluate(
        self,
        predictions: List[List[str]],
        ground_truth: List[List[str]],
        k_values: List[int] = [1, 5, 10, 20],
    ) -> Dict[str, Any]:
        """
        Evaluate recommendations.

        Args:
            predictions: List of predicted repo IDs for each query
            ground_truth: List of relevant repo IDs for each query
            k_values: K values for computing metrics

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            "precision_at_k": {},
            "recall_at_k": {},
            "ndcg_at_k": {},
            "mrr": 0.0,
        }

        # Compute metrics for each K
        for k in k_values:
            precisions = []
            recalls = []
            ndcgs = []

            for pred, truth in zip(predictions, ground_truth):
                # Precision@K
                precisions.append(self._precision_at_k(pred, truth, k))

                # Recall@K
                recalls.append(self._recall_at_k(pred, truth, k))

                # NDCG@K
                ndcgs.append(self._ndcg_at_k(pred, truth, k))

            metrics["precision_at_k"][k] = np.mean(precisions)
            metrics["recall_at_k"][k] = np.mean(recalls)
            metrics["ndcg_at_k"][k] = np.mean(ndcgs)

        # MRR
        metrics["mrr"] = self._mean_reciprocal_rank(predictions, ground_truth)

        return metrics

    def _precision_at_k(
        self, predicted: List[str], relevant: List[str], k: int
    ) -> float:
        """Compute Precision@K."""
        if not predicted or not relevant:
            return 0.0

        predicted_k = predicted[:k]
        relevant_set = set(relevant)

        num_relevant = sum(1 for item in predicted_k if item in relevant_set)
        return num_relevant / min(k, len(predicted_k))

    def _recall_at_k(
        self, predicted: List[str], relevant: List[str], k: int
    ) -> float:
        """Compute Recall@K."""
        if not predicted or not relevant:
            return 0.0

        predicted_k = predicted[:k]
        relevant_set = set(relevant)

        num_relevant = sum(1 for item in predicted_k if item in relevant_set)
        return num_relevant / len(relevant_set)

    def _ndcg_at_k(
        self, predicted: List[str], relevant: List[str], k: int
    ) -> float:
        """Compute NDCG@K (Normalized Discounted Cumulative Gain)."""
        if not predicted or not relevant:
            return 0.0

        predicted_k = predicted[:k]
        relevant_set = set(relevant)

        # DCG
        dcg = 0.0
        for i, item in enumerate(predicted_k, start=1):
            if item in relevant_set:
                dcg += 1.0 / np.log2(i + 1)

        # IDCG (ideal DCG)
        idcg = 0.0
        for i in range(1, min(len(relevant), k) + 1):
            idcg += 1.0 / np.log2(i + 1)

        return dcg / idcg if idcg > 0 else 0.0

    def _mean_reciprocal_rank(
        self, predictions: List[List[str]], ground_truth: List[List[str]]
    ) -> float:
        """Compute Mean Reciprocal Rank."""
        rrs = []

        for pred, truth in zip(predictions, ground_truth):
            truth_set = set(truth)

            for i, item in enumerate(pred, start=1):
                if item in truth_set:
                    rrs.append(1.0 / i)
                    break
            else:
                rrs.append(0.0)

        return np.mean(rrs) if rrs else 0.0

    def compute_ctr(
        self, interactions: List[Dict[str, Any]], shown_results: List[Dict[str, Any]]
    ) -> float:
        """
        Compute Click-Through Rate.

        CTR = (Number of clicks) / (Number of impressions)
        """
        if not shown_results:
            return 0.0

        num_clicks = len([i for i in interactions if i.get("interaction_type") == "click"])
        num_impressions = len(shown_results)

        return num_clicks / num_impressions if num_impressions > 0 else 0.0

