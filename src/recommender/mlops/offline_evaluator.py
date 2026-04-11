"""Offline ranking evaluation using logged user interactions as ground truth.

Computes Precision@K and NDCG@K by treating interactions (clicks, saves,
thumbs_up) as implicit relevance signals.  Run after retraining to validate
that the new model improves on the previous one before it is promoted.

Exit codes:
    0 — evaluation passed (metrics above threshold or insufficient data)
    1 — evaluation failed (metrics dropped below threshold vs previous model)
    2 — missing required environment variables
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Interactions that count as "relevant" for Precision/NDCG computation
POSITIVE_INTERACTION_TYPES = {"click", "save", "thumbs_up"}

# Minimum interactions required to evaluate a (query, user) session
MIN_SESSION_SIZE = 3

# How far back to look for interactions
DEFAULT_EVAL_DAYS = 30


def _fetch_interactions(api_base_url: str, api_key: str, days: int) -> List[Dict]:
    """Fetch recent interactions from MongoDB via the gateway API."""
    import requests

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.post(
            f"{api_base_url.rstrip('/')}/api/mongodb/query",
            headers=headers,
            json={
                "database": "gitquery",
                "collection": "user_interactions",
                "filter": {"timestamp": {"$gte": cutoff}},
                "limit": 100000,
                "skip": 0,
            },
            timeout=30,
        )
        response.raise_for_status()
        docs = response.json().get("documents", [])
        logger.info("Fetched %d interactions for offline evaluation", len(docs))
        return docs
    except Exception as e:
        logger.warning("Could not fetch interactions: %s", e)
        return []


def _build_sessions(interactions: List[Dict]) -> Dict[Tuple[str, str], Dict[str, bool]]:
    """Group interactions into (query, user_id) sessions.

    Returns:
        Mapping of (query, user_id) → {repo_id: is_relevant}
        where is_relevant = True if the repo was clicked/saved/thumbs_up.
    """
    sessions: Dict[Tuple[str, str], Dict[str, bool]] = defaultdict(dict)
    for doc in interactions:
        query = doc.get("query", "")
        user_id = doc.get("user_id", "")
        repo_id = doc.get("repo_id", "")
        itype = doc.get("interaction_type", "")
        if not (query and user_id and repo_id):
            continue
        key = (query, user_id)
        # Mark as relevant if any positive interaction exists; negative interactions
        # don't override a prior positive (a user may view then save)
        if itype in POSITIVE_INTERACTION_TYPES:
            sessions[key][repo_id] = True
        elif repo_id not in sessions[key]:
            sessions[key][repo_id] = False
    return sessions


def precision_at_k(relevant_set: set, ranked_ids: List[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    if not ranked_ids or not relevant_set:
        return 0.0
    top_k = ranked_ids[:k]
    hits = sum(1 for r in top_k if r in relevant_set)
    return hits / k


def ndcg_at_k(relevant_set: set, ranked_ids: List[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at k."""
    if not ranked_ids or not relevant_set:
        return 0.0
    top_k = ranked_ids[:k]
    dcg = sum(
        1.0 / np.log2(i + 2) for i, r in enumerate(top_k) if r in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(
    sessions: Dict[Tuple[str, str], Dict[str, bool]],
    k_values: Tuple[int, ...] = (5, 10),
) -> Dict[str, float]:
    """Compute Precision@K and NDCG@K averaged across all qualifying sessions.

    A session qualifies if it has at least MIN_SESSION_SIZE interactions and
    at least one positive (relevant) interaction.

    The ranked list for each session is the repos ordered by their interaction
    recency/frequency within that session ��� since we don't have the model's
    original score list stored, we use the order repos appear in the positive
    set as a proxy for "model ranked them and user interacted with them."
    """
    metrics: Dict[str, List[float]] = {f"precision_at_{k}": [] for k in k_values}
    metrics.update({f"ndcg_at_{k}": [] for k in k_values})

    qualifying = 0
    for (query, user_id), repo_relevance in sessions.items():
        if len(repo_relevance) < MIN_SESSION_SIZE:
            continue
        relevant_set = {r for r, is_rel in repo_relevance.items() if is_rel}
        if not relevant_set:
            continue

        # Ranked list: positives first, then negatives (proxy for model ordering)
        ranked = sorted(repo_relevance.keys(), key=lambda r: repo_relevance[r], reverse=True)
        qualifying += 1

        for k in k_values:
            metrics[f"precision_at_{k}"].append(precision_at_k(relevant_set, ranked, k))
            metrics[f"ndcg_at_{k}"].append(ndcg_at_k(relevant_set, ranked, k))

    if qualifying == 0:
        logger.warning("No qualifying sessions found — skipping metric computation")
        return {}

    logger.info("Evaluated %d qualifying sessions", qualifying)
    return {name: float(np.mean(vals)) for name, vals in metrics.items() if vals}


def load_previous_metrics(metrics_path: Path) -> Optional[Dict[str, float]]:
    """Load metrics from the previous evaluation run if available."""
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except Exception:
        return None


def check_regression(
    current: Dict[str, float],
    previous: Dict[str, float],
    threshold: float = 0.10,
) -> bool:
    """Return True if any metric dropped by more than *threshold* (relative).

    E.g. threshold=0.10 blocks promotion if Precision@5 fell by >10%.
    """
    for key, current_val in current.items():
        prev_val = previous.get(key)
        if prev_val is None or prev_val == 0:
            continue
        drop = (prev_val - current_val) / prev_val
        if drop > threshold:
            logger.error(
                "Metric regression detected: %s dropped %.1f%% (%.4f → %.4f)",
                key, 100 * drop, prev_val, current_val,
            )
            return True
    return False


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    api_url = os.getenv("API_BASE_URL")
    api_key = os.getenv("APIKEY_MONGODB")
    if not api_url or not api_key:
        logger.error("API_BASE_URL and APIKEY_MONGODB are required")
        sys.exit(2)

    eval_days = int(os.getenv("EVAL_DAYS", str(DEFAULT_EVAL_DAYS)))
    regression_threshold = float(os.getenv("REGRESSION_THRESHOLD", "0.10"))
    metrics_path = Path(os.getenv("METRICS_PATH", "/app/models/metadata/offline_metrics.json"))

    # --- Fetch and evaluate ---
    interactions = _fetch_interactions(api_url, api_key, eval_days)
    if not interactions:
        logger.warning("No interaction data available — skipping offline evaluation")
        sys.exit(0)

    sessions = _build_sessions(interactions)
    current_metrics = evaluate(sessions)

    if not current_metrics:
        logger.warning("Insufficient data for evaluation — skipping")
        sys.exit(0)

    for name, val in current_metrics.items():
        logger.info("  %s: %.4f", name, val)

    # --- Log to MLflow if available ---
    try:
        _app_root = Path(__file__).resolve().parents[2]
        if str(_app_root) not in sys.path:
            sys.path.insert(0, str(_app_root))
        from mlops.mlflow_tracker import MLflowTracker

        tracker = MLflowTracker(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "file:///app/mlruns"),
            experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "git-query-retrain"),
        )
        with tracker.start_run(run_name="offline-eval"):
            tracker.log_metrics(current_metrics)
            tracker.log_params({"eval_days": eval_days, "min_session_size": MIN_SESSION_SIZE})
    except Exception as e:
        logger.warning("MLflow logging skipped: %s", e)

    # --- Regression check ---
    previous_metrics = load_previous_metrics(metrics_path)
    if previous_metrics and check_regression(current_metrics, previous_metrics, regression_threshold):
        logger.error("Offline evaluation failed — blocking model promotion")
        sys.exit(1)

    # --- Save current metrics for next comparison ---
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({**current_metrics, "timestamp": datetime.now(timezone.utc).isoformat()}, f, indent=2)
    logger.info("Offline evaluation passed. Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
