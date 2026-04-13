"""CLI entrypoint for the drift monitor Docker container.

Loads reference data from disk, fetches current data live from MongoDB,
runs all applicable drift checks (data, embedding, CTR), saves reports,
and exits with code 1 if drift is detected so the CI pipeline can react.

Environment variables (required):
    REFERENCE_DATA_PATH       Path to reference parquet saved after last training run

Environment variables (optional — enable additional checks):
    API_BASE_URL              MongoDB gateway URL (enables live current-data fetch + CTR)
    APIKEY_MONGODB            MongoDB/gateway API key
    QDRANT_URL                Qdrant service URL (enables embedding drift)
    APIKEY_QDRANT             Qdrant API key (falls back to APIKEY_MONGODB)
    QDRANT_COLLECTION         Qdrant collection name (default: repositories_embeddings)
    REFERENCE_EMBEDDINGS_PATH Path to reference embeddings .npy file
    REFERENCE_SCORES_PATH     Path to reference prediction scores .npy file (enables prediction drift)
    MODELS_DIR                Directory to search for latest lgbm_*.pkl (default: /app/models)
    CURRENT_DATA_PATH         Path to current data parquet (alternative to live fetch)
    EVIDENTLY_REPORT_PATH     Directory for JSON drift reports (default: /app/drift_reports)
    EVIDENTLY_WORKSPACE_PATH  Evidently workspace path for UI display

Exit codes:
    0 — no drift detected, insufficient data, or reference data not yet available
    1 — drift detected
    2 — missing required environment variables or invalid configuration
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

try:
    from mlflow_tracker import MLflowTracker
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("mlflow_tracker not available — drift results will not be logged to MLflow")


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_parquet(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".json":
        return pd.read_json(p)
    raise ValueError(f"Unsupported format: {p.suffix}. Use .parquet or .json")


def _fetch_current_repos(api_url: str, api_key: str, max_repos: int = 5_000) -> pd.DataFrame:
    """Fetch a sample of current repos from MongoDB as the live current dataset."""
    import requests

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    all_docs: list[dict] = []
    batch_size = 1_000

    for skip in range(0, max_repos, batch_size):
        limit = min(batch_size, max_repos - skip)
        try:
            resp = requests.post(
                f"{api_url.rstrip('/')}/api/mongodb/query",
                headers=headers,
                json={
                    "database": "gitquery",
                    "collection": "repositories",
                    "filter": {},
                    "limit": limit,
                    "skip": skip,
                    "sort": {"_id": 1},
                },
                timeout=60,
            )
            resp.raise_for_status()
            batch = resp.json().get("documents", [])
            if not batch:
                break
            all_docs.extend(batch)
            logger.info("Fetched %d repos so far...", len(all_docs))
        except Exception as e:
            logger.warning("Repo fetch error at skip=%d: %s", skip, e)
            break

    logger.info("Fetched %d total current repos from MongoDB", len(all_docs))
    return pd.DataFrame(all_docs) if all_docs else pd.DataFrame()


def _fetch_current_embeddings(
    qdrant_url: str,
    api_key: str,
    collection: str = "repositories_embeddings",
    sample_size: int = 2_000,
) -> np.ndarray | None:
    """Sample current embedding vectors from Qdrant for embedding drift detection."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=60)
        points, _ = client.scroll(
            collection_name=collection,
            limit=sample_size,
            with_vectors=True,
        )
        if not points:
            logger.warning("No points returned from Qdrant collection '%s'", collection)
            return None
        vectors = np.array([p.vector for p in points], dtype=np.float32)
        logger.info("Fetched %d current embeddings from Qdrant (shape=%s)", len(vectors), vectors.shape)
        return vectors
    except Exception as e:
        logger.warning("Could not fetch current embeddings from Qdrant: %s", e)
        return None


def _fetch_raw_interactions(
    api_url: str,
    api_key: str,
) -> pd.DataFrame:
    """Fetch all user interaction documents from MongoDB as a raw DataFrame.

    Returns an empty DataFrame when the collection is unreachable or empty —
    callers must handle this gracefully (CTR drift and query derivation both
    skip cleanly when interactions are absent).
    """
    import requests

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    docs: list[dict] = []
    batch_size = 1_000
    try:
        while True:
            resp = requests.post(
                f"{api_url.rstrip('/')}/api/mongodb/query",
                headers=headers,
                json={
                    "database": "gitquery",
                    "collection": "user_interactions",
                    "filter": {},
                    "limit": batch_size,
                    "skip": len(docs),
                },
                timeout=30,
            )
            resp.raise_for_status()
            batch = resp.json().get("documents", [])
            if not batch:
                break
            docs.extend(batch)
    except Exception as e:
        logger.warning("Could not fetch interactions: %s", e)
        if not docs:
            return pd.DataFrame()

    if not docs:
        logger.info("No user interactions recorded yet — interaction-dependent checks will be skipped")
        return pd.DataFrame()

    df = pd.DataFrame(docs)
    logger.info("Fetched %d interaction records", len(df))
    return df


def _split_interactions_for_ctr(
    interactions_df: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Derive binary CTR labels and split into (reference, current) halves.

    Derives a binary ``clicked`` column:
      - click / save / thumbs_up  → 1
      - dismiss / thumbs_down / other → 0

    Splits chronologically so reference = older half, current = newer half.
    Returns (None, None) when fewer than 20 interactions exist — not enough
    for a meaningful statistical test.
    """
    if interactions_df.empty:
        return None, None

    if len(interactions_df) < 20:
        logger.info("Too few interactions (%d) for CTR drift — skipping", len(interactions_df))
        return None, None

    df = interactions_df.copy()
    positive_types = {"click", "save", "thumbs_up"}
    df["clicked"] = df.get("interaction_type", pd.Series(dtype=str)).isin(positive_types).astype(int)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")

    mid = len(df) // 2
    ref_df = df.iloc[:mid][["clicked"]].reset_index(drop=True)
    cur_df = df.iloc[mid:][["clicked"]].reset_index(drop=True)

    logger.info(
        "CTR split: reference=%d rows (CTR=%.3f), current=%d rows (CTR=%.3f)",
        len(ref_df),
        ref_df["clicked"].mean(),
        len(cur_df),
        cur_df["clicked"].mean(),
    )
    return ref_df, cur_df


# ---------------------------------------------------------------------------
# Prediction scoring
# ---------------------------------------------------------------------------


def _find_latest_lgbm_model(models_dir: str) -> str | None:
    """Return the most recently modified lgbm_*.pkl file under models_dir."""
    import glob

    pattern = os.path.join(models_dir, "lgbm_*.pkl")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _derive_interaction_query(
    interactions_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> str:
    """Derive a representative query string from recent positive user interactions.

    The recommender scores repos by computing the dot product between repo
    feature vectors and a user interaction profile (weighted mean of feature
    vectors of positively-interacted repos).  This function approximates that
    profile as a text query built from the names, languages, and descriptions
    of the most-clicked/saved repos — giving ``LGBMRanker.predict()`` a
    meaningful query context instead of an empty string.

    Falls back to the most common language in ``current_df`` when interaction
    data is absent or too sparse.
    """
    positive_types = {"click", "save", "thumbs_up"}

    # Try to identify top positively-interacted repos
    top_repo_ids: list = []
    if not interactions_df.empty and "interaction_type" in interactions_df.columns:
        pos = interactions_df[interactions_df["interaction_type"].isin(positive_types)]
        id_col = next((c for c in ("repo_id", "repository_id", "full_name") if c in pos.columns), None)
        if id_col and not pos.empty:
            top_repo_ids = pos[id_col].value_counts().head(20).index.tolist()

    if top_repo_ids and not current_df.empty:
        # Match top-interacted repo IDs against current data.
        # repo_id checked first — it's the same field used in user_interactions.
        match_col = next((c for c in ("repo_id", "full_name", "name", "_id", "id") if c in current_df.columns), None)
        if match_col:
            matched = current_df[current_df[match_col].isin(top_repo_ids)]
            if not matched.empty:
                terms: list[str] = []
                for col in ("name", "language", "description"):
                    if col in matched.columns:
                        terms.extend(matched[col].dropna().astype(str).head(10).tolist())
                query = " ".join(terms[:60])
                logger.info(
                    "Derived interaction query from %d positively-interacted repos: %r...",
                    len(matched),
                    query[:60],
                )
                return query

    # Fallback: most common language in current corpus
    if not current_df.empty and "language" in current_df.columns:
        counts = current_df["language"].dropna().value_counts()
        if not counts.empty:
            fallback = str(counts.index[0])
            logger.info("No interaction data — using fallback query: %r", fallback)
            return fallback

    return ""


def _score_current_repos(
    model_path: str,
    current_df: pd.DataFrame,
    interaction_query: str = "",
) -> list[float] | None:
    """Score all current repos with the production LightGBM model.

    Uses ``interaction_query`` as the query context so scores reflect the
    dot-product relationship between repo features and the user interaction
    profile — the same mechanism used in production ranking.
    Returns None on any failure so the caller can skip prediction drift.
    """
    try:
        from training.lgbm_ranker import LGBMRanker

        ranker = LGBMRanker.load(model_path)
        scores = ranker.predict(current_df, query_text=interaction_query)
        logger.info(
            "Scored %d current repos for prediction drift (model=%s, query=%r...)",
            len(scores),
            os.path.basename(model_path),
            interaction_query[:40],
        )
        return scores.tolist()
    except Exception as e:
        logger.warning("Could not score current repos for prediction drift: %s", e)
        return None


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


def _log_drift_to_mlflow(report: dict) -> None:
    """Log drift check results to MLflow so they appear alongside model metrics.

    Creates a run in the ``git-query-drift-checks`` experiment (overridable via
    MLFLOW_EXPERIMENT_NAME).  Each call produces one run tagged ``run_type=drift_check``
    so it can be filtered separately from training runs in the MLflow UI.

    Metrics logged:
      - drift_detected          — 1 if any check detected drift, else 0
      - <check>_drift_detected  — per-check flag (data, embedding, prediction, ctr)
      - prediction_score_mean_reference / _current — score distribution means
      - ctr_reference / ctr_current / ctr_change   — engagement rate values
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available — skipping drift logging to MLflow")
        return

    try:
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "git-query-drift-checks")
        tracker = MLflowTracker(experiment_name=experiment_name)

        checks: dict = report.get("checks", {})
        overall_drift: bool = report.get("overall_drift_detected", False)

        metrics: dict[str, float] = {
            "drift_detected": float(overall_drift),
        }

        for check_name in ("data_drift", "embedding_drift", "prediction_drift", "ctr_drift"):
            check = checks.get(check_name, {})
            metrics[f"{check_name}_detected"] = float(check.get("drift_detected", False))

        # Prediction score distribution
        pred = checks.get("prediction_drift", {})
        if "reference_mean" in pred:
            metrics["prediction_score_mean_reference"] = float(pred["reference_mean"])
        if "current_mean" in pred:
            metrics["prediction_score_mean_current"] = float(pred["current_mean"])

        # CTR values
        ctr = checks.get("ctr_drift", {})
        if "reference_ctr" in ctr:
            metrics["ctr_reference"] = float(ctr["reference_ctr"])
        if "current_ctr" in ctr:
            metrics["ctr_current"] = float(ctr["current_ctr"])
        if "ctr_change" in ctr:
            metrics["ctr_change"] = float(ctr["ctr_change"])

        checks_run = list(checks.keys())
        tags = {
            "run_type": "drift_check",
            "checks_run": ",".join(checks_run) if checks_run else "none",
            "drift_detected": str(overall_drift),
        }

        run_name = f"drift-check-{'drift' if overall_drift else 'clean'}"
        with tracker.start_run(run_name=run_name, tags=tags):
            tracker.log_metrics(metrics)

        logger.info("Drift results logged to MLflow experiment '%s'", experiment_name)
    except Exception as e:
        logger.warning("Could not log drift results to MLflow (non-fatal): %s", e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    reference_path = os.getenv("REFERENCE_DATA_PATH")
    current_path = os.getenv("CURRENT_DATA_PATH")
    api_url = os.getenv("API_BASE_URL")
    api_key = os.getenv("APIKEY_MONGODB")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("APIKEY_QDRANT") or api_key
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "repositories_embeddings")
    embeddings_path = os.getenv("REFERENCE_EMBEDDINGS_PATH")
    scores_path = os.getenv("REFERENCE_SCORES_PATH")
    models_dir = os.getenv("MODELS_DIR", "/app/models")
    report_dir = os.getenv("EVIDENTLY_REPORT_PATH", "/app/drift_reports")

    if not reference_path:
        logger.error("REFERENCE_DATA_PATH is required.")
        sys.exit(2)

    # --- Reference data (features from last training run) ---
    logger.info("Loading reference data from: %s", reference_path)
    try:
        reference_df = _load_parquet(reference_path)
    except FileNotFoundError:
        logger.warning(
            "Reference data not found at %s — no training run has completed yet. Skipping drift check.",
            reference_path,
        )
        sys.exit(0)
    logger.info("Reference data shape: %s", reference_df.shape)

    # --- Current data (live repos from MongoDB or file) ---
    current_df: pd.DataFrame | None = None
    if current_path and Path(current_path).exists():
        logger.info("Loading current data from file: %s", current_path)
        current_df = _load_parquet(current_path)
    elif api_url and api_key:
        logger.info("Fetching current repos from MongoDB...")
        current_df = _fetch_current_repos(api_url, api_key)
    else:
        logger.error("Provide either CURRENT_DATA_PATH or both API_BASE_URL + APIKEY_MONGODB.")
        sys.exit(2)

    if current_df is None or current_df.empty:
        logger.error("Current data is empty — cannot run drift check.")
        sys.exit(2)
    logger.info("Current data shape: %s", current_df.shape)

    # --- Reference embeddings (saved by EmbeddingIndexingPipeline) ---
    reference_embeddings: np.ndarray | None = None
    if embeddings_path and Path(embeddings_path).exists():
        reference_embeddings = np.load(embeddings_path).astype(np.float32)
        logger.info("Loaded reference embeddings: shape=%s", reference_embeddings.shape)
    else:
        logger.info("REFERENCE_EMBEDDINGS_PATH not set or not found — skipping embedding drift")

    # --- Current embeddings (sampled from Qdrant) ---
    current_embeddings: np.ndarray | None = None
    if reference_embeddings is not None and qdrant_url and qdrant_key:
        current_embeddings = _fetch_current_embeddings(qdrant_url, qdrant_key, qdrant_collection)
    elif reference_embeddings is not None:
        logger.info("QDRANT_URL not set — skipping embedding drift")

    # --- Interactions (fetched once, used for both CTR drift and interaction query) ---
    # Falls back cleanly to empty DataFrame when no interactions are recorded yet.
    all_interactions: pd.DataFrame = pd.DataFrame()
    if api_url and api_key:
        all_interactions = _fetch_raw_interactions(api_url, api_key)

    ref_interactions, cur_interactions = _split_interactions_for_ctr(all_interactions)

    # --- Reference prediction scores (saved by training pipeline) ---
    reference_scores: list[float] | None = None
    if scores_path and Path(scores_path).exists():
        reference_scores = np.load(scores_path).tolist()
        logger.info("Loaded reference scores: %d samples", len(reference_scores))
    else:
        logger.info("REFERENCE_SCORES_PATH not set or not found — skipping prediction drift")

    # --- Current prediction scores (model scores on full live repo set) ---
    # interaction_query approximates the user interaction profile used in production
    # so score distributions are comparable to real serving behaviour.
    current_scores: list[float] | None = None
    if reference_scores is not None:
        model_path = _find_latest_lgbm_model(models_dir)
        if model_path:
            interaction_query = _derive_interaction_query(all_interactions, current_df)
            logger.info("Scoring %d current repos with model: %s", len(current_df), os.path.basename(model_path))
            current_scores = _score_current_repos(model_path, current_df, interaction_query)
        else:
            logger.info("No lgbm_*.pkl found in %s — skipping prediction drift", models_dir)

    # --- Run all checks ---
    from drift_monitor import DriftMonitor

    monitor = DriftMonitor(report_dir=report_dir)
    logger.info("Running drift checks...")

    report = monitor.run_full_drift_check(
        reference_data=reference_df,
        current_data=current_df,
        reference_embeddings=reference_embeddings,
        current_embeddings=current_embeddings,
        reference_scores=reference_scores,
        current_scores=current_scores,
        reference_interactions=ref_interactions,
        current_interactions=cur_interactions,
    )

    checks_run = list(report.get("checks", {}).keys())
    drift_detected = report.get("overall_drift_detected", False)
    logger.info("Checks run: %s", checks_run)
    logger.info("Overall drift detected: %s", drift_detected)

    # --- Log to MLflow ---
    _log_drift_to_mlflow(report)

    # --- Write summary ---
    summary_path = Path(report_dir) / "drift_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "drift_detected": drift_detected,
                "timestamp": report.get("timestamp"),
                "checks_run": checks_run,
            },
            f,
            indent=2,
        )
    logger.info("Summary written to: %s", summary_path)

    if drift_detected:
        logger.warning("Drift detected — exiting with code 1.")
        sys.exit(1)

    logger.info("No drift detected.")
    sys.exit(0)


if __name__ == "__main__":
    main()
