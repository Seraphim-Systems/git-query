"""LightGBM LambdaRank training pipeline for repository reranking.

Extracts and productionises the learning-to-rank workflow from
notebooks/01_exploration_and_baseline.ipynb.

Design goals
------------
- Swappable: the ML team can train a replacement model and register it in
  MLflow under the same experiment name, then promote it via the model
  registry without touching the serving infrastructure.
- Lightweight imports: FeatureExtractor is loaded lazily so this module
  can be imported in MLOps-only contexts without triggering the full
  recommender package init (which requires torch).
- Observable: every training run is logged to MLflow via an injected
  MLflowTracker so results are reproducible and comparable.

Usage
-----
    from src.recommender.training.lgbm_ranker import LGBMRanker, build_query_groups
    from src.recommender.data.features import FeatureExtractor
    from src.recommender.mlops.mlflow_tracker import MLflowTracker

    grouped_df = build_query_groups(raw_df)
    fe = FeatureExtractor()
    tracker = MLflowTracker(experiment_name="git-query-ranker")

    ranker = LGBMRanker()
    with tracker.start_run(run_name="lgbm-v1"):
        metrics = ranker.train(grouped_df, feature_extractor=fe, tracker=tracker)
    ranker.save("/app/models/lgbm_v1.pkl")
"""

import importlib.util
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logger.warning("lightgbm not installed. LGBMRanker.train() will raise ImportError.")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 5, 10],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbose": -1,
}

EVAL_SEEDS: list[int] = [7, 21, 42, 84, 168]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def build_query_groups(
    df: pd.DataFrame,
    top_n_queries: int = 100,
    candidates_per_query: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build query-grouped training data from a raw repository DataFrame.

    Groups repos by ``language`` (used as a proxy query), samples positive
    candidates and hard negatives for each group, and assigns ``query_id``
    and ``query_text`` columns required by :class:`LGBMRanker`.

    Args:
        df: Raw repo DataFrame with at least ``language`` and ``stars`` columns.
        top_n_queries: Number of top language groups to use as queries.
        candidates_per_query: Max positive candidates per query group.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with ``query_id`` and ``query_text`` columns added.

    Raises:
        ValueError: If ``df`` has no ``language`` column or yields no groups.
    """
    if "language" not in df.columns:
        raise ValueError("DataFrame must have a 'language' column for query grouping.")

    work_df = df.copy()
    for col, default in [("name", ""), ("description", ""), ("license", ""), ("topics", None)]:
        if col not in work_df.columns:
            work_df[col] = [[] if default is None else default] * len(work_df)

    work_df["_query"] = work_df["language"].fillna("missing").astype(str).str.strip().str.lower()
    work_df = work_df[work_df["_query"] != ""].reset_index(drop=True)

    top_queries = work_df["_query"].value_counts().head(top_n_queries).index.tolist()
    rng = np.random.default_rng(random_state)
    rows: list[pd.DataFrame] = []
    qid = 0

    for q in top_queries:
        q_df = work_df[work_df["_query"] == q]
        if len(q_df) < 5:
            continue

        n_cand = min(candidates_per_query, len(q_df))
        idx = rng.choice(len(q_df), n_cand, replace=False)
        candidates = q_df.iloc[idx]

        neg_pool = work_df[work_df["_query"] != q]
        if len(neg_pool) == 0:
            continue
        neg_n = min(candidates_per_query, max(20, n_cand // 2), len(neg_pool))
        neg_idx = rng.choice(len(neg_pool), neg_n, replace=False)
        negs = neg_pool.iloc[neg_idx]

        pool = pd.concat([candidates, negs], ignore_index=True)
        pool["query_id"] = qid
        pool["query_text"] = q
        rows.append(pool)
        qid += 1

    if not rows:
        raise ValueError(
            "Could not build any query groups. "
            "Ensure the DataFrame has enough language diversity (≥5 repos per language)."
        )

    result = pd.concat(rows, ignore_index=True)
    result = result.drop(columns=["_query"], errors="ignore")
    return result


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------


class LGBMRanker:
    """LightGBM LambdaRank model for repository reranking.

    Trains a learning-to-rank model using proxy relevance labels derived from
    star counts within each query group.  Multi-seed evaluation measures
    stability across different train/val splits.

    The model is designed to be *swappable*: save/load uses a plain dict
    payload so the ML team can hot-swap an improved model by registering a
    new version in MLflow and promoting it — no serving code changes needed.
    """

    def __init__(self, params: dict[str, Any] | None = None):
        self.params: dict[str, Any] = {**DEFAULT_PARAMS, **(params or {})}
        self.model = None
        self.feature_cols: list[str] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_feature_extractor(self):
        """Load FeatureExtractor directly from its file to avoid triggering
        src/recommender/__init__.py (which imports torch-dependent engines)."""
        fe_path = Path(__file__).parent.parent / "data" / "features.py"
        spec = importlib.util.spec_from_file_location("_lgbm_features_mod", fe_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.FeatureExtractor()

    def _build_rank_data(
        self,
        grouped_df: pd.DataFrame,
        feature_extractor,
    ) -> tuple[pd.DataFrame, pd.Series, list[int]]:
        """Extract features and relevance labels from a grouped DataFrame.

        When ``grouped_df`` contains an ``interaction_score`` column (populated
        by MongoDataFetcher.fetch_training_pairs), relevance is derived from a
        dot-product similarity between each repo's feature vector and an
        interaction profile — the weighted mean of feature vectors of positively
        interacted repos.  This uses cosine similarity to avoid magnitude bias.

        Falls back to star-quartile labels when interaction data is sparse
        (<3 positive repos in the group).
        """
        feature_frames: list[pd.DataFrame] = []
        labels: list[pd.Series] = []
        group_sizes: list[int] = []
        has_interaction_col = "interaction_score" in grouped_df.columns
        dot_product_groups = 0

        for _, grp in grouped_df.groupby("query_id", sort=True):
            qtext = str(grp["query_text"].iloc[0])
            feats = feature_extractor.extract_all(grp, query=qtext)
            feat_vals = feats.values.astype(float)

            # --- Star-quartile labels (always computed as fallback) ---
            stars = pd.to_numeric(grp.get("stars", 0), errors="coerce").fillna(0)
            threshold = stars.quantile(0.75)
            star_rel = (stars >= threshold).astype(int)
            if star_rel.sum() == 0:
                star_rel = (stars == stars.max()).astype(int)

            rel = star_rel

            # --- Dot-product labels from interaction profile ---
            if has_interaction_col:
                interaction_scores = pd.to_numeric(
                    grp["interaction_score"], errors="coerce"
                ).fillna(0.0).values
                pos_mask = interaction_scores > 0
                if pos_mask.sum() >= 3:
                    # Interaction profile = weighted mean of positively interacted repo features
                    pos_weights = interaction_scores[pos_mask]
                    profile = np.average(feat_vals[pos_mask], axis=0, weights=pos_weights)

                    # Cosine similarity: normalise both profile and repo vectors
                    profile_norm = np.linalg.norm(profile)
                    if profile_norm > 0:
                        profile = profile / profile_norm
                    row_norms = np.linalg.norm(feat_vals, axis=1, keepdims=True)
                    feat_norm = feat_vals / np.where(row_norms > 0, row_norms, 1.0)

                    dot_scores = feat_norm @ profile  # shape: (n_repos,)
                    threshold_dot = np.percentile(dot_scores, 60)  # top 40% = relevant
                    rel = pd.Series(
                        (dot_scores >= threshold_dot).astype(int), index=grp.index
                    )
                    dot_product_groups += 1

            feature_frames.append(feats)
            labels.append(rel)
            group_sizes.append(len(feats))

        if has_interaction_col:
            logger.info(
                "Dot-product labels used for %d / %d query groups; star fallback for the rest",
                dot_product_groups, len(group_sizes),
            )

        X = pd.concat(feature_frames, axis=0).reset_index(drop=True)
        y = pd.concat(labels, axis=0).reset_index(drop=True)
        return X, y, group_sizes

    def _ndcg_by_group(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: list[int],
        k: int = 10,
    ) -> tuple[float, float]:
        """Return (mean, median) NDCG@k across query groups."""
        scores: list[float] = []
        start = 0
        for g in groups:
            end = start + g
            y_true = y.values[start:end].reshape(1, -1)
            y_pred = self.model.predict(X.values[start:end]).reshape(1, -1)
            try:
                scores.append(float(ndcg_score(y_true, y_pred, k=min(k, g))))
            except Exception:
                scores.append(float("nan"))
            start = end
        return float(np.nanmean(scores)), float(np.nanmedian(scores))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        grouped_df: pd.DataFrame,
        feature_extractor=None,
        tracker=None,
        dataset_version: str | None = None,
        seeds: list[int] | None = None,
        num_boost_rounds: int = 500,
        early_stopping_rounds: int = 50,
    ) -> dict[str, Any]:
        """Train LightGBM LambdaRank with multi-seed holdout evaluation.

        Args:
            grouped_df: DataFrame produced by :func:`build_query_groups`.
                Must have ``query_id``, ``query_text``, ``stars``, and raw
                repo columns consumed by ``feature_extractor``.
            feature_extractor: ``FeatureExtractor`` instance.  Loaded lazily
                if *None* (requires the recommender package to be importable).
            tracker: ``MLflowTracker`` instance.  When provided, params and
                metrics are logged to the active MLflow run.
            dataset_version: Opaque version string logged to MLflow as
                ``dataset_version`` param (see :mod:`dataset_versioner`).
            seeds: Random seeds for multi-seed stability evaluation.
                Defaults to ``EVAL_SEEDS`` = [7, 21, 42, 84, 168].
            num_boost_rounds: Maximum LightGBM boosting rounds.
            early_stopping_rounds: Early-stopping patience (val NDCG).

        Returns:
            Metrics dict with keys: ``mean_ndcg_at_10``, ``std_ndcg_at_10``,
            ``best_iteration``, ``num_train_rows``, ``num_features``,
            ``num_query_groups``, ``seed_results``.

        Raises:
            ImportError: If ``lightgbm`` is not installed.
        """
        if not LGBM_AVAILABLE:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")

        if feature_extractor is None:
            feature_extractor = self._get_feature_extractor()

        if seeds is None:
            seeds = EVAL_SEEDS

        n_groups = grouped_df["query_id"].nunique()
        logger.info("Starting LightGBM training: %d rows, %d query groups", len(grouped_df), n_groups)

        # Build full feature matrix to discover feature columns
        X_all, y_all, groups_all = self._build_rank_data(grouped_df, feature_extractor)
        self.feature_cols = X_all.columns.tolist()

        unique_qids = np.array(sorted(grouped_df["query_id"].unique()))

        # --- Primary training run (fixed seed=42) ---
        rng = np.random.default_rng(42)
        shuffled = unique_qids.copy()
        rng.shuffle(shuffled)

        has_holdout = len(unique_qids) >= 5
        if has_holdout:
            n_val = max(1, int(len(unique_qids) * 0.2))
            val_qids = set(shuffled[:n_val].tolist())
            train_qids = set(shuffled[n_val:].tolist())

            train_part = grouped_df[grouped_df["query_id"].isin(train_qids)].reset_index(drop=True)
            val_part = grouped_df[grouped_df["query_id"].isin(val_qids)].reset_index(drop=True)

            X_train, y_train, g_train = self._build_rank_data(train_part, feature_extractor)
            X_val, y_val, g_val = self._build_rank_data(val_part, feature_extractor)
            X_train = X_train.reindex(columns=self.feature_cols, fill_value=0)
            X_val = X_val.reindex(columns=self.feature_cols, fill_value=0)
        else:
            logger.warning("Fewer than 5 query groups — training without holdout split.")
            X_train, y_train, g_train = X_all, y_all, groups_all
            X_val, y_val, g_val = None, None, None

        lgb_train = lgb.Dataset(X_train.values, label=y_train.values, group=g_train)
        train_params = {**self.params, "seed": 42}

        if X_val is not None:
            lgb_val = lgb.Dataset(X_val.values, label=y_val.values, group=g_val, reference=lgb_train)
            self.model = lgb.train(
                train_params,
                lgb_train,
                num_boost_round=num_boost_rounds,
                valid_sets=[lgb_val],
                valid_names=["valid"],
                callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
            )
        else:
            self.model = lgb.train(train_params, lgb_train, num_boost_round=min(200, num_boost_rounds))

        best_iter = int(getattr(self.model, "best_iteration", None) or num_boost_rounds)

        # --- Multi-seed stability evaluation ---
        seed_results: list[dict[str, Any]] = []
        for seed in seeds:
            rng_s = np.random.default_rng(seed)
            qids_s = unique_qids.copy()
            rng_s.shuffle(qids_s)

            if len(unique_qids) < 5:
                X_eval_s, y_eval_s, g_eval_s = X_all, y_all, groups_all
            else:
                n_val_s = max(1, int(len(qids_s) * 0.2))
                val_part_s = grouped_df[grouped_df["query_id"].isin(set(qids_s[:n_val_s].tolist()))].reset_index(
                    drop=True
                )
                X_eval_s, y_eval_s, g_eval_s = self._build_rank_data(val_part_s, feature_extractor)
                X_eval_s = X_eval_s.reindex(columns=self.feature_cols, fill_value=0)

            mean_nd, med_nd = self._ndcg_by_group(X_eval_s, y_eval_s, g_eval_s)
            seed_results.append({"seed": seed, "mean_ndcg_at_10": mean_nd, "median_ndcg_at_10": med_nd})
            logger.info("Seed %d: mean NDCG@10=%.4f", seed, mean_nd)

        mean_ndcg = float(np.mean([r["mean_ndcg_at_10"] for r in seed_results]))
        std_ndcg = float(np.std([r["mean_ndcg_at_10"] for r in seed_results]))

        metrics: dict[str, Any] = {
            "mean_ndcg_at_10": mean_ndcg,
            "std_ndcg_at_10": std_ndcg,
            "best_iteration": best_iter,
            "num_train_rows": int(X_train.shape[0]),
            "num_features": len(self.feature_cols),
            "num_query_groups": int(len(unique_qids)),
            "seed_results": seed_results,
        }

        logger.info("Training complete. Mean NDCG@10=%.4f ± %.4f", mean_ndcg, std_ndcg)

        # --- MLflow logging ---
        if tracker is not None:
            loggable_params: dict[str, Any] = {
                k: v for k, v in self.params.items() if not isinstance(v, list)
            }
            loggable_params["num_boost_rounds"] = num_boost_rounds
            loggable_params["early_stopping_rounds"] = early_stopping_rounds
            loggable_params["num_features"] = len(self.feature_cols)
            if dataset_version is not None:
                loggable_params["dataset_version"] = dataset_version
            tracker.log_params(loggable_params)

            tracker.log_metrics(
                {
                    "mean_ndcg_at_10": mean_ndcg,
                    "std_ndcg_at_10": std_ndcg,
                    "best_iteration": float(best_iter),
                    "num_train_rows": float(X_train.shape[0]),
                    "num_query_groups": float(len(unique_qids)),
                }
            )

        return metrics

    def predict(self, df: pd.DataFrame, query_text: str = "", feature_extractor=None) -> np.ndarray:
        """Score candidate repositories for a given query.

        Args:
            df: DataFrame of candidate repos (raw repo columns).
            query_text: Query string for query-dependent features.
            feature_extractor: ``FeatureExtractor`` instance (lazy-loaded if None).

        Returns:
            1-D array of ranking scores — higher means more relevant.

        Raises:
            RuntimeError: If called before :meth:`train` or :meth:`load`.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        if feature_extractor is None:
            feature_extractor = self._get_feature_extractor()

        feats = feature_extractor.extract_all(df, query=query_text)
        feats = feats.reindex(columns=self.feature_cols, fill_value=0)
        return self.model.predict(feats.values)

    def save(self, path: str | Path) -> str:
        """Persist model + feature columns + hyperparams to disk via joblib.

        Args:
            path: Destination file path (e.g. ``/app/models/lgbm_v2.pkl``).

        Returns:
            Absolute path string of the saved file.

        Raises:
            RuntimeError: If called before training.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Call train() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": self.model, "feature_cols": self.feature_cols, "params": self.params}
        joblib.dump(payload, path)
        logger.info("LGBMRanker saved to %s", path)
        return str(path)

    @classmethod
    def load(cls, path: str | Path) -> "LGBMRanker":
        """Load a saved :class:`LGBMRanker` from disk.

        Args:
            path: Path to a ``.pkl`` file produced by :meth:`save`.

        Returns:
            Fully initialised :class:`LGBMRanker` ready for :meth:`predict`.
        """
        payload = joblib.load(path)
        ranker = cls(params=payload.get("params"))
        ranker.model = payload["model"]
        ranker.feature_cols = payload["feature_cols"]
        logger.info("LGBMRanker loaded from %s (%d features)", path, len(ranker.feature_cols))
        return ranker

    def save_registry_entry(self, model_path: str | Path, output_dir: str | Path) -> str:
        """Write a JSON registry entry for downstream model tracking.

        Args:
            model_path: Path where the model ``.pkl`` was saved.
            output_dir: Directory to write ``lgbm_registry_latest.json``.

        Returns:
            Path to the written registry file.
        """
        ts = int(time.time())
        registry: dict[str, Any] = {
            "model_id": f"lightgbm_{ts}",
            "model_type": "reranker",
            "variant": "lightgbm_lambdarank",
            "path": str(model_path),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feature_cols": self.feature_cols,
            "params": self.params,
        }

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        registry_path = out_dir / "lgbm_registry_latest.json"

        with open(registry_path, "w") as f:
            json.dump(registry, f, indent=2)

        logger.info("Registry entry written to %s", registry_path)
        return str(registry_path)
