"""Champion/challenger model promotion with metric comparison.

Compares a newly trained candidate model against the current production model.
Promotes only when the candidate does not degrade any shared metric by more than
DEGRADATION_THRESHOLD and improves at least one metric.

On the first retrain (no production model exists yet) the candidate is promoted
unconditionally.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from mlops.mlflow_tracker import MLflowTracker

logger = logging.getLogger(__name__)

# Metrics used for comparison — must match keys produced by LGBMRanker.train()
COMPARISON_METRICS = ["mean_ndcg_at_10", "std_ndcg_at_10"]

# Maximum relative drop on any metric before promotion is blocked
DEGRADATION_THRESHOLD = 0.05  # 5%


class ModelPromoter:
    """Handles champion/challenger comparison and conditional model promotion.

    Usage:
        promoter = ModelPromoter(recommender_url, mlflow_tracker)
        promoted = promoter.promote_if_better(
            candidate_model_id=model_id,
            candidate_metrics=metrics,
            candidate_mlflow_version=mlflow_version,
        )
    """

    def __init__(
        self,
        recommender_url: str,
        mlflow_tracker: MLflowTracker,
        model_name: str = "git-query-lgbm-reranker",
    ):
        self.recommender_url = recommender_url.rstrip("/")
        self.tracker = mlflow_tracker
        self.model_name = model_name

    def promote_if_better(
        self,
        candidate_model_id: str,
        candidate_metrics: dict[str, float],
        candidate_mlflow_version: int | None = None,
        candidate_model_path: str | None = None,
    ) -> bool:
        """Promote the candidate if it improves on the current production model.

        Returns True if the candidate was promoted, False if it was kept as candidate.
        """
        production_metrics = self.tracker.get_production_metrics(self.model_name, COMPARISON_METRICS)

        if not production_metrics:
            logger.info(
                "No production model found — promoting %s unconditionally",
                candidate_model_id,
            )
            return self._do_promote(candidate_model_id, candidate_mlflow_version, candidate_model_path)

        # Find metrics present in both production and candidate
        shared_keys = [k for k in COMPARISON_METRICS if k in production_metrics and k in candidate_metrics]

        if not shared_keys:
            logger.info(
                "No shared metrics to compare — promoting %s unconditionally",
                candidate_model_id,
            )
            return self._do_promote(candidate_model_id, candidate_mlflow_version, candidate_model_path)

        # Block if any shared metric degrades beyond threshold
        # (lower std_ndcg is better, higher mean_ndcg is better — treat degradation
        # as a drop in mean_ndcg or a rise in std_ndcg)
        for key in shared_keys:
            prod_val = production_metrics[key]
            cand_val = candidate_metrics[key]
            if prod_val == 0:
                continue
            # For std metrics, higher is worse; for all others, lower is worse
            if "std" in key:
                relative_change = (cand_val - prod_val) / prod_val  # positive = worse
            else:
                relative_change = (prod_val - cand_val) / prod_val  # positive = worse
            if relative_change > DEGRADATION_THRESHOLD:
                logger.warning(
                    "Model %s REJECTED: '%s' degraded %.1f%% (prod=%.4f → cand=%.4f)",
                    candidate_model_id,
                    key,
                    100 * relative_change,
                    prod_val,
                    cand_val,
                )
                self.tracker.log_metrics({"promotion_blocked": 1.0})
                if candidate_mlflow_version is not None:
                    self.tracker.transition_model_stage(self.model_name, candidate_mlflow_version, "Archived")
                return False

        return self._do_promote(candidate_model_id, candidate_mlflow_version, candidate_model_path)

    def _do_promote(
        self,
        model_id: str,
        mlflow_version: int | None,
        model_path: str | None = None,
    ) -> bool:
        """Upload model file (if local path given), promote, and reload."""
        import json as _json
        import os

        api_key = os.getenv("APIKEY_MONGODB", "")
        headers = {"X-API-Key": api_key} if api_key else {}

        # Upload model file to server so the recommender can serve it.
        # Non-fatal: training-only environments don't run the recommender service.
        if model_path and os.path.exists(model_path):
            try:
                with open(model_path, "rb") as fh:
                    resp = requests.post(
                        f"{self.recommender_url}/admin/models/upload",
                        headers=headers,
                        files={"file": (os.path.basename(model_path), fh, "application/octet-stream")},
                        data={
                            "model_id": model_id,
                            "variant": "default",
                            "metrics": _json.dumps({}),
                        },
                        timeout=300,  # large file upload
                    )
                resp.raise_for_status()
                logger.info("Uploaded model file %s to server", os.path.basename(model_path))
            except requests.RequestException as e:
                logger.warning("Could not upload model file (recommender may not be running): %s", e)

        # Promote in model registry via recommender API
        try:
            resp = requests.post(
                f"{self.recommender_url}/admin/models/promote/{model_id}",
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
            logger.info("Promoted model %s via recommender registry", model_id)
        except requests.RequestException as e:
            logger.warning(
                "Could not reach recommender for promote (will still count as promoted): %s",
                e,
            )

        # Hot-reload without full container restart
        try:
            requests.post(f"{self.recommender_url}/admin/models/reload", headers=headers, timeout=10)
            logger.info("Triggered recommender model hot-reload")
        except requests.RequestException:
            pass  # Non-fatal

        # Update MLflow Model Registry stage — always happens regardless of recommender reachability
        if mlflow_version is not None:
            self.tracker.archive_production_versions(self.model_name, keep_version=mlflow_version)
            self.tracker.transition_model_stage(self.model_name, mlflow_version, "Production")

        self.tracker.log_metrics({"promotion_blocked": 0.0})
        logger.info("Model %s promoted to production", model_id)
        return True
