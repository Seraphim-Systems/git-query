"""Data and model drift monitoring using Evidently AI.

This module provides drift detection for:
1. Data Drift: Changes in repository metadata distributions
2. Embedding Drift: Shifts in embedding space
3. Prediction Drift: Changes in recommendation score distributions
4. Target Drift: Changes in user engagement (CTR, ratings)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Check if Evidently is available
try:
    import pandas as pd
    from evidently.legacy.base_metric import ColumnMapping
    from evidently.legacy.metric_preset import TargetDriftPreset
    from evidently.legacy.metrics import (
        ColumnDriftMetric,
        DataDriftTable,
        DatasetDriftMetric,
    )
    from evidently.legacy.report import Report

    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("Evidently not installed. Drift monitoring will be disabled.")


class DriftMonitor:
    """
    Monitor for data and model drift in the recommendation system.

    Compares current data against reference (training) data to detect:
    - Distribution shifts in repository features
    - Embedding space drift
    - Changes in prediction patterns
    - User engagement trend changes

    Usage:
        monitor = DriftMonitor(report_dir="./drift_reports")

        # Compare current vs reference data
        report = monitor.check_data_drift(
            reference_data=training_df,
            current_data=production_df
        )

        # Save report
        monitor.save_report(report, "data_drift_report")
    """

    def __init__(
        self,
        report_dir: str = "/app/drift_reports",
        reference_data_path: str | None = None,
    ):
        """
        Initialize drift monitor.

        Args:
            report_dir: Directory to save drift reports
            reference_data_path: Path to reference (training) data
        """
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.reference_data_path = reference_data_path

        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available. Drift checks will return empty results.")

    def check_data_drift(
        self,
        reference_data: "pd.DataFrame",
        current_data: "pd.DataFrame",
        column_mapping: Optional["ColumnMapping"] = None,
    ) -> dict[str, Any] | None:
        """
        Check for data drift between reference and current data.

        Args:
            reference_data: DataFrame with training/reference data
            current_data: DataFrame with current/production data
            column_mapping: Evidently column mapping configuration

        Returns:
            Dictionary with drift detection results
        """
        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available, skipping data drift check")
            return None

        try:
            # Drop columns containing arrays/lists — Evidently can't process them
            def _is_scalar_col(series):
                sample = series.dropna().head(10)
                return not any(isinstance(v, (list, np.ndarray)) for v in sample)

            scalar_cols = [
                c for c in reference_data.columns
                if _is_scalar_col(reference_data[c]) and reference_data[c].notna().any()
            ]
            reference_data = reference_data[scalar_cols]
            current_data = current_data[scalar_cols]

            report = Report(
                metrics=[
                    DatasetDriftMetric(),
                    DataDriftTable(),
                ]
            )

            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping,
            )

            # Extract results
            result = report.as_dict()

            drift_info = {
                "timestamp": datetime.now().isoformat(),
                "type": "data_drift",
                "reference_rows": len(reference_data),
                "current_rows": len(current_data),
                "drift_detected": self._extract_drift_status(result),
                "metrics": result.get("metrics", []),
            }

            logger.info(f"Data drift check complete. Drift detected: {drift_info['drift_detected']}")
            return drift_info

        except Exception as e:
            logger.error(f"Error checking data drift: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def check_embedding_drift(
        self,
        reference_embeddings: np.ndarray,
        current_embeddings: np.ndarray,
        sample_size: int = 1000,
    ) -> dict[str, Any] | None:
        """
        Check for drift in embedding space.

        Uses statistical methods to detect if embedding distributions have shifted.

        Args:
            reference_embeddings: Reference embedding vectors (N x D)
            current_embeddings: Current embedding vectors (M x D)
            sample_size: Number of samples for drift calculation

        Returns:
            Dictionary with embedding drift metrics
        """
        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available, skipping embedding drift check")
            return None

        try:
            # Sample if too large
            if len(reference_embeddings) > sample_size:
                idx = np.random.choice(len(reference_embeddings), sample_size, replace=False)
                reference_embeddings = reference_embeddings[idx]

            if len(current_embeddings) > sample_size:
                idx = np.random.choice(len(current_embeddings), sample_size, replace=False)
                current_embeddings = current_embeddings[idx]

            # Create DataFrames for Evidently
            ref_df = pd.DataFrame(
                reference_embeddings, columns=[f"emb_{i}" for i in range(reference_embeddings.shape[1])]
            )
            cur_df = pd.DataFrame(current_embeddings, columns=[f"emb_{i}" for i in range(current_embeddings.shape[1])])

            # Run drift check on embedding dimensions
            report = Report(
                metrics=[
                    DatasetDriftMetric(),
                ]
            )

            report.run(reference_data=ref_df, current_data=cur_df)
            result = report.as_dict()

            # Calculate additional metrics
            ref_norms = np.linalg.norm(reference_embeddings, axis=1)
            cur_norms = np.linalg.norm(current_embeddings, axis=1)

            drift_info = {
                "timestamp": datetime.now().isoformat(),
                "type": "embedding_drift",
                "reference_samples": len(reference_embeddings),
                "current_samples": len(current_embeddings),
                "embedding_dim": reference_embeddings.shape[1],
                "drift_detected": self._extract_drift_status(result),
                "norm_stats": {
                    "reference_mean_norm": float(np.mean(ref_norms)),
                    "current_mean_norm": float(np.mean(cur_norms)),
                    "norm_shift": float(np.mean(cur_norms) - np.mean(ref_norms)),
                },
            }

            logger.info(f"Embedding drift check complete. Drift detected: {drift_info['drift_detected']}")
            return drift_info

        except Exception as e:
            logger.error(f"Error checking embedding drift: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def check_prediction_drift(
        self,
        reference_scores: list[float],
        current_scores: list[float],
    ) -> dict[str, Any] | None:
        """
        Check for drift in recommendation scores.

        Args:
            reference_scores: Historical recommendation scores
            current_scores: Recent recommendation scores

        Returns:
            Dictionary with prediction drift metrics
        """
        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available, skipping prediction drift check")
            return None

        try:
            ref_df = pd.DataFrame({"score": reference_scores})
            cur_df = pd.DataFrame({"score": current_scores})

            report = Report(
                metrics=[
                    ColumnDriftMetric(column_name="score"),
                ]
            )

            report.run(reference_data=ref_df, current_data=cur_df)
            result = report.as_dict()

            drift_info = {
                "timestamp": datetime.now().isoformat(),
                "type": "prediction_drift",
                "reference_count": len(reference_scores),
                "current_count": len(current_scores),
                "reference_mean": float(np.mean(reference_scores)),
                "current_mean": float(np.mean(current_scores)),
                "drift_detected": self._extract_drift_status(result),
            }

            logger.info(f"Prediction drift check complete. Drift detected: {drift_info['drift_detected']}")
            return drift_info

        except Exception as e:
            logger.error(f"Error checking prediction drift: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def check_target_drift(
        self,
        reference_interactions: "pd.DataFrame",
        current_interactions: "pd.DataFrame",
        target_column: str = "clicked",
    ) -> dict[str, Any] | None:
        """
        Check for drift in user engagement (target variable).

        Args:
            reference_interactions: Historical user interactions
            current_interactions: Recent user interactions
            target_column: Name of the target column

        Returns:
            Dictionary with target drift metrics
        """
        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available, skipping target drift check")
            return None

        try:
            column_mapping = ColumnMapping(target=target_column)

            report = Report(metrics=[TargetDriftPreset()])

            report.run(
                reference_data=reference_interactions,
                current_data=current_interactions,
                column_mapping=column_mapping,
            )
            result = report.as_dict()

            # Calculate CTR
            ref_ctr = reference_interactions[target_column].mean() if target_column in reference_interactions else 0
            cur_ctr = current_interactions[target_column].mean() if target_column in current_interactions else 0

            drift_info = {
                "timestamp": datetime.now().isoformat(),
                "type": "target_drift",
                "reference_ctr": float(ref_ctr),
                "current_ctr": float(cur_ctr),
                "ctr_change": float(cur_ctr - ref_ctr),
                "drift_detected": self._extract_drift_status(result),
            }

            logger.info(f"Target drift check complete. CTR change: {drift_info['ctr_change']:.4f}")
            return drift_info

        except Exception as e:
            logger.error(f"Error checking target drift: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def _extract_drift_status(self, result: dict[str, Any]) -> bool:
        """Extract overall drift detection status from Evidently result."""
        try:
            metrics = result.get("metrics", [])
            for metric in metrics:
                metric_result = metric.get("result", {})
                if metric_result.get("dataset_drift", False):
                    return True
                if metric_result.get("drift_detected", False):
                    return True
            return False
        except Exception:
            return False

    def save_report(
        self,
        drift_info: dict[str, Any],
        report_name: str,
        format: str = "json",
    ) -> str:
        """
        Save drift report to file.

        Args:
            drift_info: Drift detection results
            report_name: Base name for the report file
            format: Output format ('json' or 'html')

        Returns:
            Path to saved report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_name}_{timestamp}.{format}"
        filepath = self.report_dir / filename

        if format == "json":
            with open(filepath, "w") as f:
                json.dump(drift_info, f, indent=2, default=str)
        else:
            # For HTML, we need the actual Evidently report object
            logger.warning("HTML export requires Evidently report object, saving as JSON instead")
            filepath = self.report_dir / f"{report_name}_{timestamp}.json"
            with open(filepath, "w") as f:
                json.dump(drift_info, f, indent=2, default=str)

        logger.info(f"Drift report saved to: {filepath}")
        return str(filepath)

    def run_full_drift_check(
        self,
        reference_data: Optional["pd.DataFrame"] = None,
        current_data: Optional["pd.DataFrame"] = None,
        reference_embeddings: np.ndarray | None = None,
        current_embeddings: np.ndarray | None = None,
        reference_scores: list[float] | None = None,
        current_scores: list[float] | None = None,
    ) -> dict[str, Any]:
        """
        Run all applicable drift checks.

        Args:
            reference_data: Reference repository data
            current_data: Current repository data
            reference_embeddings: Reference embedding vectors
            current_embeddings: Current embedding vectors
            reference_scores: Reference prediction scores
            current_scores: Current prediction scores

        Returns:
            Combined drift report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "overall_drift_detected": False,
        }

        # Data drift
        if reference_data is not None and current_data is not None:
            data_drift = self.check_data_drift(reference_data, current_data)
            if data_drift:
                report["checks"]["data_drift"] = data_drift
                if data_drift.get("drift_detected"):
                    report["overall_drift_detected"] = True

        # Embedding drift
        if reference_embeddings is not None and current_embeddings is not None:
            emb_drift = self.check_embedding_drift(reference_embeddings, current_embeddings)
            if emb_drift:
                report["checks"]["embedding_drift"] = emb_drift
                if emb_drift.get("drift_detected"):
                    report["overall_drift_detected"] = True

        # Prediction drift
        if reference_scores is not None and current_scores is not None:
            pred_drift = self.check_prediction_drift(reference_scores, current_scores)
            if pred_drift:
                report["checks"]["prediction_drift"] = pred_drift
                if pred_drift.get("drift_detected"):
                    report["overall_drift_detected"] = True

        # Save combined report
        self.save_report(report, "full_drift_report")

        return report


def create_monitor_from_env() -> DriftMonitor:
    """
    Create a DriftMonitor from environment variables.

    Environment variables:
        EVIDENTLY_REPORT_PATH: Path to save drift reports

    Returns:
        Configured DriftMonitor instance
    """
    return DriftMonitor(
        report_dir=os.getenv("EVIDENTLY_REPORT_PATH", "/app/drift_reports"),
    )


if __name__ == "__main__":
    # Example usage / CLI entry point
    logging.basicConfig(level=logging.INFO)

    monitor = create_monitor_from_env()

    # Check if we have data to compare
    # This would typically be called from a scheduled job
    logger.info("Drift monitor initialized. Ready for drift checks.")
    logger.info(f"Reports will be saved to: {monitor.report_dir}")
