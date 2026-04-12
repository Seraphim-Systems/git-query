"""MLflow experiment tracking wrapper for the recommendation system."""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Check if MLflow is available
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Tracking will be disabled.")


class MLflowTracker:
    """
    Wrapper for MLflow experiment tracking.

    Provides a simple interface for logging:
    - Hyperparameters (params)
    - Metrics (precision@k, ndcg, etc.)
    - Model artifacts
    - Training metadata

    Usage:
        tracker = MLflowTracker(experiment_name="git-query-recommender")
        with tracker.start_run(run_name="embedding-training-v1"):
            tracker.log_params({"model_name": "all-MiniLM-L6-v2", "batch_size": 32})
            tracker.log_metrics({"precision_at_5": 0.85, "ndcg_at_10": 0.72})
            tracker.log_artifact("/path/to/model")
    """

    def __init__(
        self,
        experiment_name: str = "git-query-recommender",
        tracking_uri: str | None = None,
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI. Defaults to MLFLOW_TRACKING_URI env var.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "mlruns")
        self._run = None
        self._client = None

        if MLFLOW_AVAILABLE:
            self._setup_mlflow()
        else:
            logger.warning("MLflow not available. All tracking operations will be no-ops.")

    def _setup_mlflow(self):
        """Configure MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._client = MlflowClient(self.tracking_uri)
            logger.info(f"MLflow configured: experiment='{self.experiment_name}', uri='{self.tracking_uri}'")
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
            raise

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        nested: bool = False,
    ):
        """
        Start an MLflow run as a context manager.

        Args:
            run_name: Name for this run
            tags: Tags to attach to the run
            nested: Whether this is a nested run

        Yields:
            The MLflow run object (or None if MLflow unavailable)
        """
        if not MLFLOW_AVAILABLE:
            yield None
            return

        try:
            self._run = mlflow.start_run(run_name=run_name, nested=nested)

            if tags:
                mlflow.set_tags(tags)

            logger.info(f"Started MLflow run: {run_name or self._run.info.run_id}")
            yield self._run

        except Exception as e:
            logger.error(f"Error in MLflow run: {e}")
            raise
        finally:
            if self._run:
                mlflow.end_run()
                logger.info("MLflow run ended")
                self._run = None

    def log_params(self, params: dict[str, Any]):
        """
        Log hyperparameters.

        Args:
            params: Dictionary of parameter names and values
        """
        if not MLFLOW_AVAILABLE or not self._run:
            logger.debug(f"Skipping log_params (MLflow unavailable or no active run): {params}")
            return

        try:
            # MLflow params must be strings, convert values
            str_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(str_params)
            logger.debug(f"Logged params: {list(params.keys())}")
        except Exception as e:
            logger.warning(f"Failed to log params: {e}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metric names and values
            step: Step number for tracking metrics over time
        """
        if not MLFLOW_AVAILABLE or not self._run:
            logger.debug(f"Skipping log_metrics (MLflow unavailable or no active run): {metrics}")
            return

        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged metrics: {list(metrics.keys())}")
        except Exception as e:
            logger.warning(f"Failed to log metrics: {e}")

    def log_metric(self, key: str, value: float, step: int | None = None):
        """
        Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Step number
        """
        self.log_metrics({key: value}, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None):
        """
        Log an artifact (file or directory).

        Args:
            local_path: Path to the local file or directory
            artifact_path: Destination path within the artifact directory
        """
        if not MLFLOW_AVAILABLE or not self._run:
            logger.debug(f"Skipping log_artifact (MLflow unavailable or no active run): {local_path}")
            return

        try:
            if Path(local_path).is_dir():
                mlflow.log_artifacts(local_path, artifact_path)
            else:
                mlflow.log_artifact(local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.warning(f"Failed to log artifact: {e}")

    def log_model_info(self, model_metadata: dict[str, Any]):
        """
        Log model-related information as params and metrics.

        Args:
            model_metadata: Dictionary containing model information like:
                - model_name
                - embedding_dim
                - num_repos
                - training_time_seconds
        """
        # Separate into params and metrics
        params = {}
        metrics = {}

        param_keys = ["model_name", "embedding_dim", "device", "normalized", "batch_size"]
        metric_keys = ["num_repos", "training_time_seconds"]

        for key, value in model_metadata.items():
            if key in param_keys:
                params[key] = value
            elif key in metric_keys:
                try:
                    metrics[key] = float(value)
                except (TypeError, ValueError):
                    params[key] = value
            elif key != "timestamp":  # Skip timestamp
                params[key] = value

        if params:
            self.log_params(params)
        if metrics:
            self.log_metrics(metrics)

    def log_evaluation_metrics(self, eval_metrics: dict[str, Any]):
        """
        Log evaluation metrics from the RecommenderEvaluator.

        Args:
            eval_metrics: Dictionary containing:
                - precision_at_k: {k: value}
                - recall_at_k: {k: value}
                - ndcg_at_k: {k: value}
                - mrr: float
        """
        flat_metrics = {}

        # Flatten nested metrics
        for metric_name, values in eval_metrics.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    flat_metrics[f"{metric_name}_{k}"] = float(v)
            else:
                flat_metrics[metric_name] = float(values)

        self.log_metrics(flat_metrics)

    def set_tag(self, key: str, value: str):
        """
        Set a tag on the current run.

        Args:
            key: Tag name
            value: Tag value
        """
        if not MLFLOW_AVAILABLE or not self._run:
            return

        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            logger.warning(f"Failed to set tag: {e}")

    def register_model_version(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str,
    ) -> int | None:
        """Register a model artifact in the MLflow Model Registry and transition to Staging.

        Uses the MlflowClient low-level API so that plain artifacts logged via
        log_artifact() can be registered without requiring an MLmodel manifest
        (which mlflow.register_model() requires from MLflow 2.9+).

        Returns the registered version number, or None if unavailable.
        """
        if not MLFLOW_AVAILABLE or not self._client:
            return None
        try:
            # Ensure the registered model exists
            try:
                self._client.create_registered_model(model_name)
                logger.info("Created registered model '%s'", model_name)
            except Exception:
                pass  # Already exists — that's fine

            # Build the source URI from the run's artifact root
            run_info = self._client.get_run(run_id)
            artifact_uri = run_info.info.artifact_uri.rstrip("/")
            source = f"{artifact_uri}/{artifact_path}"

            mv = self._client.create_model_version(
                name=model_name,
                source=source,
                run_id=run_id,
            )
            version = int(mv.version)
            self._client.transition_model_version_stage(name=model_name, version=str(version), stage="Staging")
            logger.info("Registered model '%s' version %d → Staging", model_name, version)
            return version
        except Exception as e:
            logger.warning("Failed to register model version: %s", e)
            return None

    def transition_model_stage(self, model_name: str, version: int, stage: str) -> None:
        """Transition a model version to a new stage (Staging / Production / Archived)."""
        if not MLFLOW_AVAILABLE or not self._client:
            return
        try:
            self._client.transition_model_version_stage(name=model_name, version=str(version), stage=stage)
            logger.info("Model '%s' version %d → %s", model_name, version, stage)
        except Exception as e:
            logger.warning("Failed to transition model stage: %s", e)

    def get_production_metrics(self, model_name: str, metric_keys: list[str]) -> dict[str, float]:
        """Return metrics from the current Production version of a registered model.

        Returns an empty dict when no Production version exists yet (first run).
        """
        if not MLFLOW_AVAILABLE or not self._client:
            return {}
        try:
            versions = self._client.get_latest_versions(model_name, stages=["Production"])
            if not versions:
                return {}
            run_data = self._client.get_run(versions[0].run_id).data
            return {k: run_data.metrics[k] for k in metric_keys if k in run_data.metrics}
        except Exception as e:
            logger.warning("Failed to fetch production metrics for '%s': %s", model_name, e)
            return {}

    def get_run_id(self) -> str | None:
        """Get the current run ID."""
        if self._run:
            return self._run.info.run_id
        return None

    def get_experiment_id(self) -> str | None:
        """Get the current experiment ID."""
        if MLFLOW_AVAILABLE:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                return experiment.experiment_id
        return None


def create_tracker_from_env() -> MLflowTracker:
    """
    Create an MLflowTracker from environment variables.

    Environment variables:
        MLFLOW_TRACKING_URI: MLflow server URI
        MLFLOW_EXPERIMENT_NAME: Experiment name (default: git-query-recommender)

    Returns:
        Configured MLflowTracker instance
    """
    return MLflowTracker(
        experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "git-query-recommender"),
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
    )
