"""
Manual model registration script.

Usage:
    python3 register_model.py <path-to-model.pkl>

Example:
    python3 register_model.py /app/models/lgbm_default_20260414_123456.pkl
"""

import sys
import mlflow
from mlflow import MlflowClient

TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "git-query-lgbm-reranker"


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 register_model.py <path-to-model.pkl>")
        sys.exit(1)

    model_path = sys.argv[1]
    mlflow.set_tracking_uri(TRACKING_URI)

    print(f"Uploading {model_path} to MLflow at {TRACKING_URI}...")

    with mlflow.start_run(run_name="manual-upload") as run:
        mlflow.log_artifact(model_path, "models")
        run_id = run.info.run_id
        print(f"Logged artifact. Run ID: {run_id}")

    model_filename = model_path.split("/")[-1]
    mv = mlflow.register_model(
        f"runs:/{run_id}/models/{model_filename}",
        MODEL_NAME,
    )
    print(f"Registered as version {mv.version}")

    client = MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Production",
    )
    print(f"Model version {mv.version} promoted to Production")


if __name__ == "__main__":
    main()
