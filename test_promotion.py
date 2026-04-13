"""Manual test script for ModelPromoter + MLflow Model Registry.

Run this twice:
  1st run: no production model exists → promotes unconditionally
  2nd run (better metrics): promotes (improvement detected)
  2nd run (worse metrics): blocks promotion

Usage:
    python test_promotion.py --ndcg 0.45           # first run / better model
    python test_promotion.py --ndcg 0.30           # worse model — blocked
    python test_promotion.py --ndcg 0.50           # better model — promoted
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from recommender.mlops.mlflow_tracker import MLflowTracker
from recommender.mlops.model_promoter import ModelPromoter

MLFLOW_URI = "file:///tmp/test_mlruns"
EXPERIMENT = "git-query-promotion-test"
MODEL_NAME = "git-query-lgbm-reranker"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndcg", type=float, default=0.45, help="mean_ndcg_at_10 to simulate")
    parser.add_argument("--recommender-url", default="http://localhost:8095")
    args = parser.parse_args()

    tracker = MLflowTracker(
        experiment_name=EXPERIMENT,
        tracking_uri=MLFLOW_URI,
    )

    candidate_metrics = {
        "mean_ndcg_at_10": args.ndcg,
        "std_ndcg_at_10": 0.02,
        "num_train_rows": 5000.0,
        "num_query_groups": 100.0,
    }

    print(f"\nSimulating retrain with mean_ndcg_at_10={args.ndcg}")
    print(f"MLflow tracking: {MLFLOW_URI}")
    print(f"Experiment: {EXPERIMENT}\n")

    with tracker.start_run(run_name=f"test-retrain-ndcg{args.ndcg}"):
        tracker.log_params({"num_boost_rounds": "300", "variant": "default"})
        tracker.log_metrics(candidate_metrics)

        run_id = tracker.get_run_id()
        print(f"Run ID: {run_id}")

        # Register a dummy model artifact (empty file stands in for the .pkl)
        dummy_model = Path("/tmp/dummy_lgbm_model.pkl")
        dummy_model.write_bytes(b"dummy")
        tracker.log_artifact(str(dummy_model), "models")

        mlflow_version = tracker.register_model_version(
            run_id=run_id,
            model_name=MODEL_NAME,
            artifact_path=f"models/{dummy_model.name}",
        )
        print(f"MLflow version: {mlflow_version} (Staging)")

        promoter = ModelPromoter(
            recommender_url=args.recommender_url,
            mlflow_tracker=tracker,
        )
        promoted = promoter.promote_if_better(
            candidate_model_id=f"lgbm_default_test_{run_id[:8]}",
            candidate_metrics=candidate_metrics,
            candidate_mlflow_version=mlflow_version,
        )

    status = "PROMOTED to Production" if promoted else "BLOCKED (kept as Staging)"
    print(f"\nResult: {status}")
    print(f"\nView in MLflow UI:")
    print(f"  mlflow ui --backend-store-uri {MLFLOW_URI}")
    print(f"  Then open: http://localhost:5000")


if __name__ == "__main__":
    main()
