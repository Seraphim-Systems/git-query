# MLflow Server Deployment (Docker)

This project already logs training metadata through `MLflowTracker`. The missing piece for server deployments is running an always-on MLflow tracking server and pointing training/app containers at it.

This guide uses the existing compose fragments in `infrastructure/docker`.

## What this setup gives you

- Persistent MLflow tracking UI/API on your server
- Persistent Model Registry state (`sqlite` backend)
- Persistent artifact storage for trained model files
- Training runs that log to the same central MLflow instance

## 1) Configure environment

Use `infrastructure/docker/.env` and set these values:

```dotenv
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_PORT=5000
MLFLOW_EXPERIMENT_NAME=git-query-retrain
MLFLOW_BACKEND_STORE_URI=sqlite:////mlflow/mlflow.db
MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
```

Notes:

- `MLFLOW_TRACKING_URI` is the in-network URL used by containers.
- `MLFLOW_PORT` is only for exposing MLflow UI/API on the host.
- If port `5000` is used by another service, set a different `MLFLOW_PORT`.

## 2) Start app + databases + MLflow on server

From `infrastructure/docker`:

```bash
docker compose \
  -f docker-compose.base.yml \
  -f docker-compose.databases.yml \
  -f docker-compose.app.yml \
  -f docker-compose.mlflow.yml \
  -f docker-compose.prod.yml \
  up -d --build
```

Then verify:

```bash
docker compose -f docker-compose.base.yml -f docker-compose.mlflow.yml ps mlflow
```

MLflow UI/API will be available at:

- `http://<server-host>:${MLFLOW_PORT}`

## 3) Run retraining that logs to server MLflow

From `infrastructure/docker`:

```bash
docker compose \
  -f docker-compose.base.yml \
  -f docker-compose.databases.yml \
  -f docker-compose.app.yml \
  -f docker-compose.mlflow.yml \
  -f docker-compose.training.yml \
  up --build training
```

The training service now defaults to `MLFLOW_TRACKING_URI=http://mlflow:5000`, so runs/metrics/artifacts and model version transitions are logged to the central server.

## 4) Persisted data

MLflow data is persisted in Docker named volumes:

- `git-query-mlflow-runs` (metadata DB and run state)
- `git-query-mlflow-artifacts` (model/artifact files)

To back up:

- back up both volumes, or
- move to bind mounts or object storage (S3/Azure Blob/MinIO) later.

## 5) Operations tips

- Keep MLflow internal if possible and expose via reverse proxy with auth/TLS.
- For multi-node scaling, replace sqlite with Postgres/MySQL backend store.
- For large artifacts, use object storage for `MLFLOW_DEFAULT_ARTIFACT_ROOT`.
