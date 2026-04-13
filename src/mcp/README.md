# MCP Server

The MCP (Model Context Protocol) server is a FastAPI application that exposes recommender-system functionality as callable tools for LLM agents. It acts as a thin routing layer between the agent/client and the underlying recommender service.

Default port: **8090** (Docker service name: `mcp-server`)

## Running

**Docker Compose (recommended)**

```sh
docker-compose \
  -f infrastructure/docker/docker-compose.base.yml \
  -f infrastructure/docker/docker-compose.databases.yml \\
  -f infrastructure/docker/docker-compose.app.yml \\
  up --build
```

**Locally (requires `.env`)**

```sh
pip install -r requirements.txt
python -m uvicorn src.mcp.server:app --host 0.0.0.0 --port 8090 --reload
```

## Endpoints

| Method | Path             | Description                                            |
| ------ | ---------------- | ------------------------------------------------------ |
| `GET`  | `/health`        | Liveness check — returns `{"status": "healthy"}`       |
| `POST` | `/tools/list`    | List all registered tools with their parameter schemas |
| `POST` | `/tools/execute` | Execute a named tool                                   |

### Execute a tool

```
POST /tools/execute
Content-Type: application/json

{
  "tool_name": "<name>",
  "parameters": { ... }
}
```

Response:

```json
{
  "success": true,
  "result": { ... },
  "error": null
}
```

## Available tools

### `recommend_repositories`

Calls `POST http://recommender:8095/recommend` and returns a ranked list of GitHub repositories.

| Parameter                | Type    | Required                | Description                                           |
| ------------------------ | ------- | ----------------------- | ----------------------------------------------------- |
| `query`                  | string  | yes                     | Natural-language description of what to find          |
| `top_k`                  | integer | no (default 10, max 50) | Number of results                                     |
| `user_id`                | string  | no                      | Enables personalised results                          |
| `language`               | string  | no                      | Filter by programming language, e.g. `"Python"`       |
| `min_stars`              | integer | no                      | Minimum GitHub stars                                  |
| `license`                | string  | no                      | Filter by licence, e.g. `"MIT"`                       |
| `max_age_days`           | integer | no                      | Exclude repos not updated in N days                   |
| `enable_personalization` | boolean | no (default true)       | Apply user-preference personalisation                 |
| `variant`                | string  | no                      | A/B variant: `baseline` \| `hybrid` \| `personalized` |

**Result shape per item:**

```json
{
  "repo_id": "owner/repo",
  "name": "repo",
  "full_name": "owner/repo",
  "description": "...",
  "language": "Python",
  "stars": 1234,
  "forks": 56,
  "url": "https://github.com/owner/repo",
  "license": "MIT",
  "score": 0.9123,
  "rank": 1,
  "explanation": {
    "sources": ["semantic", "keyword"],
    "details": "..."
  }
}
```

---

### `log_repository_interaction`

Calls `POST http://recommender:8095/interaction` to record a user's interaction with a result.

| Parameter             | Type    | Required                  | Description                                                            |
| --------------------- | ------- | ------------------------- | ---------------------------------------------------------------------- |
| `user_id`             | string  | yes                       | The user who interacted                                                |
| `query`               | string  | yes                       | The original search query                                              |
| `repo_id`             | string  | yes                       | Repository that was interacted with                                    |
| `interaction_type`    | string  | yes                       | One of: `click`, `save`, `dismiss`, `thumbs_up`, `thumbs_down`, `view` |
| `position_in_results` | integer | no                        | 0-based position in the result list                                    |
| `variant`             | string  | no (default `"baseline"`) | Which variant was shown                                                |

---

### `get_user_preferences`

Calls `GET http://recommender:8095/preferences/{user_id}` to retrieve learned preference scores.

| Parameter | Type   | Required | Description                     |
| --------- | ------ | -------- | ------------------------------- |
| `user_id` | string | yes      | The user whose profile to fetch |

---

### `query_repository_data`

Runs a read-only MongoDB query over repository-related collections.

| Parameter    | Type    | Required | Description                                                       |
| ------------ | ------- | -------- | ----------------------------------------------------------------- |
| `collection` | string  | yes      | Collection name, e.g. `repositories`, `users` |
| `filters`    | object  | no       | MongoDB filter object                                             |
| `projection` | array   | no       | List of fields to include                                         |
| `sort_by`    | string  | no       | Field name for sorting                                            |
| `sort_order` | string  | no       | `asc` or `desc` (default `desc`)                                  |
| `limit`      | integer | no       | Max number of docs to return (1-100, default 20)                  |
| `skip`       | integer | no       | Number of matched docs to skip                                    |
| `database`   | string  | no       | Optional DB name (defaults to configured Mongo database)          |

---

### `explain_repository`

Explains what a repository is about by combining MongoDB metadata and GitHub API/README data.

| Parameter                  | Type    | Required | Description                                        |
| -------------------------- | ------- | -------- | -------------------------------------------------- |
| `repo_url`                 | string  | no       | GitHub URL, e.g. `https://github.com/owner/repo`   |
| `full_name`                | string  | no       | GitHub full name, e.g. `owner/repo`                |
| `include_database_context` | boolean | no       | Include MongoDB repository metadata when available |
| `include_readme_excerpt`   | boolean | no       | Fetch and include short README excerpt from GitHub |

If no `repo_url`/`full_name` is provided, this tool falls back to explaining the local Git-Query project from `README.md`.

## Environment variables (hooks)

All settings are read from the environment (or a `.env` file at repository root).

| Variable          | Default                                       | Description                                            |
| ----------------- | --------------------------------------------- | ------------------------------------------------------ |
| `MCP_HOST`        | `0.0.0.0`                                     | Bind address for the MCP server                        |
| `MCP_PORT`        | `8000`                                        | Listen port (overridden to 8090 in Docker compose)     |
| `LOG_LEVEL`       | `INFO`                                        | Python log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `RECOMMENDER_URL` | `http://recommender:8095`                     | Base URL of the recommender service                    |
| `OPENAI_API_KEY`  | —                                             | OpenAI key (passed through; not used server-side)      |
| `MONGODB_URL`     | —                                             | MongoDB connection string                              |
| `QDRANT_URL`      | —                                             | Qdrant connection string                               |
| `REDIS_URL`       | —                                             | Redis connection string                                |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://localhost:8080` | Comma-separated list of CORS origins                   |

For local development where the recommender runs on the host, override:

```env
RECOMMENDER_URL=http://localhost:8095
```

## Example curl calls

**Health check**

```sh
curl http://localhost:8090/health
```

**List tools**

```sh
curl -X POST http://localhost:8090/tools/list
```

**Recommend repositories**

```sh
curl -X POST http://localhost:8090/tools/execute \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "recommend_repositories",
    "parameters": {
      "query": "machine learning python",
      "top_k": 10,
      "user_id": "user-123",
      "language": "Python",
      "min_stars": 100,
      "license": "MIT",
      "variant": "hybrid"
    }
  }'
```

**Log an interaction**

```sh
curl -X POST http://localhost:8090/tools/execute \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "log_repository_interaction",
    "parameters": {
      "user_id": "user-123",
      "query": "machine learning python",
      "repo_id": "scikit-learn/scikit-learn",
      "interaction_type": "click",
      "position_in_results": 0
    }
  }'
```

**Get user preferences**

```sh
curl -X POST http://localhost:8090/tools/execute \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "get_user_preferences",
    "parameters": { "user_id": "user-123" }
  }'
```

**Query repository data (MongoDB)**

```sh
curl -X POST http://localhost:8090/tools/execute \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "query_repository_data",
    "parameters": {
      "collection": "repositories",
      "filters": {"language": "Python", "stars": {"$gte": 500}},
      "projection": ["full_name", "description", "language", "stars", "url"],
      "sort_by": "stars",
      "sort_order": "desc",
      "limit": 5
    }
  }'
```

**Explain a GitHub repository**

```sh
curl -X POST http://localhost:8090/tools/execute \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "explain_repository",
    "parameters": {
      "repo_url": "https://github.com/pallets/flask",
      "include_database_context": true,
      "include_readme_excerpt": true
    }
  }'
```
