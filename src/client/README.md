# Client

The client is an interactive CLI chatbot powered by [Pydantic AI](https://github.com/pydantic/pydantic-ai). It runs an LLM agent (OpenAI by default) that can call tools on the MCP server to search and recommend GitHub repositories in natural language.

```
User  →  CLI  →  Pydantic AI Agent  →  MCPClient  →  MCP Server (:8090)
                                                              ↓
                                                   Recommender (:8095)
```

## Running

**Docker Compose (recommended — interactive session)**

```sh
docker-compose \
  -f infrastructure/docker/docker-compose.base.yml \
  -f infrastructure/docker/docker-compose.mcp.yml \
  run --rm client
```

The client container is built but not started in daemon mode — always use `run --rm` for an interactive terminal session.

**Locally**

```sh
pip install -r requirements.txt
python -m src.client.client
```

On startup the client:

1. Checks connectivity to the MCP server and prints available tools.
2. Drops into an interactive `You:` prompt.
3. Type `exit`, `quit`, or `bye` to end the session.

## Usage

```
You: find me popular Python machine learning libraries with over 1000 stars
Bot: Here are some highly-starred Python ML repositories ...

You: log that I clicked on scikit-learn/scikit-learn
Bot: Interaction logged.

You: what are my preferences?
Bot: Based on your history you seem interested in Python and data science topics.
```

Tool calls made during each turn are printed in dimmed text below the response.

## Agent tools

The Pydantic AI agent exposes the following tools. They internally call the MCP server via `MCPClient.execute_tool`.

| Tool                         | Description                                               |
| ---------------------------- | --------------------------------------------------------- |
| `recommend_repositories`     | Find GitHub repos matching a natural-language query       |
| `log_repository_interaction` | Record a click, save, thumbs-up, etc. for a result        |
| `get_user_preferences`       | View the user's learned language/topic preference profile |
| `get_recommendation`         | Legacy recommendation tool (passes through to MCP)        |
| `search_items`               | Generic item search (legacy)                              |

The `user_id` for personalisation is injected automatically from the session — individual tool calls do not require it.

## MCPClient API

`src/client/mcp_client.py` exposes a thin async HTTP client you can import directly:

```python
from src.client.mcp_client import mcp_client

# Check server health
ok = await mcp_client.health_check()

# List available tools
tools = await mcp_client.list_tools()

# Execute a tool
result = await mcp_client.execute_tool(
    "recommend_repositories",
    {"query": "graph neural networks", "top_k": 5},
)

await mcp_client.close()
```

A module-level singleton `mcp_client` is pre-configured from the environment. To point at a different MCP server instantiate a new client:

```python
from src.client.mcp_client import MCPClient

client = MCPClient(base_url="http://my-mcp-server:8090")
```

## Environment variables (hooks)

All settings are read from the environment or a `.env` file at repository root.

| Variable             | Default                 | Description                                                       |
| -------------------- | ----------------------- | ----------------------------------------------------------------- |
| `MCP_SERVER_URL`     | `http://localhost:8090` | URL of the MCP server the client connects to                      |
| `OPENAI_API_KEY`     | —                       | OpenAI API key                                                    |
| `APP_OPENAI_API_KEY` | —                       | Alternative env name for the OpenAI key                           |
| `MODEL_NAME`         | `gpt-4o`                | OpenAI model used by the agent                                    |
| `LOG_LEVEL`          | `INFO`                  | Python log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)            |
| `MONGODB_URL`        | —                       | MongoDB connection string (optional, not used directly by client) |
| `REDIS_URL`          | —                       | Redis connection string (optional, not used directly by client)   |

Minimum required `.env` for the client:

```env
APP_OPENAI_API_KEY=sk-...
MCP_SERVER_URL=http://localhost:8090   # or http://mcp-server:8090 inside Docker
```

To switch LLM model:

```env
MODEL_NAME=gpt-4o-mini
```

To point the client at an already-running MCP server on a different host:

```env
MCP_SERVER_URL=http://192.168.1.50:8090
```
