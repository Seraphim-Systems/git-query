import os

from fastapi.testclient import TestClient

import src.mcp.server as mcp_server


def test_root_redirects_to_localhost_when_env_not_set(monkeypatch):
    monkeypatch.delenv("SVC_NGINX_SERVER_NAME", raising=False)

    with TestClient(mcp_server.app) as client:
        response = client.get("/", follow_redirects=False)

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "http://localhost:8080"


def test_root_redirects_to_nginx_host_when_env_is_set(monkeypatch):
    monkeypatch.setenv("SVC_NGINX_SERVER_NAME", "gitquery.example.com")

    with TestClient(mcp_server.app) as client:
        response = client.get("/", follow_redirects=False)

    assert response.status_code in (302, 307)
    assert response.headers["location"] == "https://gitquery.example.com"


def test_health_returns_expected_payload():
    with TestClient(mcp_server.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "version": "1.0.0"}


def test_tools_list_returns_tools_and_count(monkeypatch):
    fake_tools = [
        {"name": "tool_a", "description": "A"},
        {"name": "tool_b", "description": "B"},
    ]
    monkeypatch.setattr(mcp_server, "list_tools", lambda: fake_tools)

    with TestClient(mcp_server.app) as client:
        response = client.post("/tools/list")

    assert response.status_code == 200
    assert response.json() == {"tools": fake_tools, "count": 2}


def test_tools_list_returns_empty_list_when_no_tools(monkeypatch):
    monkeypatch.setattr(mcp_server, "list_tools", lambda: [])

    with TestClient(mcp_server.app) as client:
        response = client.post("/tools/list")

    assert response.status_code == 200
    assert response.json() == {"tools": [], "count": 0}


def test_execute_tool_success(monkeypatch):
    async def fake_tool(**kwargs):
        return {"echo": kwargs}

    monkeypatch.setattr(mcp_server, "get_tool", lambda tool_name: fake_tool)

    with TestClient(mcp_server.app) as client:
        response = client.post(
            "/tools/execute",
            json={
                "tool_name": "recommend_repositories",
                "parameters": {"query": "fastapi", "top_k": 5},
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["result"] == {"echo": {"query": "fastapi", "top_k": 5}}
    assert body["error"] is None


def test_execute_tool_returns_404_for_unknown_tool(monkeypatch):
    monkeypatch.setattr(mcp_server, "get_tool", lambda tool_name: None)

    with TestClient(mcp_server.app) as client:
        response = client.post(
            "/tools/execute",
            json={"tool_name": "missing_tool", "parameters": {}},
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Tool 'missing_tool' not found"


def test_execute_tool_returns_error_on_type_error(monkeypatch):
    async def fake_tool(**kwargs):
        raise TypeError("missing required argument: query")

    monkeypatch.setattr(mcp_server, "get_tool", lambda tool_name: fake_tool)

    with TestClient(mcp_server.app) as client:
        response = client.post(
            "/tools/execute",
            json={"tool_name": "recommend_repositories", "parameters": {}},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert "Invalid parameters" in body["error"]


def test_execute_tool_returns_error_on_generic_exception(monkeypatch):
    async def fake_tool(**kwargs):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(mcp_server, "get_tool", lambda tool_name: fake_tool)

    with TestClient(mcp_server.app) as client:
        response = client.post(
            "/tools/execute",
            json={
                "tool_name": "recommend_repositories",
                "parameters": {"query": "python"},
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["error"] == "backend exploded"


def test_execute_tool_uses_default_empty_parameters(monkeypatch):
    async def fake_tool(**kwargs):
        return {"received": kwargs}

    monkeypatch.setattr(mcp_server, "get_tool", lambda tool_name: fake_tool)

    with TestClient(mcp_server.app) as client:
        response = client.post(
            "/tools/execute",
            json={"tool_name": "simple_tool"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["result"] == {"received": {}}


def test_execute_tool_request_validation_error_when_tool_name_missing():
    with TestClient(mcp_server.app) as client:
        response = client.post(
            "/tools/execute",
            json={"parameters": {"query": "missing name"}},
        )

    assert response.status_code == 422


def test_execute_tool_request_validation_error_when_parameters_wrong_type():
    with TestClient(mcp_server.app) as client:
        response = client.post(
            "/tools/execute",
            json={"tool_name": "x", "parameters": "not-a-dict"},
        )

    assert response.status_code == 422


def test_chat_endpoint_success(monkeypatch):
    async def fake_bot_chat(message, user_id=None):
        return ("hello from bot", [{"tool": "recommend_repositories"}])

    import src.client.bot as real_bot

    monkeypatch.setattr(real_bot, "chat", fake_bot_chat)

    with TestClient(mcp_server.app) as client:
        response = client.post(
            "/chat",
            json={"message": "hi", "user_id": "u-1"},
        )

    assert response.status_code == 200
    assert response.json() == {"response": "hello from bot"}


def test_chat_endpoint_returns_friendly_error_on_runtime_failure(monkeypatch):
    async def fake_bot_chat(message, user_id=None):
        raise RuntimeError("upstream failure")

    import src.client.bot as real_bot

    monkeypatch.setattr(real_bot, "chat", fake_bot_chat)

    with TestClient(mcp_server.app) as client:
        response = client.post(
            "/chat",
            json={"message": "hi", "user_id": "u-1"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "response": "I encountered an error processing your request. Please try again."
    }


def test_chat_endpoint_validates_missing_message():
    with TestClient(mcp_server.app) as client:
        response = client.post("/chat", json={"user_id": "u-1"})

    assert response.status_code == 422


def test_chat_endpoint_accepts_optional_context_and_preferences_keys():
    async def fake_bot_chat(message, user_id=None):
        return ("ok", [])

    import src.client.bot as real_bot

    monkeypatch = None  # keeps linters quiet in editors that inspect statically
    real_bot.chat = fake_bot_chat

    with TestClient(mcp_server.app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "hello",
                "user_id": "u-2",
                "preferences": {"language": "python"},
                "context": {"source": "ui"},
            },
        )

    assert response.status_code == 200
    assert response.json()["response"] == "ok"

def test_tools_list_invalid_method():
    with TestClient(mcp_server.app) as client:
        response = client.get("/tools/list")
    assert response.status_code in (405, 404)

def test_execute_tool_empty_body():
    with TestClient(mcp_server.app) as client:
        response = client.post("/tools/execute", json={})
    assert response.status_code == 422

def test_chat_endpoint_forwards_message_history_from_context(monkeypatch):
    seen = {}

    async def fake_bot_chat(message, user_id=None, message_history=None):
        seen["message"] = message
        seen["user_id"] = user_id
        seen["message_history"] = message_history
        return ("ok", [])

    import src.client.bot as real_bot
    monkeypatch.setattr(real_bot, "chat", fake_bot_chat)

    with TestClient(mcp_server.app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "hello",
                "user_id": "u-42",
                "context": {
                    "message_history": [
                        {"role": "user", "content": "previous turn"},
                        {"role": "assistant", "content": "previous answer"},
                    ]
                },
            },
        )

    assert response.status_code == 200
    assert response.json() == {"response": "ok"}
    assert seen["message"] == "hello"
    assert seen["user_id"] == "u-42"
    assert seen["message_history"] == [
        {"role": "user", "content": "previous turn"},
        {"role": "assistant", "content": "previous answer"},
    ]