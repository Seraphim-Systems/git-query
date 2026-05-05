"""Edge-case tests for the MCP protocol layer.

These tests focus on malformed inputs, missing/empty payloads, and empty tool results.
"""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import src.mcp.server as mcp_server
from src.mcp.models import ToolExecuteRequest, ToolExecuteResponse


def test_tool_execute_request_rejects_non_mapping_parameters():
    with pytest.raises(ValidationError):
        ToolExecuteRequest(
            tool_name="recommend_repositories",
            parameters=["bad", "shape"],  # type: ignore[arg-type]
        )


def test_tool_execute_request_defaults_missing_parameters_to_empty_dict():
    request = ToolExecuteRequest(tool_name="recommend_repositories")
    assert request.tool_name == "recommend_repositories"
    assert request.parameters == {}


def test_tool_execute_response_allows_empty_collection_results():
    response_list = ToolExecuteResponse(success=True, result=[])
    assert response_list.success is True
    assert response_list.result == []
    assert response_list.error is None

    response_dict = ToolExecuteResponse(success=True, result={})
    assert response_dict.success is True
    assert response_dict.result == {}
    assert response_dict.error is None


def test_execute_tool_returns_invalid_parameters_for_unexpected_keyword(monkeypatch):
    async def fake_tool(query: str):
        return {"query": query}

    monkeypatch.setattr(mcp_server, "get_tool", lambda tool_name: fake_tool)

    with TestClient(mcp_server.app) as client:
        response = client.post(
            "/tools/execute",
            json={
                "tool_name": "recommend_repositories",
                "parameters": {
                    "query": "fastapi",
                    "unexpected": 1,
                },
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["result"] is None
    assert body["error"] is not None
    assert body["error"].startswith("Invalid parameters:")


def test_execute_tool_preserves_empty_list_result(monkeypatch):
    async def fake_tool(**kwargs):
        return []

    monkeypatch.setattr(mcp_server, "get_tool", lambda tool_name: fake_tool)

    with TestClient(mcp_server.app) as client:
        response = client.post(
            "/tools/execute",
            json={
                "tool_name": "search_items",
                "parameters": {},
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "result": [],
        "error": None,
    }


def test_execute_tool_preserves_empty_dict_result(monkeypatch):
    async def fake_tool(**kwargs):
        return {}

    monkeypatch.setattr(mcp_server, "get_tool", lambda tool_name: fake_tool)

    with TestClient(mcp_server.app) as client:
        response = client.post(
            "/tools/execute",
            json={
                "tool_name": "search_items",
                "parameters": {},
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "result": {},
        "error": None,
    }


def test_execute_tool_returns_404_for_empty_tool_name(monkeypatch):
    monkeypatch.setattr(mcp_server, "get_tool", lambda tool_name: None)

    with TestClient(mcp_server.app) as client:
        response = client.post(
            "/tools/execute",
            json={
                "tool_name": "",
                "parameters": {},
            },
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "Tool '' not found"


def test_execute_tool_request_validation_error_when_parameters_is_null():
    with TestClient(mcp_server.app) as client:
        response = client.post(
            "/tools/execute",
            json={
                "tool_name": "recommend_repositories",
                "parameters": None,
            },
        )

    assert response.status_code == 422