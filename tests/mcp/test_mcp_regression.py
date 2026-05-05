"""Regression tests for MCP server behavior."""

from fastapi.testclient import TestClient

import src.mcp.server as mcp_server


def test_execute_tool_preserves_none_result(monkeypatch):
    """Regression test: tools that return None must still produce a valid success response."""

    async def fake_tool(**kwargs):
        return None

    monkeypatch.setattr(mcp_server, "get_tool", lambda tool_name: fake_tool)

    with TestClient(mcp_server.app) as client:
        response = client.post(
            "/tools/execute",
            json={
                "tool_name": "nullable_tool",
                "parameters": {"query": "anything"},
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "result": None,
        "error": None,
    }