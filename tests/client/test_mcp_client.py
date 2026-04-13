import pytest

from src.client.mcp_client import MCPClient


class DummyResponse:
    def __init__(self, json_data=None, status_code=200, raise_error=None):
        self._json_data = json_data or {}
        self.status_code = status_code
        self._raise_error = raise_error

    def raise_for_status(self):
        if self._raise_error:
            raise self._raise_error

    def json(self):
        return self._json_data


class DummyAsyncClient:
    def __init__(self, *args, **kwargs):
        self.base_url = kwargs.get("base_url")
        self.timeout = kwargs.get("timeout")
        self.post_calls = []
        self.get_calls = []
        self.closed = False
        self.next_post_response = DummyResponse()
        self.next_get_response = DummyResponse()

    async def post(self, url, json=None):
        self.post_calls.append((url, json))
        return self.next_post_response

    async def get(self, url):
        self.get_calls.append(url)
        return self.next_get_response

    async def aclose(self):
        self.closed = True


@pytest.fixture
def dummy_http_client(monkeypatch):
    holder = {}

    def factory(*args, **kwargs):
        client = DummyAsyncClient(*args, **kwargs)
        holder["client"] = client
        return client

    monkeypatch.setattr("src.client.mcp_client.httpx.AsyncClient", factory)
    return holder


@pytest.mark.asyncio
async def test_init_uses_provided_base_url(dummy_http_client):
    client = MCPClient(base_url="http://test-mcp:9999")

    assert client.base_url == "http://test-mcp:9999"
    assert dummy_http_client["client"].base_url == "http://test-mcp:9999"
    assert dummy_http_client["client"].timeout == 30.0

    await client.close()


@pytest.mark.asyncio
async def test_list_tools_returns_tools(dummy_http_client):
    client = MCPClient(base_url="http://mcp")
    dummy_http_client["client"].next_post_response = DummyResponse(
        json_data={
            "tools": [
                {"name": "recommend_repositories"},
                {"name": "search_items"},
            ]
        }
    )

    tools = await client.list_tools()

    assert tools == [
        {"name": "recommend_repositories"},
        {"name": "search_items"},
    ]
    assert dummy_http_client["client"].post_calls == [("/tools/list", None)]

    await client.close()


@pytest.mark.asyncio
async def test_list_tools_returns_empty_list_when_tools_key_missing(dummy_http_client):
    client = MCPClient(base_url="http://mcp")
    dummy_http_client["client"].next_post_response = DummyResponse(json_data={})

    tools = await client.list_tools()

    assert tools == []

    await client.close()


@pytest.mark.asyncio
async def test_list_tools_returns_empty_list_on_exception(dummy_http_client, monkeypatch):
    client = MCPClient(base_url="http://mcp")

    async def broken_post(url, json=None):
        raise RuntimeError("network down")

    dummy_http_client["client"].post = broken_post

    tools = await client.list_tools()

    assert tools == []

    await client.close()


@pytest.mark.asyncio
async def test_execute_tool_returns_response_json(dummy_http_client):
    client = MCPClient(base_url="http://mcp")
    dummy_http_client["client"].next_post_response = DummyResponse(
        json_data={"success": True, "result": {"items": [1, 2, 3]}}
    )

    result = await client.execute_tool(
        "recommend_repositories",
        {"query": "fastapi", "top_k": 3},
    )

    assert result == {"success": True, "result": {"items": [1, 2, 3]}}
    assert dummy_http_client["client"].post_calls == [
        (
            "/tools/execute",
            {
                "tool_name": "recommend_repositories",
                "parameters": {"query": "fastapi", "top_k": 3},
            },
        )
    ]

    await client.close()


@pytest.mark.asyncio
async def test_execute_tool_returns_error_dict_on_exception(dummy_http_client):
    client = MCPClient(base_url="http://mcp")

    async def broken_post(url, json=None):
        raise RuntimeError("boom")

    dummy_http_client["client"].post = broken_post

    result = await client.execute_tool("recommend_repositories", {"query": "python"})

    assert result["success"] is False
    assert "boom" in result["error"]

    await client.close()


@pytest.mark.asyncio
async def test_execute_tool_returns_error_when_raise_for_status_fails(dummy_http_client):
    client = MCPClient(base_url="http://mcp")
    dummy_http_client["client"].next_post_response = DummyResponse(
        json_data={},
        raise_error=RuntimeError("503 service unavailable"),
    )

    result = await client.execute_tool("recommend_repositories", {"query": "python"})

    assert result["success"] is False
    assert "503 service unavailable" in result["error"]

    await client.close()


@pytest.mark.asyncio
async def test_health_check_returns_true_on_200(dummy_http_client):
    client = MCPClient(base_url="http://mcp")
    dummy_http_client["client"].next_get_response = DummyResponse(status_code=200)

    ok = await client.health_check()

    assert ok is True
    assert dummy_http_client["client"].get_calls == ["/health"]

    await client.close()


@pytest.mark.asyncio
async def test_health_check_returns_false_on_exception(dummy_http_client):
    client = MCPClient(base_url="http://mcp")

    async def broken_get(url):
        raise RuntimeError("timeout")

    dummy_http_client["client"].get = broken_get

    ok = await client.health_check()

    assert ok is False

    await client.close()


@pytest.mark.asyncio
async def test_health_check_returns_false_on_raise_for_status_failure(dummy_http_client):
    client = MCPClient(base_url="http://mcp")
    dummy_http_client["client"].next_get_response = DummyResponse(
        status_code=500,
        raise_error=RuntimeError("server error"),
    )

    ok = await client.health_check()

    assert ok is False

    await client.close()


@pytest.mark.asyncio
async def test_close_calls_aclose(dummy_http_client):
    client = MCPClient(base_url="http://mcp")

    assert dummy_http_client["client"].closed is False

    await client.close()

    assert dummy_http_client["client"].closed is True

@pytest.mark.asyncio
async def test_execute_tool_empty_parameters(dummy_http_client):
    client = MCPClient(base_url="http://mcp")
    dummy_http_client["client"].next_post_response = DummyResponse(json_data={"ok": True})

    result = await client.execute_tool("tool", {})

    assert result is not None
    await client.close()