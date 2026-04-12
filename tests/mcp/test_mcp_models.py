import pytest
from pydantic import ValidationError

from src.mcp.models import (
    HealthResponse,
    Tool,
    ToolExecuteRequest,
    ToolExecuteResponse,
    ToolParameter,
)


def test_tool_parameter_defaults():
    param = ToolParameter(
        name="query",
        type="string",
        description="Search query",
    )

    assert param.name == "query"
    assert param.type == "string"
    assert param.description == "Search query"
    assert param.required is True
    assert param.default is None


def test_tool_parameter_with_optional_default():
    param = ToolParameter(
        name="top_k",
        type="integer",
        description="Number of results",
        required=False,
        default=10,
    )

    assert param.required is False
    assert param.default == 10


def test_tool_parameter_accepts_any_default_type():
    param = ToolParameter(
        name="filters",
        type="object",
        description="Optional filters",
        required=False,
        default={"language": "python"},
    )

    assert param.default == {"language": "python"}


def test_tool_parameter_requires_name():
    with pytest.raises(ValidationError):
        ToolParameter(
            type="string",
            description="Missing name",
        )


def test_tool_parameter_requires_type():
    with pytest.raises(ValidationError):
        ToolParameter(
            name="query",
            description="Missing type",
        )


def test_tool_parameter_requires_description():
    with pytest.raises(ValidationError):
        ToolParameter(
            name="query",
            type="string",
        )


def test_tool_defaults_parameters_to_empty_list():
    tool = Tool(
        name="recommend_repositories",
        description="Recommend repositories",
    )

    assert tool.name == "recommend_repositories"
    assert tool.description == "Recommend repositories"
    assert tool.parameters == []


def test_tool_accepts_parameter_list():
    tool = Tool(
        name="recommend_repositories",
        description="Recommend repositories",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="Search query",
            ),
            ToolParameter(
                name="top_k",
                type="integer",
                description="Maximum results",
                required=False,
                default=10,
            ),
        ],
    )

    assert len(tool.parameters) == 2
    assert tool.parameters[0].name == "query"
    assert tool.parameters[1].default == 10


def test_tool_execute_request_defaults_parameters_to_empty_dict():
    request = ToolExecuteRequest(tool_name="search_items")

    assert request.tool_name == "search_items"
    assert request.parameters == {}


def test_tool_execute_request_accepts_parameters_dict():
    request = ToolExecuteRequest(
        tool_name="search_items",
        parameters={"query": "fastapi", "top_k": 5},
    )

    assert request.tool_name == "search_items"
    assert request.parameters["query"] == "fastapi"
    assert request.parameters["top_k"] == 5


def test_tool_execute_request_requires_tool_name():
    with pytest.raises(ValidationError):
        ToolExecuteRequest(parameters={"query": "abc"})


def test_tool_execute_response_success_with_result():
    response = ToolExecuteResponse(
        success=True,
        result={"items": ["a", "b"]},
    )

    assert response.success is True
    assert response.result == {"items": ["a", "b"]}
    assert response.error is None


def test_tool_execute_response_failure_with_error():
    response = ToolExecuteResponse(
        success=False,
        error="Tool crashed",
    )

    assert response.success is False
    assert response.result is None
    assert response.error == "Tool crashed"


def test_tool_execute_response_allows_empty_payload():
    response = ToolExecuteResponse(success=True)

    assert response.success is True
    assert response.result is None
    assert response.error is None


def test_health_response_defaults_version():
    response = HealthResponse(status="healthy")

    assert response.status == "healthy"
    assert response.version == "1.0.0"


def test_health_response_accepts_custom_version():
    response = HealthResponse(status="healthy", version="2.1.0")

    assert response.status == "healthy"
    assert response.version == "2.1.0"


def test_models_dump_cleanly():
    tool = Tool(
        name="recommend_repositories",
        description="Recommend repositories",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="Search query",
            )
        ],
    )

    payload = tool.model_dump()

    assert payload["name"] == "recommend_repositories"
    assert payload["description"] == "Recommend repositories"
    assert payload["parameters"][0]["name"] == "query"


def test_tool_execute_request_model_dump_contains_parameters():
    request = ToolExecuteRequest(
        tool_name="recommend_repositories",
        parameters={"query": "machine learning", "top_k": 3},
    )

    payload = request.model_dump()

    assert payload == {
        "tool_name": "recommend_repositories",
        "parameters": {"query": "machine learning", "top_k": 3},
    }