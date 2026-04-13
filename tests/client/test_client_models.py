import pytest
from pydantic import ValidationError

from src.client.models import ChatMessage, ChatSession


def test_chat_message_defaults_tool_calls_to_empty_list():
    message = ChatMessage(role="user", content="Hello")

    assert message.role == "user"
    assert message.content == "Hello"
    assert message.tool_calls == []


def test_chat_message_accepts_tool_calls():
    message = ChatMessage(
        role="assistant",
        content="Here are some repos",
        tool_calls=[
            {
                "tool": "recommend_repositories",
                "parameters": {"query": "fastapi"},
            }
        ],
    )

    assert len(message.tool_calls) == 1
    assert message.tool_calls[0]["tool"] == "recommend_repositories"
    assert message.tool_calls[0]["parameters"]["query"] == "fastapi"


def test_chat_message_requires_role():
    with pytest.raises(ValidationError):
        ChatMessage(content="Missing role")


def test_chat_message_requires_content():
    with pytest.raises(ValidationError):
        ChatMessage(role="user")


def test_chat_message_allows_empty_content_string():
    message = ChatMessage(role="assistant", content="")
    assert message.content == ""


def test_chat_session_defaults_messages_to_empty_list():
    session = ChatSession(session_id="session-1")

    assert session.session_id == "session-1"
    assert session.messages == []
    assert session.user_id is None


def test_chat_session_accepts_messages():
    session = ChatSession(
        session_id="session-2",
        user_id="user-123",
        messages=[
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello!"),
        ],
    )

    assert session.user_id == "user-123"
    assert len(session.messages) == 2
    assert session.messages[0].role == "user"
    assert session.messages[1].content == "Hello!"


def test_chat_session_requires_session_id():
    with pytest.raises(ValidationError):
        ChatSession()


def test_chat_session_model_dump_is_serializable():
    session = ChatSession(
        session_id="session-3",
        user_id="user-999",
        messages=[
            ChatMessage(
                role="assistant",
                content="Used tool",
                tool_calls=[{"tool": "search_items", "parameters": {"query": "llm"}}],
            )
        ],
    )

    payload = session.model_dump()

    assert payload["session_id"] == "session-3"
    assert payload["user_id"] == "user-999"
    assert payload["messages"][0]["role"] == "assistant"
    assert payload["messages"][0]["tool_calls"][0]["tool"] == "search_items"


def test_chat_session_accepts_empty_message_history():
    session = ChatSession(session_id="session-4", messages=[])

    assert session.messages == []


@pytest.mark.parametrize(
    ("role", "content"),
    [
        ("user", "Find me repos"),
        ("assistant", "Here you go"),
        ("system", "You are helpful"),
    ],
)
def test_chat_message_supports_multiple_roles(role, content):
    message = ChatMessage(role=role, content=content)

    assert message.role == role
    assert message.content == content


def test_chat_message_tool_calls_can_store_nested_payloads():
    message = ChatMessage(
        role="assistant",
        content="Detailed call",
        tool_calls=[
            {
                "tool": "recommend_repositories",
                "parameters": {
                    "query": "python",
                    "filters": {"language": "Python", "top_k": 5},
                },
            }
        ],
    )

    assert message.tool_calls[0]["parameters"]["filters"]["language"] == "Python"
    assert message.tool_calls[0]["parameters"]["filters"]["top_k"] == 5