import pytest
import src.client.client as cli


class FakeConsole:
    def __init__(self):
        self.messages = []

    def print(self, *args, **kwargs):
        self.messages.append(args)

@pytest.mark.asyncio
async def test_interactive_chat_happy_path_then_exit(monkeypatch):
    fake_console = FakeConsole()
    monkeypatch.setattr(cli, "console", fake_console)

    inputs = iter(["hello bot", "exit"])
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(inputs))

    async def fake_check():
        return True

    async def fake_chat(user_input):
        assert user_input == "hello bot"
        return (
            "Here are some recommendations",
            [{"tool": "recommend_repositories", "parameters": {"query": "python"}}],
        )

    monkeypatch.setattr(cli, "check_mcp_connection", fake_check)
    monkeypatch.setattr(cli, "chat", fake_chat)

    await cli.interactive_chat()

    flattened = " ".join(str(item) for call in fake_console.messages for item in call)
    assert "Ready! Start chatting" in flattened
    assert "Thinking..." in flattened
    assert "Tools used:" in flattened
    assert "recommend_repositories" in flattened
    assert "Goodbye!" in flattened

@pytest.mark.asyncio
async def test_interactive_chat_only_exit(monkeypatch):
    fake_console = FakeConsole()
    monkeypatch.setattr(cli, "console", fake_console)

    inputs = iter(["exit"])
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: next(inputs))

    async def fake_check():
        return True

    monkeypatch.setattr(cli, "check_mcp_connection", fake_check)

    await cli.interactive_chat()

    flattened = " ".join(str(x) for call in fake_console.messages for x in call)
    assert "Goodbye!" in flattened