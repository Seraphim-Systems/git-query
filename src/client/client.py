"""Interactive CLI client for the chatbot."""

import asyncio
import logging
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

from src.client.config import settings
from src.client.mcp_client import mcp_client

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()


async def chat(message: str):
    """Import chatbot lazily to avoid module import side effects during test collection."""
    from src.client.bot import chat as bot_chat

    return await bot_chat(message)


async def check_mcp_connection():
    """Check connection to MCP server."""
    console.print("\n[yellow]Checking connection to MCP server...[/yellow]")
    is_connected = await mcp_client.health_check()

    if is_connected:
        console.print("[green]✓ Connected to MCP server[/green]")

        # List available tools
        tools = await mcp_client.list_tools()
        if tools:
            console.print(f"[green]✓ Found {len(tools)} available tools:[/green]")
            for tool in tools:
                console.print(f"  • {tool['name']}: {tool['description']}")
    else:
        console.print("[red]✗ Failed to connect to MCP server[/red]")
        console.print(f"[yellow]Make sure the MCP server is running at {settings.mcp_server_url}[/yellow]")
        return False

    return True


async def interactive_chat():
    """Run interactive chat session."""
    console.print(
        Panel.fit(
            "[bold blue]Welcome to the ChatbotClient![/bold blue]\n"
            "Powered by Pydantic AI\n\n"
            "Type 'exit' or 'quit' to end the session.",
            title="Chatbot Client",
        )
    )

    # Check MCP connection
    if not await check_mcp_connection():
        console.print("\n[yellow]Continuing anyway... (some features may not work)[/yellow]")

    console.print("\n[green]Ready! Start chatting...[/green]\n")

    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")

            if user_input.lower() in ["exit", "quit", "bye"]:
                console.print("\n[yellow]Goodbye! 👋[/yellow]\n")
                break

            if not user_input.strip():
                continue

            # Show thinking indicator
            console.print("[dim]Thinking...[/dim]")

            # Get response from chatbot
            response, tool_calls = await chat(user_input)

            # Display response
            console.print("\n[bold green]Bot:[/bold green]")
            console.print(Panel(Markdown(response)))

            # Show tool calls if any
            if tool_calls:
                console.print("\n[dim]Tools used:[/dim]")
                for call in tool_calls:
                    console.print(f"[dim]  • {call['tool']}({call['parameters']})[/dim]")

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted. Goodbye! 👋[/yellow]\n")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            logger.error("Error in interactive chat: %s", e, exc_info=True)


async def main():
    """Main entry point."""
    try:
        await interactive_chat()
    finally:
        # Cleanup
        await mcp_client.close()


if __name__ == "__main__":
    asyncio.run(main())
