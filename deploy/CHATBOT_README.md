# Chatbot & MCP Server Setup

This directory contains Docker Compose configurations for the ChatbotClient and MCP Server components.

## Quick Start

1. **Set up environment files:**
   ```bash
   # MCP Server
   cd ../MCP
   copy .env.example .env
   # Edit .env with your settings (optional - defaults work fine)
   
   # ChatbotClient
   cd ../ChatbotClient
   copy .env.example .env
   # Edit .env with your OpenAI API key (REQUIRED)
   ```

2. **Start all services:**
   ```bash
   docker-compose -f docker-compose.chatbot.yml up -d mcp-server
   docker-compose -f docker-compose.chatbot.yml run --rm chatbot-client
   ```

3. **Or run MCP server only and use ChatbotClient locally:**
   ```bash
   # Start MCP server
   docker-compose -f docker-compose.chatbot.yml up -d mcp-server
   
   # Run client locally
   cd ../ChatbotClient
   python client.py
   ```

4. **View MCP server logs:**
   ```bash
   docker-compose -f docker-compose.chatbot.yml logs -f mcp-server
   ```

5. **Stop services:**
   ```bash
   docker-compose -f docker-compose.chatbot.yml down
   ```

## Architecture

```
┌──────────────────┐
│  ChatbotClient   │
│  (Pydantic AI)   │
│   CLI Interface  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   MCP Server     │ (Port 8000)
│     (Tools)      │
└──────────────────┘
```

## Services

### MCP Server
- **Port:** 8000
- **Purpose:** Hosts tools that the chatbot can use
- **Environment:** See `MCP/.env.example`
- **Tools Available:**
  - `get_recommendation` - Get personalized recommendations for users
  - `search_items` - Search for items in the system

### ChatbotClient
- **Purpose:** Interactive CLI chatbot using Pydantic AI
- **Environment:** See `ChatbotClient/.env.example`
- **Requires:** OpenAI API key
- **Features:**
  - Interactive command-line chat
  - Automatic tool discovery from MCP server
  - Rich formatted output
  - Session management

## Development

For local development without Docker:

1. **Start MCP Server:**
   ```bash
   cd MCP
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   python server.py
   ```

2. **Start ChatbotClient:**
   ```bash
   cd ChatbotClient
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   python client.py
   ```

## Adding New Tools

To add new tools to the MCP server:

1. Create a new tool function in `MCP/tools/example_tool.py`
2. Add the tool definition to the `TOOL_DEFINITIONS` list
3. Register it in `MCP/tools/__init__.py`
4. Restart the MCP server
5. The ChatbotClient will automatically discover the new tool!

## Example Usage

```
You: Can you get recommendations for user123?
Bot: [Calls get_recommendation tool] Here are personalized recommendations...

You: Search for "laptop"
Bot: [Calls search_items tool] I found these items matching "laptop"...
```
