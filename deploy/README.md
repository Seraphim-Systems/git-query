# Deployment

## Quick Start

Build and run all services:

```bash
cd deploy
docker-compose up --build
```

Access:
- Web UI: http://localhost
- Backend API: http://localhost:8000
- Chat Agent MCP: http://localhost:8001
- Recommender MCP: http://localhost:8002

## Services

- **web**: Nginx serving frontend + proxying `/api/*` to backend
- **backend**: Flask/FastAPI API gateway
- **chat-agent**: MCP server for chat agent (calls recommender tools)
- **recommender**: MCP tool server for recommendations
- **db**: PostgreSQL database

## Environment Variables

Create `.env` file in project root (copy from `.env.example`):

```env
DATABASE_URL=postgresql://user:pass@db:5432/gitquery
CHAT_AGENT_URL=http://chat-agent:8001
RECOMMENDER_URL=http://recommender:8002
```

## Individual Service Commands

```bash
# Build specific service
docker-compose build backend

# Run specific service
docker-compose up web

# View logs
docker-compose logs -f chat-agent

# Stop all
docker-compose down

# Clean volumes
docker-compose down -v
```
