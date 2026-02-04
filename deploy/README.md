# Deployment

## Quick Start

### Development (exposes ports to host)

```bash
cd deploy
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

Access:
- Web UI: http://localhost
- App API: http://localhost:8000
- Chat Agent MCP: http://localhost:8001
- Recommender MCP: http://localhost:8002
- PostgreSQL: localhost:5432

### Production (internal networking only)

First, set up GitHub Variables for versioning in your repository (Settings → Secrets and variables → Actions → Variables):
- `APP_VERSION` (e.g., `v1.0.0`, `latest`)
- `CHAT_AGENT_VERSION`
- `RECOMMENDER_VERSION`
- `NGINX_VERSION`
- `DB_VERSION`

Then set up GitHub Secrets:
- `DATABASE_URL`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`
- `SECRET_KEY`, `JWT_SECRET`, `API_SECRET`
- `OPENAI_API_KEY`
- `REGISTRY_URL`, `REGISTRY_USERNAME`, `REGISTRY_PASSWORD`
- `PROD_HOST`, `PROD_USER`, `PROD_SSH_KEY`

Manual deployment:

```bash
# Create .env file with production secrets (see .env.example)
cp .env.example .env
# Edit .env with real values

# Deploy
cd deploy
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d
```

Automated deployment: Push to `main` branch triggers the deploy workflow

Access only via nginx (port 80) - configure external load balancer to route to the nginx container

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
