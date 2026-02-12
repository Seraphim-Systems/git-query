# GitHub Actions Workflows

Quick reference for deployment workflows in `.github/workflows/`.

## Available Workflows

### 1. **deploy-infrastructure.yml**
**What:** Core infrastructure (databases + API gateway)  
**Deploys:** MongoDB, Redis, Qdrant, nginx, db-query-api  
**When to use:** First deployment or infrastructure updates  
**Secrets needed:** `INFRA_*`, `DB_*`, `APIKEY_*`, `SVC_NGINX_SERVER_NAME`  
**Duration:** ~5-10 minutes

### 2. **deploy-pipelines.yml**
**What:** Data collection services  
**Deploys:** scraper, processing services  
**When to use:** After infrastructure is up, for data ingestion  
**Secrets needed:** `INFRA_*`, `DB_*`, `APIKEY_*`  
**Duration:** ~3-5 minutes

### 3. **deploy-mcp.yml**
**What:** Model Context Protocol servers  
**Deploys:** MCP server with tools and integrations  
**When to use:** For chatbot/AI features  
**Secrets needed:** `INFRA_*`, `DB_*`, `APIKEY_MCP`, `APP_OPENAI_API_KEY`  
**Duration:** ~3-5 minutes

### 4. **deploy-reco.yml**
**What:** Recommendation engine  
**Deploys:** Recommender API service  
**When to use:** For recommendation features  
**Secrets needed:** `INFRA_*`, `DB_*`, `APIKEY_*`  
**Duration:** ~3-5 minutes

### 5. **deploy-training.yml**
**What:** ML model training jobs  
**Deploys:** Training containers (one-time or scheduled)  
**When to use:** To train/retrain ML models  
**Secrets needed:** `INFRA_*`, `DB_*`  
**Duration:** Varies (can be hours)

---

## Deployment Order

### First Time Setup
1. **deploy-infrastructure.yml** - Get databases and API running
2. **deploy-pipelines.yml** - Start collecting data
3. **deploy-mcp.yml** - Enable AI features
4. **deploy-reco.yml** - Enable recommendations
5. **deploy-training.yml** - Train models (optional, can be scheduled)

### Updates
- Infrastructure changes → **deploy-infrastructure.yml**
- Scraper code changes → **deploy-pipelines.yml**
- Chatbot changes → **deploy-mcp.yml**
- Recommendation logic → **deploy-reco.yml**
- Model updates → **deploy-training.yml**

---

## Quick Commands

### Trigger Manually (GitHub UI)
1. Go to **Actions** tab
2. Select workflow
3. Click **Run workflow**
4. Choose branch → **Run**

### Check Status
- Green checkmark ✅ = Success
- Red X ❌ = Failed (check logs)
- Yellow circle 🟡 = Running

### View Logs
1. Click on workflow run
2. Click on job name
3. Expand steps to see output

---

## Common Issues

**Problem:** Secrets not found  
**Fix:** Check [GITHUB_SECRETS.md](./GITHUB_SECRETS.md) and add missing secrets

**Problem:** SSH connection failed  
**Fix:** Verify `INFRA_HETZNER_*` secrets are correct

**Problem:** Container fails to start  
**Fix:** Check workflow logs, verify database credentials

**Problem:** API returns 503  
**Fix:** Wait for health checks, verify services are running: `docker ps`

---

## Service Ports

| Service | Port | Health Check |
|---------|------|-------------|
| nginx | 80 | `http://server/` |
| db-query-api | 8000 | `http://server:8000/health` |
| gateway | 8000 | `http://server:8000/api/health` |
| mcp-server | 8001 | Check logs |
| recommender | 8002 | Check logs |

---

## Need Help?

1. Check workflow logs in GitHub Actions
2. SSH to server: `ssh -i <key> <INFRA_HETZNER_USER>@<INFRA_HETZNER_HOST>`
3. Check running containers: `docker ps`
4. Check container logs: `docker logs <container-name>`
5. Restart services: Re-run the appropriate workflow
