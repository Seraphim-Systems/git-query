# GitHub Secrets Configuration

This document lists all required GitHub Secrets for the git-query deployment workflows.

## Required Secrets

### 🔐 Server Access (Hetzner)

| Secret Name | Description | Example | Required For |
|------------|-------------|---------|--------------|
| `HETZNER_HOST` | Hetzner server IP address or hostname | `123.456.789.012` or `example.com` | All deployments |
| `HETZNER_USER` | SSH username for Hetzner server | `root` or `ubuntu` | All deployments |
| `HETZNER_SSH_KEY` | Private SSH key for server access | `-----BEGIN RSA PRIVATE KEY-----...` | All deployments |

### 🗄️ Database Credentials

| Secret Name | Description | Example | Required For |
|------------|-------------|---------|--------------|
| `MONGO_USER` | MongoDB admin username | `admin` | Infrastructure, Pipelines, MCP, Reco, Training |
| `MONGO_PASSWORD` | MongoDB admin password | `secureMongoPass123` | Infrastructure, Pipelines, MCP, Reco, Training |
| `MONGO_DB` | MongoDB database name | `gitquery` | Infrastructure |
| `REDIS_PASSWORD` | Redis authentication password | `secureRedisPass123` | Infrastructure, Pipelines, MCP, Reco |
| `COSMOS_PARTITION_COUNT` | Azure Cosmos DB partition count | `10` (dev) or `20` (prod) | Infrastructure |
| `QDRANT_LOG_LEVEL` | Qdrant vector DB log level | `INFO` or `WARN` | Infrastructure |
| `QDRANT_API_KEY` | Qdrant API key (optional for self-hosted) | Leave empty for local | Pipelines, MCP, Reco, Training |

### 🔑 Application Secrets

| Secret Name | Description | Example | Required For |
|------------|-------------|---------|--------------|
| `SECRET_KEY` | Application secret key | Random 32+ char string | Production |
| `JWT_SECRET` | JWT token signing secret | Random 32+ char string | Production |
| `OPENAI_API_KEY` | OpenAI API key for LLM features | `sk-...` | Production |
| `API_SECRET` | General API authentication secret | Random string | Production |
| `DATA_INGESTION_API_KEY` | Data ingestion endpoint auth key | Random string | Infrastructure (db-api) |

### 🌐 Server Configuration

| Secret Name | Description | Example | Required For |
|------------|-------------|---------|--------------|
| `SERVER_NAME` | Nginx server name / domain | `api.gitquery.com` or `_` | Infrastructure, Production |

## Workflow Coverage

### deploy-infrastructure.yml
**Purpose:** Deploy database infrastructure (MongoDB, Cosmos DB, Qdrant, Redis, db-api, nginx)

**Required Secrets:**
- `HETZNER_HOST`
- `HETZNER_USER`
- `HETZNER_SSH_KEY`
- `MONGO_USER`
- `MONGO_PASSWORD`
- `MONGO_DB`
- `REDIS_PASSWORD`
- `COSMOS_PARTITION_COUNT`
- `QDRANT_LOG_LEVEL`

### deploy-pipelines.yml
**Purpose:** Deploy data scraper and processing services

**Required Secrets:**
- `HETZNER_HOST`
- `HETZNER_USER`
- `HETZNER_SSH_KEY`
- `MONGO_USER`
- `MONGO_PASSWORD`
- `REDIS_PASSWORD`
- `QDRANT_API_KEY` (optional)

### deploy-mcp.yml
**Purpose:** Deploy Model Context Protocol servers

**Required Secrets:**
- `HETZNER_HOST`
- `HETZNER_USER`
- `HETZNER_SSH_KEY`
- `MONGO_USER`
- `MONGO_PASSWORD`
- `REDIS_PASSWORD`
- `QDRANT_API_KEY` (optional)

### deploy-reco.yml
**Purpose:** Deploy recommendation engine

**Required Secrets:**
- `HETZNER_HOST`
- `HETZNER_USER`
- `HETZNER_SSH_KEY`
- `MONGO_USER`
- `MONGO_PASSWORD`
- `REDIS_PASSWORD`
- `QDRANT_API_KEY` (optional)

### deploy-training.yml
**Purpose:** Run ML model training jobs

**Required Secrets:**
- `HETZNER_HOST`
- `HETZNER_USER`
- `HETZNER_SSH_KEY`
- `MONGO_USER`
- `MONGO_PASSWORD`
- `QDRANT_API_KEY` (optional)

### deploy-prod.yml
**Purpose:** Full production deployment (legacy)

**Required Secrets:**
- `HETZNER_HOST`
- `HETZNER_USER`
- `HETZNER_SSH_KEY`
- `MONGO_USER`
- `MONGO_PASSWORD`
- `REDIS_PASSWORD`
- `SECRET_KEY`
- `JWT_SECRET`
- `OPENAI_API_KEY`
- `API_SECRET`
- `SERVER_NAME`

## How to Add Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Enter the secret name and value
5. Click **Add secret**

## Security Best Practices

✅ **DO:**
- Use strong, randomly generated passwords (32+ characters)
- Rotate secrets periodically
- Use different credentials for dev/staging/production
- Keep SSH keys secure and never commit them
- Use minimum required permissions

❌ **DON'T:**
- Commit secrets to git repository
- Share secrets via insecure channels
- Use the same password across environments
- Hard-code secrets in workflow files

## Generating Secure Secrets

### Random Password (Linux/macOS)
```bash
openssl rand -base64 32
```

### SSH Key Pair
```bash
ssh-keygen -t rsa -b 4096 -C "github-actions@gitquery"
# Use the private key for HETZNER_SSH_KEY
# Add the public key to server's ~/.ssh/authorized_keys
```

### JWT Secret
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Optional Secrets

The following secrets have default values and are optional:

- `QDRANT_API_KEY` - Empty by default (self-hosted Qdrant doesn't require auth)
- `COSMOS_PARTITION_COUNT` - Defaults to 10
- `QDRANT_LOG_LEVEL` - Defaults to INFO

## Server Setup Requirements

Before running workflows, ensure your Hetzner server has:

1. **Docker & Docker Compose installed**
   ```bash
   curl -fsSL https://get.docker.com | sh
   ```

2. **Git repository cloned to `~/git-query`**
   ```bash
   mkdir -p ~/git-query
   git clone https://github.com/Seraphim-Systems/git-query.git ~/git-query
   ```

3. **SSH key authorized**
   ```bash
   mkdir -p ~/.ssh
   echo "YOUR_PUBLIC_KEY" >> ~/.ssh/authorized_keys
   chmod 600 ~/.ssh/authorized_keys
   ```

4. **Required ports open**
   - 80 (HTTP)
   - 22 (SSH)

## Troubleshooting

### "Secret not found" error
- Verify secret name matches exactly (case-sensitive)
- Check secret is created at repository level, not organization level
- Ensure workflow has permission to access secrets

### SSH connection fails
- Verify `HETZNER_HOST` is correct
- Check `HETZNER_SSH_KEY` is the complete private key including headers
- Ensure public key is in server's `~/.ssh/authorized_keys`
- Verify firewall allows port 22

### Database connection fails
- Check `MONGO_USER` and `MONGO_PASSWORD` match infrastructure deployment
- Verify database services are running: `docker ps | grep git-query`
- Check `.env` file exists on server in `~/git-query`
