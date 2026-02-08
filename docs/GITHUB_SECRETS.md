# GitHub Secrets Configuration

This document lists all required GitHub Secrets for git-query deployment workflows.

## Naming Convention

Pattern: `<CATEGORY>_<SERVICE>_<PURPOSE>` (no environment suffix for simplicity)

- **CATEGORY**: `INFRA`, `DB`, `APIKEY`, `APP`, `SVC`
- **SERVICE**: Specific service name (e.g., `MONGODB`, `REDIS`, `HETZNER`)
- **PURPOSE**: What the secret is for (e.g., `USER`, `PASSWORD`, `KEY`)

---

## Required Secrets by Category

### Infrastructure Access
| Secret Name | Description | Example |
|------------|-------------|---------|
| `INFRA_HETZNER_HOST` | Server IP address or hostname | `123.456.789.012` |
| `INFRA_HETZNER_USER` | SSH username | `root` |
| `INFRA_HETZNER_SSH_KEY` | Private SSH key for server access | `-----BEGIN RSA PRIVATE KEY-----...` |

### Database Credentials
| Secret Name | Description | Example |
|------------|-------------|---------|
| `DB_MONGODB_USER` | MongoDB admin username | `admin` |
| `DB_MONGODB_PASSWORD` | MongoDB admin password | `secure-password-123` |
| `DB_MONGODB_DATABASE` | Default database name | `gitquery` |
| `DB_REDIS_PASSWORD` | Redis authentication password | `secure-redis-pass-123` |
| `DB_QDRANT_API_KEY` | Qdrant API key (optional for self-hosted) | Leave empty for local |

### API Keys (per-service authentication)
| Secret Name | Description | Example |
|------------|-------------|---------|
| `APIKEY_MONGODB` | API key for MongoDB endpoints | `mongodb-api-key-change-in-prod` |
| `APIKEY_REDIS` | API key for Redis endpoints | `redis-api-key-change-in-prod` |
| `APIKEY_QDRANT` | API key for Qdrant endpoints | `qdrant-api-key-change-in-prod` |
| `APIKEY_MCP` | API key for MCP server endpoints | `mcp-api-key-change-in-prod` |

### Application Secrets
| Secret Name | Description | Example |
|------------|-------------|---------|
| `APP_JWT_SECRET` | JWT signing secret | Random 32+ char string |
| `APP_SESSION_SECRET` | Session encryption key | Random 32+ char string |
| `APP_OPENAI_API_KEY` | OpenAI API key for LLM features | `sk-...` |

### Service Configuration
| Secret Name | Description | Example |
|------------|-------------|---------|
| `SVC_NGINX_SERVER_NAME` | Domain name or server name | `api.gitquery.com` or `_` |
| `SVC_CORS_ORIGINS` | Comma-separated allowed origins | `https://example.com,https://app.example.com` |
| `SVC_LOG_LEVEL` | Global log level | `INFO` |

---

## Workflow Usage Matrix

| Workflow | Required Secrets |
|----------|-----------------|
| **deploy-infrastructure.yml** | `INFRA_*`, `DB_*`, `SVC_NGINX_SERVER_NAME` |
| **deploy-pipelines.yml** | `INFRA_*`, `DB_MONGODB_*`, `DB_REDIS_PASSWORD`, `APIKEY_*` |
| **deploy-mcp.yml** | `INFRA_*`, `DB_*`, `APIKEY_MCP`, `APP_OPENAI_API_KEY` |
| **deploy-reco.yml** | `INFRA_*`, `DB_*`, `APIKEY_*` |
| **deploy-training.yml** | `INFRA_*`, `DB_*` |

---

## Environment Variables vs Secrets

**Use GitHub Secrets for:**
- Passwords, API keys, tokens
- SSH keys
- Any sensitive authentication data

**Use environment variables (hardcoded) for:**
- Non-sensitive configuration (ports, hostnames)
- Feature flags
- Public URLs

---

## Security Best Practices

1. **Rotate keys regularly** - Especially API keys and database passwords
2. **Use strong passwords** - Minimum 32 characters for production
3. **Separate dev/prod** - Use different keys for development and production
4. **Limit scope** - Each API key should only access what it needs
5. **Monitor usage** - Log all API key usage and alert on suspicious activity

---

## Quick Setup Checklist

- [ ] Add all `INFRA_*` secrets for server access
- [ ] Add all `DB_*` secrets for database credentials
- [ ] Generate and add all `APIKEY_*` secrets (one per service)
- [ ] Add `APP_*` secrets for application security
- [ ] Configure `SVC_*` secrets for service configuration
- [ ] Test each workflow with the secrets configured
