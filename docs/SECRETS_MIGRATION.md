# GitHub Secrets Migration Guide

This document explains the changes to GitHub Secrets naming convention and how to migrate.

## What Changed

**Old Naming Pattern:** Inconsistent, no clear hierarchy

**New Naming Pattern:** `<CATEGORY>_<SERVICE>_<PURPOSE>`

## Migration Map

### Infrastructure Secrets
| Old Name | New Name | Notes |
|----------|----------|-------|
| `HETZNER_HOST` | `INFRA_HETZNER_HOST` | Added INFRA prefix |
| `HETZNER_USER` | `INFRA_HETZNER_USER` | Added INFRA prefix |
| `HETZNER_SSH_KEY` | `INFRA_HETZNER_SSH_KEY` | Added INFRA prefix |

### Database Secrets
| Old Name | New Name | Notes |
|----------|----------|-------|
| `MONGO_USER` | `DB_MONGODB_USER` | Added DB prefix, clarified MONGO→MONGODB |
| `MONGO_PASSWORD` | `DB_MONGODB_PASSWORD` | Added DB prefix, clarified MONGO→MONGODB |
| `MONGO_DB` | `DB_MONGODB_DATABASE` | Added DB prefix, clarified purpose |
| `REDIS_PASSWORD` | `DB_REDIS_PASSWORD` | Added DB prefix |
| `COSMOS_PARTITION_COUNT` | `DB_COSMOS_PARTITION_COUNT` | Added DB prefix (if still needed) |
| `QDRANT_API_KEY` | `DB_QDRANT_API_KEY` | Added DB prefix |
| `QDRANT_LOG_LEVEL` | `SVC_QDRANT_LOG_LEVEL` | Moved to SVC (service config) |

### New API Key Secrets
| Old Name | New Name | Notes |
|----------|----------|-------|
| `DATA_INGESTION_API_KEY` | `APIKEY_MONGODB` | Replaced with per-service keys |
| *N/A* | `APIKEY_REDIS` | New - for Redis API access |
| *N/A* | `APIKEY_QDRANT` | New - for Qdrant API access |
| *N/A* | `APIKEY_MCP` | New - for MCP server access |

### Application Secrets
| Old Name | New Name | Notes |
|----------|----------|-------|
| `SECRET_KEY` | `APP_SESSION_SECRET` | Clarified purpose |
| `JWT_SECRET` | `APP_JWT_SECRET` | Added APP prefix |
| `OPENAI_API_KEY` | `APP_OPENAI_API_KEY` | Added APP prefix |
| `API_SECRET` | *Removed* | Replaced by per-service APIKEY_* |

### Service Configuration
| Old Name | New Name | Notes |
|----------|----------|-------|
| `SERVER_NAME` | `SVC_NGINX_SERVER_NAME` | Added SVC prefix, clarified service |
| *N/A* | `SVC_CORS_ORIGINS` | New - for CORS configuration |
| *N/A* | `SVC_LOG_LEVEL` | New - for global log level |

## Why This Change?

### 1. **Prefix Grouping**
Easy to find all secrets of a type:
- All infrastructure: `INFRA_*`
- All database credentials: `DB_*`
- All API keys: `APIKEY_*`
- All app secrets: `APP_*`

### 2. **Per-Service API Keys**
Old system had one generic `DATA_INGESTION_API_KEY`. New system has:
- `APIKEY_MONGODB` - Only access MongoDB
- `APIKEY_REDIS` - Only access Redis
- `APIKEY_QDRANT` - Only access Qdrant
- `APIKEY_MCP` - Only access MCP server

**Benefits:**
- Better security (least privilege)
- Easier to rotate individual keys
- Clear audit trail per service

### 3. **Future-Proof**
Easy to add new services without naming conflicts:
- `DB_POSTGRES_*` for PostgreSQL
- `APIKEY_NEWSERVICE` for new services

## Migration Steps

### Step 1: Add New Secrets (Keep Old Ones)
1. Go to GitHub repository → Settings → Secrets and variables → Actions
2. Add all new secrets from [GITHUB_SECRETS.md](./GITHUB_SECRETS.md)
3. Keep old secrets temporarily for rollback

### Step 2: Update Workflow Files
Update all `.github/workflows/*.yml` files to use new secret names:

```yaml
# Old
env:
  MONGO_USER: ${{ secrets.MONGO_USER }}
  MONGO_PASSWORD: ${{ secrets.MONGO_PASSWORD }}

# New
env:
  DB_MONGODB_USER: ${{ secrets.DB_MONGODB_USER }}
  DB_MONGODB_PASSWORD: ${{ secrets.DB_MONGODB_PASSWORD }}
```

### Step 3: Update Docker Compose Files
Update environment variable mappings:

```yaml
# Old
environment:
  - MONGO_USER=${MONGO_USER}

# New
environment:
  - MONGO_USER=${DB_MONGODB_USER}
```

### Step 4: Test Deployment
1. Test each workflow with new secrets
2. Verify all services start correctly
3. Check API access with new API keys

### Step 5: Remove Old Secrets
Once confirmed working:
1. Delete old secrets from GitHub
2. Remove old references from documentation

## Checklist

- [ ] Add all new `INFRA_*` secrets
- [ ] Add all new `DB_*` secrets
- [ ] Generate and add 4 new `APIKEY_*` secrets
- [ ] Add new `APP_*` secrets
- [ ] Add new `SVC_*` secrets
- [ ] Update all workflow YAML files
- [ ] Update Docker compose environment mappings
- [ ] Test infrastructure deployment
- [ ] Test pipeline deployment
- [ ] Test MCP deployment
- [ ] Delete old secrets

## Rollback Plan

If issues occur:
1. Old secrets are still in GitHub (don't delete until confirmed)
2. Revert workflow file changes
3. Redeploy with old configuration

## Questions?

Check [GITHUB_SECRETS.md](./GITHUB_SECRETS.md) for full secret list and descriptions.
