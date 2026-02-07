# Service Category Deployment Guide

## Overview

This guide provides detailed deployment instructions for each service category in the git-query platform.

## Service Categories

### 1. Infrastructure Layer (Databases & Core Services)

**Services:**
- MongoDB (document database)
- Cosmos DB (Azure CosmosDB emulator)
- Qdrant (vector database)
- Redis (cache)
- DB Query API (unified database access)
- Nginx (reverse proxy/gateway)

**Manual Deployment:**
```bash
# Via GitHub Actions
# Navigate to Actions → Select workflow → Run workflow → Choose environment
# All deployments are manual - no automatic deployments on code push
```

**Dependencies:** None (these are foundation services)

**Rollback:**
```bash
cd /home/user/git-query
docker-compose -f infrastructure/docker/docker-compose.base.yml \
  -f infrastructure/docker/docker-compose.mongodb.yml \
  down
# Pull previous image or restore from backup
```

---

### 2. Data Pipelines

**Services:**
- Scraper (collects data from external sources)
- Processing (processes and transforms data)

**Manual Deployment:**
```bash
# Via GitHub Actions
Actions → deploy-pipelines.yml → Run workflow
  - Select environment (dev/prod)
  - Choose services:
    • All Pipeline Services
    • Scraper Only
    • Processing Only
```

**Dependencies:**
- ✅ MongoDB must be running
- ✅ Redis must be running
- ✅ Qdrant must be running (for processing)

**Deployment checks:**
The workflow automatically validates dependencies before deploying.

**Rollback:**
```bash
cd /home/user/git-query
docker-compose -f infrastructure/docker/docker-compose.base.yml \
  -f infrastructure/docker/docker-compose.pipelines.yml \
  down

# Restart with previous version
docker-compose -f infrastructure/docker/docker-compose.base.yml \
  -f infrastructure/docker/docker-compose.pipelines.yml \
  up -d
```

**Troubleshooting:**
- **Scraper not starting:** Check MongoDB and Redis connectivity
- **Processing errors:** Verify Qdrant is healthy and accessible
- **High memory usage:** Adjust `BATCH_SIZE` environment variable

---

### 3. MCP (Model Context Protocol) Servers

**Services:**
- MCP Server (provides git-query specific tools to AI agents)

**Manual Deployment:**
```bash
# Via GitHub Actions
Actions → deploy-mcp.yml → Run workflow
  - Select environment (dev/prod)
```

**Dependencies:**
- ✅ MongoDB must be running
- ✅ Redis must be running
- ✅ Qdrant must be running

**Health Check:**
```bash
curl http://localhost:8090/health
```

**Rollback:**
```bash
cd /home/user/git-query
docker-compose -f infrastructure/docker/docker-compose.base.yml \
  -f infrastructure/docker/docker-compose.mcp.yml \
  down

# Restart with previous version
docker-compose -f infrastructure/docker/docker-compose.base.yml \
  -f infrastructure/docker/docker-compose.mcp.yml \
  up -d
```

**Troubleshooting:**
- **MCP not accessible:** Check if port 8090 is exposed and not conflicting
- **Database connection errors:** Verify database credentials in .env file
- **Tool registration failures:** Check Qdrant vector store is populated

---

### 4. Recommendation Engine

**Services:**
- Recommender (serves real-time recommendations)

**Manual Deployment:**
```bash
# Via GitHub Actions
Actions → deploy-reco.yml → Run workflow
  - Select environment (dev/prod)
```

**Dependencies:**
- ✅ MongoDB must be running
- ✅ Redis must be running
- ✅ Qdrant must be running
- ⚠️ Trained model should exist (run training workflow first)

**Health Check:**
```bash
curl http://localhost:8095/health
```

**Model Loading:**
The recommender loads models from `/app/models` volume. Ensure training has completed successfully and models are available.

**Rollback:**
```bash
cd /home/user/git-query
docker-compose -f infrastructure/docker/docker-compose.base.yml \
  -f infrastructure/docker/docker-compose.reco.yml \
  down

# Restart with previous version
docker-compose -f infrastructure/docker/docker-compose.base.yml \
  -f infrastructure/docker/docker-compose.reco.yml \
  up -d
```

**Troubleshooting:**
- **Model not found:** Run training workflow to generate models
- **Slow inference:** Adjust `INFERENCE_BATCH_SIZE` environment variable
- **High memory usage:** Reduce batch size or model size
- **Cold start issues:** Models load on first request (30-60s delay expected)

---

### 5. ML Training (Batch Job)

**Services:**
- Training (trains recommendation models)

**Manual Deployment:**
```bash
# Via GitHub Actions
Actions → deploy-training.yml → Run workflow
  - Select environment (dev/prod)
  - Set hyperparameters:
    • Epochs (default: 10)
    • Batch Size (default: 64)
    • Learning Rate (default: 0.001)
```

**Scheduled Runs:**
Training runs automatically every Sunday at 2 AM UTC.

**Dependencies:**
- ✅ MongoDB must be running (training data source)
- ✅ Qdrant must be running (vector embeddings)

**Execution Time:**
- Small dataset: 15-30 minutes
- Medium dataset: 1-2 hours
- Large dataset: 2-4 hours

**Output:**
Trained models are stored in `git-query-training-models` volume and can be accessed by the recommender service.

**Monitoring Training:**
```bash
# View training logs in real-time
docker logs -f git-query-training

# Check model outputs
docker run --rm -v git-query-training-models:/models alpine ls -lh /models
```

**Troubleshooting:**
- **Training fails immediately:** Check if sufficient data exists in MongoDB
- **OOM (Out of Memory):** Reduce `TRAINING_BATCH_SIZE`
- **Training stuck:** Check logs for data loading issues
- **Poor model performance:** Adjust hyperparameters (epochs, learning rate)

**Rollback:**
Training is a batch job and doesn't require rollback. If a training run produces poor results:
1. Keep previous models in the volume
2. Don't deploy the new models to the recommender
3. Re-run training with different hyperparameters

---

## General Deployment Best Practices

### Pre-Deployment Checklist

- [ ] Review changes in the codebase
- [ ] Check current system health
- [ ] Verify dependencies are running
- [ ] Backup critical data (if applicable)
- [ ] Announce deployment in team chat
- [ ] Have rollback plan ready

### During Deployment

- [ ] Monitor workflow execution in GitHub Actions
- [ ] Watch service logs on server
- [ ] Check health endpoints after deployment
- [ ] Verify basic functionality

### Post-Deployment

- [ ] Run smoke tests
- [ ] Monitor error rates for 15 minutes
- [ ] Check resource usage (CPU, memory)
- [ ] Confirm with team that services are working
- [ ] Document any issues encountered

### Emergency Rollback

If a deployment causes critical issues:

1. **Immediate Action:**
   ```bash
   docker-compose -f docker-compose.base.yml -f docker-compose.[service].yml down
   ```

2. **Restore Previous Version:**
   - Re-run workflow with previous git commit/tag
   - Or manually pull previous Docker image

3. **Verify Recovery:**
   - Check health endpoints
   - Run smoke tests
   - Monitor error rates

4. **Post-Incident:**
   - Document what went wrong
   - Create post-mortem
   - Fix issues before next deployment

---

## Deployment Coordination

### Who Deploys What

**Infrastructure Team:**
- Databases (MongoDB, Cosmos, Qdrant, Redis)
- DB Query API
- Nginx

**Data Team:**
- Pipelines (scraper, processing)
- Training jobs

**ML Team:**
- MCP servers
- Recommendation engine
- Training hyperparameter tuning

**Application Team:**
- Gateway, client apps, web frontend (when added)

### Communication Protocol

1. **Announce before deploying:** Post in team channel 15 minutes before
2. **Update during deployment:** Post progress updates
3. **Confirm after deployment:** Post success/failure with relevant metrics
4. **Coordinate cross-team deployments:** Use thread for real-time coordination

---

## Monitoring & Alerts

### Key Metrics to Watch

**Infrastructure:**
- Database connection pool usage
- Redis memory usage
- Qdrant query latency

**Pipelines:**
- Scraper job success rate
- Processing throughput
- Queue depths

**ML Services:**
- MCP tool call success rate
- Recommender inference latency
- Model accuracy metrics

**System:**
- CPU usage
- Memory usage
- Disk usage
- Network I/O

### Health Check Endpoints

```bash
# DB API
curl http://localhost:8080/health

# MCP Server
curl http://localhost:8090/health

# Recommender
curl http://localhost:8095/health

# Nginx
curl http://localhost/health
```

---

## Helpful Commands

### View All Running Services
```bash
docker ps --filter "name=git-query-"
```

### View Service Logs
```bash
docker logs -f git-query-[service-name]
```

### Check Resource Usage
```bash
docker stats
```

### Clean Up Unused Resources
```bash
docker system prune -af --volumes=false
```

### Backup Database
```bash
docker exec git-query-mongodb mongodump --out /backup
```

### Inspect Volumes
```bash
docker volume ls | grep git-query
docker run --rm -v git-query-[volume-name]:/data alpine ls -lh /data
```

---

## FAQ

**Q: Can I deploy multiple services at once?**
A: Yes, but ensure you respect dependencies. Deploy infrastructure first, then data/ML services.

**Q: How do I know if a deployment was successful?**
A: Check the GitHub Actions workflow status, verify health endpoints, and monitor logs.

**Q: What if a deployment is taking too long?**
A: Check the workflow logs. Some operations (training, image builds) naturally take longer.

**Q: Can I manually deploy without GitHub Actions?**
A: Yes, SSH into the server and use docker-compose commands directly.

**Q: How do I share trained models between training and recommender?**
A: They share the same Docker volume (`git-query-training-models`). Deploy recommender after training completes.

**Q: What's the difference between dev and prod deployments?**
A: Dev auto-deploys on push; prod requires manual dispatch. Prod should have stricter validation.

**Q: How often should I run training?**
A: It depends on data freshness requirements. Default is weekly, but adjust based on your needs.
