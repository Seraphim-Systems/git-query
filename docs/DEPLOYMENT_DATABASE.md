# Database Deployment Guide

This guide covers deploying the Git-Query databases to a Hetzner server with query endpoints.

## Architecture

```
Internet → External NPM (Port 443) → Hetzner Server (Port 80) → Nginx → DB Query API
                                                                         ↓
                                                      [MongoDB | Cosmos | Qdrant | Redis]
```

## Prerequisites

1. Hetzner server with Docker and Docker Compose installed
2. External Nginx Proxy Manager (NPM) configured
3. GitHub repository secrets configured
4. Domain name configured in DNS

## GitHub Secrets Required

Configure these secrets in your GitHub repository:

```bash
# Server Access
HETZNER_HOST=your-server-ip
HETZNER_USER=root
HETZNER_SSH_KEY=<your-ssh-private-key>

# Database Credentials
MONGO_USER=admin
MONGO_PASSWORD=<strong-password>
REDIS_PASSWORD=<strong-password>
QDRANT_API_KEY=<api-key>

# API Configuration
DATA_INGESTION_API_KEY=<strong-api-key>
SERVER_NAME=db.yourdomain.com
```

## Deployment Steps

### 1. Manual Deployment via GitHub Actions

1. Go to your repository on GitHub
2. Navigate to **Actions** tab
3. Select **Deploy Databases to Hetzner** workflow
4. Click **Run workflow**
5. Select which databases to deploy:
   - ✅ Deploy MongoDB
   - ✅ Deploy Cosmos DB
   - ✅ Deploy Qdrant
   - ✅ Deploy Redis
6. Click **Run workflow**

### 2. Configure External NPM

In your Nginx Proxy Manager:

1. Add a new Proxy Host
2. Configure:
   - **Domain Names**: `db.yourdomain.com`
   - **Scheme**: `http`
   - **Forward Hostname/IP**: `<hetzner-server-ip>`
   - **Forward Port**: `80`
   - **Cache Assets**: Enabled
   - **Block Common Exploits**: Enabled
   - **Websockets Support**: Enabled

3. SSL Tab:
   - **SSL Certificate**: Select or generate Let's Encrypt cert
   - **Force SSL**: Enabled
   - **HTTP/2 Support**: Enabled

4. Advanced Tab (optional rate limiting):
   ```nginx
   # Rate limiting
   limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
   limit_req zone=api_limit burst=20 nodelay;
   
   # Additional security headers
   add_header X-Content-Type-Options "nosniff" always;
   add_header X-Frame-Options "SAMEORIGIN" always;
   add_header X-XSS-Protection "1; mode=block" always;
   ```

### 3. Verify Deployment

After deployment, verify the services:

```bash
# Check health endpoint
curl https://db.yourdomain.com/health

# Check API documentation
curl https://db.yourdomain.com/docs

# Test MongoDB query
curl -X POST https://db.yourdomain.com/api/mongodb/collections
```

## Server Setup (One-Time)

SSH into your Hetzner server and prepare the environment:

```bash
# Create deployment directory (using your username)
mkdir -p /home/$USER/git-query

# Install Docker and Docker Compose (if not already installed)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt-get update
apt-get install -y docker-compose-plugin

# Configure firewall
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 22/tcp
ufw enable
```

## Manual Deployment (Without GitHub Actions)

If you prefer to deploy manually:

```bash
# On your local machine
cd /path/to/git-query

# Create .env file
cat > .env <<EOF
MONGO_USER=admin
MONGO_PASSWORD=your-password
MONGO_DB=gitquery
REDIS_PASSWORD=your-redis-password
QDRANT_API_KEY=your-qdrant-key
DATA_INGESTION_API_KEY=your-api-key
SERVER_NAME=db.yourdomain.com
NGINX_PORT=80
EOF

# Copy files to server
rsync -avz --exclude='.git' \
  infrastructure/ \
  src/storage/ \
  web/nginx-db.conf.template \
  .env \
  user@your-server:/home/user/git-query/

# SSH to server
ssh user@your-server

# Deploy
cd /home/$USER/git-query
docker-compose -f infrastructure/docker/docker-compose.db.yml up -d

# Check status
docker-compose -f infrastructure/docker/docker-compose.db.yml ps
```

## Database Access

### Interactive API Documentation

Visit `https://db.yourdomain.com/docs` for interactive Swagger UI documentation.

### Available Endpoints

#### Public Endpoints (No Authentication)
- `GET /health` - Health check
- `POST /api/mongodb/query` - Query MongoDB
- `GET /api/mongodb/collections` - List collections
- `POST /api/qdrant/search` - Search vectors
- `GET /api/qdrant/collections` - List collections
- `GET /api/redis/get/{key}` - Get Redis key
- `GET /api/redis/keys` - List Redis keys

#### Authenticated Endpoints (Require X-API-Key Header)
- `POST /api/mongodb/insert` - Insert documents
- `POST /api/qdrant/insert` - Insert vectors
- `POST /api/redis/set` - Set Redis key
- `POST /api/batch/insert` - Batch insert across databases

### Authentication

For write operations, include the API key in the header:

```bash
curl -X POST https://db.yourdomain.com/api/mongodb/insert \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{...}'
```

## Monitoring and Maintenance

### View Logs

```bash
# On Hetzner server
cd /home/$USER/git-query

# View all logs
docker-compose -f infrastructure/docker/docker-compose.db.yml logs -f

# View specific service logs
docker-compose -f infrastructure/docker/docker-compose.db.yml logs -f mongodb
docker-compose -f infrastructure/docker/docker-compose.db.yml logs -f db-query-api
docker-compose -f infrastructure/docker/docker-compose.db.yml logs -f nginx
```

### Database Backups

#### MongoDB Backup
```bash
docker exec git-query-mongodb mongodump \
  --username=admin \
  --password=your-password \
  --authenticationDatabase=admin \
  --out=/backup/mongodb-$(date +%Y%m%d)

# Copy backup locally
docker cp git-query-mongodb:/backup ./backups/
```

#### Qdrant Backup
```bash
# Create snapshot
curl -X POST https://db.yourdomain.com/api/qdrant/collections/repository_embeddings/snapshots

# List snapshots
docker exec git-query-qdrant ls -la /qdrant/snapshots/
```

#### Redis Backup
```bash
# Create RDB snapshot
docker exec git-query-redis redis-cli save

# Copy RDB file
docker cp git-query-redis:/data/dump.rdb ./backups/redis-$(date +%Y%m%d).rdb
```

### Update Deployment

To update the databases:

```bash
# Re-run the GitHub Actions workflow, OR:

# SSH to server
cd /home/$USER/git-query
docker-compose -f infrastructure/docker/docker-compose.db.yml pull
docker-compose -f infrastructure/docker/docker-compose.db.yml up -d
docker system prune -af
```

## Scaling Considerations

### Horizontal Scaling

For production workloads, consider:

1. **MongoDB Replica Set**: Deploy MongoDB as a replica set for high availability
2. **Redis Cluster**: Use Redis Cluster for distributed caching
3. **Qdrant Cluster**: Deploy Qdrant in cluster mode for large-scale vector search
4. **Load Balancing**: Add multiple API instances behind NPM

### Resource Allocation

Recommended resources per service:

```yaml
# Add to docker-compose.db.yml
services:
  mongodb:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
  
  qdrant:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Security Best Practices

1. **API Key Rotation**: Regularly rotate the `DATA_INGESTION_API_KEY`
2. **Database Credentials**: Use strong passwords and rotate regularly
3. **Network Security**: Use firewall rules to restrict access
4. **SSL/TLS**: Always use HTTPS through NPM
5. **Monitoring**: Set up monitoring and alerting for unusual activity
6. **Rate Limiting**: Configure rate limits in NPM
7. **Backup Encryption**: Encrypt backups at rest

## Troubleshooting

### API Not Responding

```bash
# Check container status
docker-compose -f infrastructure/docker/docker-compose.db.yml ps

# Restart API container
docker-compose -f infrastructure/docker/docker-compose.db.yml restart db-query-api

# Check API logs
docker logs git-query-db-api
```

### Database Connection Issues

```bash
# Test MongoDB connection
docker exec -it git-query-mongodb mongosh \
  -u admin -p your-password --authenticationDatabase admin

# Test Redis connection
docker exec -it git-query-redis redis-cli -a your-redis-password ping

# Test Qdrant connection
curl http://localhost:6333/collections
```

### NPM Not Forwarding Requests

1. Check NPM logs
2. Verify Hetzner server firewall allows port 80
3. Test direct connection: `curl http://<hetzner-ip>/health`
4. Check NPM proxy host configuration

## Support

For additional help:
- Documentation: `/docs/DATA_SCIENTIST_GUIDE.md`
- API Docs: `https://db.yourdomain.com/docs`
- GitHub Issues: https://github.com/Seraphim-Systems/git-query/issues
