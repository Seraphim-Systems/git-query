#!/bin/bash
# Database Health Check Script for Linux/Mac
# Verifies all databases are running and accepting connections

echo -e "\n🔍 Checking database health for Git-Query...\n"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

check_service() {
    local service=$1
    local check_cmd=$2
    
    if eval "$check_cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $service is healthy"
        return 0
    else
        echo -e "${RED}✗${NC} $service is not responding"
        return 1
    fi
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}✗${NC} Docker is not running"
    echo -e "  ${YELLOW}Please start Docker and try again${NC}\n"
    exit 1
fi
echo -e "${GREEN}✓${NC} Docker is running"
echo ""

# Check MongoDB
echo -e "${CYAN}Checking MongoDB...${NC}"
check_service "MongoDB" "docker exec gitquery-mongodb mongosh --quiet --eval 'db.adminCommand(\"ping\")'"
echo ""

# Check Qdrant
echo -e "${CYAN}Checking Qdrant...${NC}"
check_service "Qdrant" "curl -s -o /dev/null -w '%{http_code}' http://localhost:6333/collections | grep -q 200"
echo ""

# Check Redis
echo -e "${CYAN}Checking Redis...${NC}"
check_service "Redis" "docker exec gitquery-redis redis-cli ping | grep -q PONG"
echo ""

# Check PostgreSQL
echo -e "${CYAN}Checking PostgreSQL...${NC}"
check_service "PostgreSQL" "docker exec gitquery-postgres pg_isready -U user"
echo ""

# Check Cosmos DB Emulator (optional)
echo -e "${CYAN}Checking Cosmos DB Emulator (optional)...${NC}"
if docker ps --filter "name=gitquery-cosmos-db" | grep -q "Up"; then
    echo -e "${GREEN}✓${NC} Cosmos DB Emulator is running"
else
    echo -e "${YELLOW}○${NC} Cosmos DB Emulator is not running (optional)"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Summary
echo ""
echo -e "${CYAN}📊 Database Container Status:${NC}\n"
docker ps --filter "name=gitquery-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""

echo -e "${YELLOW}💡 Quick Commands:${NC}"
echo "  View logs:          docker logs -f [container-name]"
echo "  Restart service:    docker restart [container-name]"
echo "  Stop all:           docker-compose -f docker-compose.all.yml down"
echo "  Start all:          docker-compose -f docker-compose.all.yml up -d"
echo ""

echo -e "${YELLOW}📚 Documentation:${NC}"
echo "  See README.md for detailed database information"
echo "  See DB_Deploy/deployment-guide.md for deployment instructions"
echo ""
