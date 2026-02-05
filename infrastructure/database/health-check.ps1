# Database Health Check Script for Windows PowerShell
# Verifies all databases are running and accepting connections

Write-Host "`n🔍 Checking database health for Git-Query...`n" -ForegroundColor Cyan

function Test-Service {
    param(
        [string]$ServiceName,
        [scriptblock]$CheckCommand
    )
    
    try {
        $result = & $CheckCommand
        if ($LASTEXITCODE -eq 0 -or $result) {
            Write-Host "✓ $ServiceName is healthy" -ForegroundColor Green
            return $true
        } else {
            Write-Host "✗ $ServiceName is not responding" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "✗ $ServiceName error: $_" -ForegroundColor Red
        return $false
    }
}

# Check if Docker is running
try {
    docker info 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Docker is running" -ForegroundColor Green
    } else {
        Write-Host "✗ Docker is not running" -ForegroundColor Red
        Write-Host "  Please start Docker Desktop and try again`n" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "✗ Docker is not running" -ForegroundColor Red
    Write-Host "  Please start Docker Desktop and try again`n" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Check MongoDB
Write-Host "Checking MongoDB..." -ForegroundColor Cyan
Test-Service "MongoDB" {
    docker exec git-query-mongodb mongosh --quiet --eval "db.adminCommand('ping')" 2>&1 | Out-Null
}
Write-Host ""

# Check Qdrant
Write-Host "Checking Qdrant..." -ForegroundColor Cyan
Test-Service "Qdrant" {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:6333/collections" -UseBasicParsing -ErrorAction SilentlyContinue
        $response.StatusCode -eq 200
    } catch {
        $false
    }
}
Write-Host ""

# Check Redis
Write-Host "Checking Redis..." -ForegroundColor Cyan
Test-Service "Redis" {
    $result = docker exec git-query-redis redis-cli ping 2>&1
    $result -match "PONG"
}
Write-Host ""

# Check Cosmos DB Emulator (optional, might be slow)
Write-Host "Checking Cosmos DB Emulator (optional)..." -ForegroundColor Cyan
try {
    $cosmosResponse = docker ps --filter "name=git-query-cosmos-db" --format "{{.Status}}"
    if ($cosmosResponse -match "Up") {
        Write-Host "✓ Cosmos DB Emulator is running" -ForegroundColor Green
    } else {
        Write-Host "○ Cosmos DB Emulator is not running (optional)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "○ Cosmos DB Emulator is not running (optional)" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host ""

# Summary
Write-Host "📊 Database Container Status:`n" -ForegroundColor Cyan
docker ps --filter "name=git-query-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
Write-Host ""

Write-Host "💡 Quick Commands:" -ForegroundColor Yellow
Write-Host "  View logs:          docker logs -f [container-name]"
Write-Host "  Restart service:    docker restart [container-name]"
Write-Host "  Stop all:           docker compose -f infrastructure/docker/docker-compose.yml down"
Write-Host "  Start dev:          docker compose -f infrastructure/docker/docker-compose.yml -f infrastructure/docker/docker-compose.dev.yml up -d"
Write-Host "  Start prod:         docker compose -f infrastructure/docker/docker-compose.yml -f infrastructure/docker/docker-compose.prod.yml up -d"
Write-Host ""

Write-Host "📚 Documentation:" -ForegroundColor Yellow
Write-Host "  See README.md for detailed database information"
Write-Host "  See docs/ for deployment instructions"
Write-Host ""

Write-Host "💡 Quick Commands:" -ForegroundColor Yellow
Write-Host "  View logs:          docker logs -f [container-name]"
Write-Host "  Restart service:    docker restart [container-name]"
Write-Host "  Stop all:           docker-compose -f docker-compose.all.yml down"
Write-Host "  Start all:          docker-compose -f docker-compose.all.yml up -d"
Write-Host ""

Write-Host "📚 Documentation:" -ForegroundColor Yellow
Write-Host "  See README.md for detailed database information"
Write-Host "  See DB_Deploy/deployment-guide.md for deployment instructions"
Write-Host ""
