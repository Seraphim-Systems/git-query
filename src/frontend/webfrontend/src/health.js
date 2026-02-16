// Health monitoring functionality
class HealthMonitor {
    constructor() {
        this.services = [
            { id: 'gateway', name: 'Gateway', icon: '🌐', endpoint: '/api/health' },
            { id: 'mongodb', name: 'MongoDB', icon: '🍃', endpoint: '/api/health/mongodb' },
            { id: 'redis', name: 'Redis', icon: '💾', endpoint: '/api/health/redis' },
            { id: 'qdrant', name: 'Qdrant', icon: '🔍', endpoint: '/api/health/qdrant' },
            { id: 'mcp', name: 'MCP Server', icon: '🔌', endpoint: 'http://localhost:8090/health', isDirect: true }
        ];
        
        this.healthData = {};
        this.isPanelOpen = false;
        this.refreshInterval = null;
        this.AUTO_REFRESH_INTERVAL = 30000; // 30 seconds
        
        this.init();
    }
    
    init() {
        // Create the health panel HTML
        this.createHealthPanel();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Initial health check
        this.checkAllHealth();
        
        // Start auto-refresh
        this.startAutoRefresh();
    }
    
    createHealthPanel() {
        const toggleBtn = document.createElement('button');
        toggleBtn.id = 'healthToggleBtn';
        toggleBtn.className = 'health-toggle-btn';
        toggleBtn.innerHTML = '🏥';
        toggleBtn.title = 'System Health';
        
        const panel = document.createElement('div');
        panel.id = 'healthPanel';
        panel.className = 'health-panel collapsed';
        
        panel.innerHTML = `
            <div class="health-header">
                <h3 class="health-title">System Health</h3>
                <button id="healthRefresh" class="health-refresh">🔄 Refresh</button>
            </div>
            
            <div id="healthOverall" class="health-overall checking">
                <div class="health-overall-status">Checking...</div>
                <div class="health-overall-count">0/5 services healthy</div>
            </div>
            
            <div id="healthServices" class="health-services">
                ${this.services.map(service => `
                    <div id="health-${service.id}" class="health-service checking">
                        <div class="health-service-name">
                            <span class="health-service-icon">${service.icon}</span>
                            ${service.name}
                        </div>
                        <div class="health-status">
                            <span class="health-status-dot checking"></span>
                            <span class="health-status-text">Checking...</span>
                        </div>
                    </div>
                `).join('')}
            </div>
            
            <div id="healthTimestamp" class="health-timestamp">
                Last checked: Never
            </div>
        `;
        
        document.body.appendChild(toggleBtn);
        document.body.appendChild(panel);
    }
    
    setupEventListeners() {
        const toggleBtn = document.getElementById('healthToggleBtn');
        const refreshBtn = document.getElementById('healthRefresh');
        
        toggleBtn.addEventListener('click', () => this.togglePanel());
        refreshBtn.addEventListener('click', () => this.checkAllHealth());
        
        // Close panel when clicking outside
        document.addEventListener('click', (e) => {
            const panel = document.getElementById('healthPanel');
            const toggleBtn = document.getElementById('healthToggleBtn');
            
            if (this.isPanelOpen && 
                !panel.contains(e.target) && 
                !toggleBtn.contains(e.target)) {
                this.togglePanel();
            }
        });
    }
    
    togglePanel() {
        const panel = document.getElementById('healthPanel');
        const toggleBtn = document.getElementById('healthToggleBtn');
        
        this.isPanelOpen = !this.isPanelOpen;
        
        if (this.isPanelOpen) {
            panel.classList.remove('collapsed');
            toggleBtn.classList.add('panel-open');
            // Check health when opening
            this.checkAllHealth();
        } else {
            panel.classList.add('collapsed');
            toggleBtn.classList.remove('panel-open');
        }
    }
    
    async checkAllHealth() {
        const refreshBtn = document.getElementById('healthRefresh');
        refreshBtn.classList.add('spinning');
        
        // Reset all services to checking state
        this.services.forEach(service => {
            this.updateServiceStatus(service.id, 'checking', 'Checking...');
        });
        
        // Check all services
        const promises = this.services.map(service => this.checkServiceHealth(service));
        await Promise.all(promises);
        
        // Update overall status
        this.updateOverallStatus();
        
        // Update timestamp
        this.updateTimestamp();
        
        refreshBtn.classList.remove('spinning');
    }
    
    async checkServiceHealth(service) {
        try {
            const isDirectEndpoint = service.isDirect === true;
            const fetchUrl = isDirectEndpoint ? service.endpoint : service.endpoint;
            
            const response = await fetch(fetchUrl, {
                method: 'GET',
                headers: isDirectEndpoint ? {} : {
                    'Authorization': `Bearer ${localStorage.getItem('authToken') || ''}`
                }
            });
            
            const isHealthy = response.ok;
            let data = {};
            
            try {
                data = await response.json();
            } catch (e) {
                // If JSON parsing fails, just use the response status
            }
            
            this.healthData[service.id] = {
                healthy: isHealthy,
                status: response.status,
                data: data,
                timestamp: new Date()
            };
            
            const statusText = isHealthy ? 'Healthy' : 'Unhealthy';
            const statusClass = isHealthy ? 'healthy' : 'unhealthy';
            
            this.updateServiceStatus(service.id, statusClass, statusText);
            
        } catch (error) {
            console.error(`Health check failed for ${service.name}:`, error);
            
            this.healthData[service.id] = {
                healthy: false,
                error: error.message,
                timestamp: new Date()
            };
            
            this.updateServiceStatus(service.id, 'unhealthy', 'Error');
        }
    }
    
    updateServiceStatus(serviceId, statusClass, statusText) {
        const serviceElement = document.getElementById(`health-${serviceId}`);
        if (!serviceElement) return;
        
        // Update service container class
        serviceElement.className = `health-service ${statusClass}`;
        
        // Update status dot and text
        const statusDot = serviceElement.querySelector('.health-status-dot');
        const statusTextElement = serviceElement.querySelector('.health-status-text');
        
        if (statusDot) {
            statusDot.className = `health-status-dot ${statusClass}`;
        }
        
        if (statusTextElement) {
            statusTextElement.textContent = statusText;
        }
    }
    
    updateOverallStatus() {
        const overallElement = document.getElementById('healthOverall');
        if (!overallElement) return;
        
        const healthyCount = Object.values(this.healthData).filter(d => d.healthy).length;
        const totalCount = this.services.length;
        
        const isAllHealthy = healthyCount === totalCount;
        const isSomeHealthy = healthyCount > 0;
        
        let statusClass, statusText;
        
        if (isAllHealthy) {
            statusClass = 'healthy';
            statusText = 'All Systems Operational';
        } else if (isSomeHealthy) {
            statusClass = 'unhealthy';
            statusText = 'Partial Outage';
        } else {
            statusClass = 'unhealthy';
            statusText = 'System Down';
        }
        
        overallElement.className = `health-overall ${statusClass}`;
        overallElement.querySelector('.health-overall-status').textContent = statusText;
        overallElement.querySelector('.health-overall-count').textContent = 
            `${healthyCount}/${totalCount} services healthy`;
    }
    
    updateTimestamp() {
        const timestampElement = document.getElementById('healthTimestamp');
        if (!timestampElement) return;
        
        const now = new Date();
        const timeString = now.toLocaleTimeString();
        
        timestampElement.textContent = `Last checked: ${timeString}`;
    }
    
    startAutoRefresh() {
        // Clear any existing interval
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        
        // Set up new interval
        this.refreshInterval = setInterval(() => {
            // Only auto-refresh if panel is open
            if (this.isPanelOpen) {
                this.checkAllHealth();
            }
        }, this.AUTO_REFRESH_INTERVAL);
    }
    
    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }
    
    destroy() {
        this.stopAutoRefresh();
        
        const panel = document.getElementById('healthPanel');
        const toggleBtn = document.getElementById('healthToggleBtn');
        
        if (panel) panel.remove();
        if (toggleBtn) toggleBtn.remove();
    }
}

// Initialize health monitor when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Only initialize on home page
    if (window.location.pathname.includes('home.html') || 
        window.location.pathname === '/' && localStorage.getItem('authToken')) {
        window.healthMonitor = new HealthMonitor();
    }
});

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (window.healthMonitor) {
        window.healthMonitor.destroy();
    }
});
