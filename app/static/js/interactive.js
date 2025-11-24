// Interactive JavaScript for Reddit Sentiment Analyzer Pro

class DashboardInteractivity {
    constructor() {
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupCharts();
        this.setupRealTimeUpdates();
        this.setupExportHandlers();
    }

    setupEventListeners() {
        // Metric card interactions
        document.querySelectorAll('.metric-card').forEach(card => {
            card.addEventListener('click', this.handleMetricClick.bind(this));
        });

        // Filter controls
        document.querySelectorAll('.filter-control').forEach(control => {
            control.addEventListener('change', this.handleFilterChange.bind(this));
        });

        // Export buttons
        document.querySelectorAll('.export-btn').forEach(btn => {
            btn.addEventListener('click', this.handleExportClick.bind(this));
        });

        // Real-time update toggles
        document.querySelectorAll('.real-time-toggle').forEach(toggle => {
            toggle.addEventListener('change', this.handleRealTimeToggle.bind(this));
        });
    }

    setupCharts() {
        // Initialize any dynamic charts
        this.initializeSentimentChart();
        this.initializeEmotionChart();
        this.initializeTrendChart();
    }

    setupRealTimeUpdates() {
        // Set up WebSocket or polling for real-time updates
        this.setupWebSocket();
        this.startDataPolling();
    }

    setupExportHandlers() {
        // Handle various export formats
        this.setupCSVExport();
        this.setupJSONExport();
        this.setupPDFExport();
    }

    handleMetricClick(event) {
        const card = event.currentTarget;
        const metricType = card.dataset.metricType;
        
        this.showMetricDetails(metricType);
        this.highlightRelatedCharts(metricType);
    }

    handleFilterChange(event) {
        const filterType = event.target.name;
        const filterValue = event.target.value;
        
        this.applyFilters(filterType, filterValue);
        this.updateCharts();
    }

    handleExportClick(event) {
        event.preventDefault();
        const format = event.currentTarget.dataset.format;
        this.exportData(format);
    }

    handleRealTimeToggle(event) {
        const enabled = event.target.checked;
        
        if (enabled) {
            this.enableRealTimeUpdates();
        } else {
            this.disableRealTimeUpdates();
        }
    }

    // Chart Management
    initializeSentimentChart() {
        // Initialize sentiment distribution chart
        const sentimentData = this.getSentimentData();
        this.renderSentimentChart(sentimentData);
    }

    initializeEmotionChart() {
        // Initialize emotion distribution chart
        const emotionData = this.getEmotionData();
        this.renderEmotionChart(emotionData);
    }

    initializeTrendChart() {
        // Initialize trend chart
        const trendData = this.getTrendData();
        this.renderTrendChart(trendData);
    }

    renderSentimentChart(data) {
        // Use Chart.js or similar to render sentiment chart
        console.log('Rendering sentiment chart with data:', data);
        // Implementation would use a charting library
    }

    renderEmotionChart(data) {
        console.log('Rendering emotion chart with data:', data);
        // Implementation would use a charting library
    }

    renderTrendChart(data) {
        console.log('Rendering trend chart with data:', data);
        // Implementation would use a charting library
    }

    updateCharts() {
        // Update all charts with current data
        this.initializeSentimentChart();
        this.initializeEmotionChart();
        this.initializeTrendChart();
    }

    // Data Management
    getSentimentData() {
        // Fetch or calculate sentiment data
        return {
            positive: 45,
            negative: 25,
            neutral: 30
        };
    }

    getEmotionData() {
        // Fetch or calculate emotion data
        return {
            joy: 120,
            anger: 45,
            sadness: 30,
            fear: 25,
            surprise: 40,
            love: 60
        };
    }

    getTrendData() {
        // Fetch or calculate trend data
        return [
            { date: '2024-01-01', positive: 40, negative: 30, neutral: 30 },
            { date: '2024-01-02', positive: 45, negative: 25, neutral: 30 },
            { date: '2024-01-03', positive: 50, negative: 20, neutral: 30 }
        ];
    }

    // Filter Management
    applyFilters(filterType, filterValue) {
        console.log(`Applying filter: ${filterType} = ${filterValue}`);
        
        // Update data based on filters
        this.filteredData = this.applyDataFilters(this.rawData, filterType, filterValue);
        this.updateDisplay();
    }

    applyDataFilters(data, filterType, filterValue) {
        // Apply filters to dataset
        let filtered = [...data];
        
        switch (filterType) {
            case 'sentiment':
                filtered = filtered.filter(item => item.sentiment === filterValue);
                break;
            case 'emotion':
                filtered = filtered.filter(item => item.emotion === filterValue);
                break;
            case 'date-range':
                filtered = filtered.filter(item => this.isInDateRange(item.date, filterValue));
                break;
        }
        
        return filtered;
    }

    isInDateRange(date, range) {
        // Check if date is within specified range
        // Implementation depends on date format and range specification
        return true;
    }

    // Real-time Updates
    setupWebSocket() {
        // Set up WebSocket connection for real-time data
        try {
            this.ws = new WebSocket('ws://localhost:8000/ws');
            this.ws.onmessage = this.handleWebSocketMessage.bind(this);
            this.ws.onopen = this.handleWebSocketOpen.bind(this);
            this.ws.onclose = this.handleWebSocketClose.bind(this);
        } catch (error) {
            console.warn('WebSocket not available, falling back to polling');
            this.startDataPolling();
        }
    }

    handleWebSocketMessage(event) {
        const data = JSON.parse(event.data);
        this.processRealTimeData(data);
    }

    handleWebSocketOpen() {
        console.log('WebSocket connection established');
    }

    handleWebSocketClose() {
        console.log('WebSocket connection closed');
        this.startDataPolling();
    }

    startDataPolling() {
        // Fallback to polling if WebSockets aren't available
        this.pollingInterval = setInterval(() => {
            this.fetchLatestData();
        }, 5000); // Poll every 5 seconds
    }

    stopDataPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
        }
    }

    async fetchLatestData() {
        try {
            const response = await fetch('/api/latest-data');
            const data = await response.json();
            this.processRealTimeData(data);
        } catch (error) {
            console.error('Error fetching latest data:', error);
        }
    }

    processRealTimeData(data) {
        // Update dashboard with real-time data
        this.updateMetrics(data.metrics);
        this.updateChartsWithNewData(data.chartData);
        this.showNotification('Data updated', 'success');
    }

    enableRealTimeUpdates() {
        this.setupWebSocket();
        this.showNotification('Real-time updates enabled', 'success');
    }

    disableRealTimeUpdates() {
        if (this.ws) {
            this.ws.close();
        }
        this.stopDataPolling();
        this.showNotification('Real-time updates disabled', 'info');
    }

    // Export Functionality
    setupCSVExport() {
        // Set up CSV export functionality
        document.getElementById('export-csv').addEventListener('click', () => {
            this.exportToCSV();
        });
    }

    setupJSONExport() {
        // Set up JSON export functionality
        document.getElementById('export-json').addEventListener('click', () => {
            this.exportToJSON();
        });
    }

    setupPDFExport() {
        // Set up PDF export functionality
        document.getElementById('export-pdf').addEventListener('click', () => {
            this.exportToPDF();
        });
    }

    async exportData(format) {
        try {
            const data = await this.prepareExportData();
            
            switch (format) {
                case 'csv':
                    this.downloadCSV(data, `sentiment-analysis-${Date.now()}.csv`);
                    break;
                case 'json':
                    this.downloadJSON(data, `sentiment-analysis-${Date.now()}.json`);
                    break;
                case 'pdf':
                    this.generatePDF(data);
                    break;
            }
            
            this.showNotification(`Data exported as ${format.toUpperCase()}`, 'success');
        } catch (error) {
            console.error('Export failed:', error);
            this.showNotification('Export failed', 'error');
        }
    }

    async prepareExportData() {
        // Prepare data for export
        const response = await fetch('/api/export-data');
        return await response.json();
    }

    downloadCSV(data, filename) {
        const csv = this.convertToCSV(data);
        this.downloadFile(csv, filename, 'text/csv');
    }

    downloadJSON(data, filename) {
        const json = JSON.stringify(data, null, 2);
        this.downloadFile(json, filename, 'application/json');
    }

    convertToCSV(data) {
        // Convert data to CSV format
        const headers = Object.keys(data[0] || {});
        const csvRows = [headers.join(',')];
        
        for (const row of data) {
            const values = headers.map(header => {
                const escaped = ('' + row[header]).replace(/"/g, '""');
                return `"${escaped}"`;
            });
            csvRows.push(values.join(','));
        }
        
        return csvRows.join('\n');
    }

    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    generatePDF(data) {
        // Generate PDF report
        // This would integrate with a PDF generation library
        console.log('Generating PDF with data:', data);
    }

    // UI Helpers
    showMetricDetails(metricType) {
        // Show detailed view for specific metric
        const detailView = document.getElementById('metric-detail-view');
        if (detailView) {
            detailView.innerHTML = this.getMetricDetailHTML(metricType);
            detailView.classList.add('visible');
        }
    }

    getMetricDetailHTML(metricType) {
        // Generate HTML for metric detail view
        return `
            <div class="metric-detail">
                <h3>${metricType} Details</h3>
                <p>Detailed information about ${metricType} metric...</p>
            </div>
        `;
    }

    highlightRelatedCharts(metricType) {
        // Highlight charts related to the selected metric
        document.querySelectorAll('.chart-widget').forEach(chart => {
            if (chart.dataset.metric === metricType) {
                chart.classList.add('highlighted');
            } else {
                chart.classList.remove('highlighted');
            }
        });
    }

    updateDisplay() {
        // Update all displayed elements with filtered data
        this.updateMetricsDisplay();
        this.updateCharts();
        this.updateDataTables();
    }

    updateMetricsDisplay() {
        // Update metric cards with current data
        document.querySelectorAll('.metric-card').forEach(card => {
            const metricType = card.dataset.metricType;
            const value = this.calculateMetricValue(metricType);
            card.querySelector('.metric-value').textContent = value;
        });
    }

    calculateMetricValue(metricType) {
        // Calculate value for specific metric type
        const data = this.filteredData || this.rawData;
        // Implementation depends on metric type and data structure
        return data.length; // Placeholder
    }

    updateDataTables() {
        // Update data tables with filtered data
        document.querySelectorAll('.data-table').forEach(table => {
            this.populateDataTable(table, this.filteredData);
        });
    }

    populateDataTable(table, data) {
        // Populate data table with provided data
        const tbody = table.querySelector('tbody');
        tbody.innerHTML = '';
        
        data.forEach(item => {
            const row = this.createTableRow(item);
            tbody.appendChild(row);
        });
    }

    createTableRow(item) {
        // Create table row from data item
        const row = document.createElement('tr');
        // Implementation depends on data structure
        return row;
    }

    showNotification(message, type = 'info') {
        // Show notification to user
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <span class="notification-message">${message}</span>
            <button class="notification-close">&times;</button>
        `;
        
        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.remove();
        });
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }

    // Utility Methods
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.dashboard = new DashboardInteractivity();
});

// Utility functions for global use
window.AppUtils = {
    formatNumber: function(num) {
        return new Intl.NumberFormat().format(num);
    },
    
    formatPercentage: function(num) {
        return new Intl.NumberFormat('en-US', {
            style: 'percent',
            minimumFractionDigits: 1,
            maximumFractionDigits: 1
        }).format(num);
    },
    
    formatDate: function(date) {
        return new Intl.DateTimeFormat('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        }).format(new Date(date));
    },
    
    truncateText: function(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.substr(0, maxLength) + '...';
    },
    
    copyToClipboard: function(text) {
        navigator.clipboard.writeText(text).then(() => {
            console.log('Text copied to clipboard');
        }).catch(err => {
            console.error('Failed to copy text: ', err);
        });
    }
};