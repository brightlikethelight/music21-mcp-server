# Music21 MCP Server - Enterprise Deployment Guide

This guide covers deploying the Music21 MCP Server in production environments with enterprise-grade security, monitoring, and resilience features.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Security Setup](#security-setup)
4. [Monitoring Setup](#monitoring-setup)
5. [Deployment Options](#deployment-options)
6. [Configuration](#configuration)
7. [Health Checks & Monitoring](#health-checks--monitoring)
8. [Backup & Recovery](#backup--recovery)
9. [Troubleshooting](#troubleshooting)
10. [Compliance](#compliance)

## Architecture Overview

The enterprise edition includes:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Music21 MCP Server Enterprise              │
├─────────────────────────────────────────────────────────────────┤
│  Security Layer                                                │
│  ├─ OAuth 2.1 + PKCE Authentication                           │
│  ├─ Input Validation & Sanitization                           │
│  ├─ Real-time Security Monitoring                             │
│  └─ SOC2/GDPR Audit Logging                                   │
├─────────────────────────────────────────────────────────────────┤
│  Resilience Layer                                             │
│  ├─ Circuit Breakers                                          │
│  ├─ Bulkhead Isolation                                        │
│  ├─ Retry Patterns                                            │
│  └─ Timeout Management                                        │
├─────────────────────────────────────────────────────────────────┤
│  Monitoring Layer                                             │
│  ├─ Prometheus Metrics                                        │
│  ├─ OpenTelemetry Tracing                                     │
│  ├─ Health Checks                                             │
│  └─ Performance Analytics                                     │
├─────────────────────────────────────────────────────────────────┤
│  Music21 Core Tools                                           │
│  ├─ Analysis Tools (13 tools)                                 │
│  ├─ Composition Tools                                         │
│  └─ Export/Import Tools                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### System Requirements

- **CPU**: 4+ cores (8+ recommended for production)
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **Storage**: 50GB+ SSD (100GB+ recommended)
- **Network**: 1Gbps+ connection for high-throughput scenarios

### Software Requirements

- Python 3.10+
- Docker 20.10+ (for containerized deployment)
- PostgreSQL 13+ (optional, for persistent storage)
- Redis 6+ (for caching and session storage)
- Prometheus 2.40+ (for metrics)
- Grafana 9.0+ (for dashboards)

### Security Requirements

- SSL/TLS certificates
- OAuth 2.1 provider setup
- Firewall configuration
- Intrusion detection system
- Log aggregation system

## Security Setup

### 1. SSL/TLS Configuration

```bash
# Generate SSL certificates (use Let's Encrypt for production)
sudo certbot certonly --standalone -d music21-mcp.yourdomain.com

# Configure SSL in enterprise.env
SSL_ENABLED=true
SSL_CERT_PATH=/etc/letsencrypt/live/music21-mcp.yourdomain.com/fullchain.pem
SSL_KEY_PATH=/etc/letsencrypt/live/music21-mcp.yourdomain.com/privkey.pem
```

### 2. OAuth 2.1 Setup

```python
# Configure OAuth provider
from music21_mcp.security.oauth_provider import OAuth2Provider

oauth_provider = OAuth2Provider(issuer="https://music21-mcp.yourdomain.com")

# Register production client
oauth_provider.register_client(
    client_id="production_client_id",
    client_secret="secure_client_secret_from_env",
    redirect_uris=["https://yourapp.com/oauth/callback"],
    grant_types=["authorization_code", "refresh_token"],
    scope="music21:read music21:write music21:admin"
)
```

### 3. Firewall Configuration

```bash
# UFW configuration example
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 443/tcp  # HTTPS
sudo ufw allow 9090/tcp # Prometheus (internal only)
sudo ufw enable
```

### 4. Security Monitoring

```yaml
# Security monitoring configuration
security:
  enabled: true
  intrusion_detection: true
  anomaly_detection: true
  rate_limiting:
    requests_per_minute: 100
    burst_threshold: 20
    lockout_duration: 300
  threat_intelligence:
    enabled: true
    update_interval: 3600
```

## Monitoring Setup

### 1. Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'music21-mcp'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 30s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 2. Grafana Dashboards

Import the pre-built dashboards:

```bash
# Import Music21 MCP dashboard
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @grafana/music21-mcp-dashboard.json
```

### 3. OpenTelemetry Configuration

```yaml
# otel-collector.yml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:

exporters:
  jaeger:
    endpoint: jaeger:14250
    tls:
      insecure: true
  prometheus:
    endpoint: "0.0.0.0:8889"

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [jaeger]
    metrics:
      receivers: [otlp]
      processors: [batch]
      exporters: [prometheus]
```

## Deployment Options

### Option 1: Docker Compose (Recommended)

```yaml
# docker-compose.enterprise.yml
version: '3.8'

services:
  music21-mcp:
    build:
      context: .
      dockerfile: Dockerfile.enterprise
    ports:
      - "8080:8080"
    environment:
      - DEPLOYMENT_ENV=production
    env_file:
      - configs/enterprise.env
    volumes:
      - ./logs:/var/log/music21_mcp
      - ./data:/var/lib/music21_mcp
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --requirepass ${REDIS_PASSWORD}
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${DB_NAME}
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/alert_rules.yml:/etc/prometheus/alert_rules.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    restart: unless-stopped

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "14250:14250"
    environment:
      COLLECTOR_OTLP_ENABLED: true
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:
```

### Option 2: Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: music21-mcp-enterprise
  labels:
    app: music21-mcp
    version: enterprise
spec:
  replicas: 3
  selector:
    matchLabels:
      app: music21-mcp
  template:
    metadata:
      labels:
        app: music21-mcp
    spec:
      containers:
      - name: music21-mcp
        image: music21-mcp:enterprise
        ports:
        - containerPort: 8080
        env:
        - name: DEPLOYMENT_ENV
          value: "production"
        envFrom:
        - configMapRef:
            name: music21-mcp-config
        - secretRef:
            name: music21-mcp-secrets
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Option 3: Systemd Service

```ini
# /etc/systemd/system/music21-mcp.service
[Unit]
Description=Music21 MCP Server Enterprise
After=network.target
Requires=network.target

[Service]
Type=exec
User=music21mcp
Group=music21mcp
WorkingDirectory=/opt/music21-mcp
Environment=PYTHONPATH=/opt/music21-mcp
EnvironmentFile=/opt/music21-mcp/configs/enterprise.env
ExecStart=/opt/music21-mcp/venv/bin/python -m music21_mcp.server_enterprise
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=music21-mcp
KillMode=mixed
TimeoutStopSec=5

[Install]
WantedBy=multi-user.target
```

## Configuration

### Environment Variables

Copy and customize the enterprise configuration:

```bash
cp configs/enterprise.env /opt/music21-mcp/.env
```

Key configuration sections:

#### Security Configuration
```bash
SECURITY_ENABLED=true
REQUIRE_AUTH=true
OAUTH_CLIENT_ID=your_client_id
OAUTH_CLIENT_SECRET=your_client_secret
STRICT_VALIDATION=true
```

#### Performance Configuration
```bash
MAX_CONCURRENT_REQUESTS=50
REQUEST_TIMEOUT=30.0
MAX_MEMORY_GB=8.0
ENABLE_CACHING=true
```

#### Monitoring Configuration
```bash
METRICS_ENABLED=true
PROMETHEUS_PORT=9090
OTEL_ENABLED=true
AUDIT_LOG_ENABLED=true
```

### Circuit Breaker Configuration

```python
# Custom circuit breaker settings
circuit_breaker_config = {
    "music_analysis": {
        "failure_threshold": 5,
        "recovery_timeout": 60.0,
        "monitoring_window": 120.0
    },
    "score_import": {
        "failure_threshold": 3,
        "recovery_timeout": 30.0,
        "monitoring_window": 60.0
    }
}
```

## Health Checks & Monitoring

### Health Check Endpoints

```bash
# Basic health check
curl http://localhost:8080/health

# Detailed server statistics
curl http://localhost:8080/server_stats

# Prometheus metrics
curl http://localhost:8080/metrics
```

### Key Metrics to Monitor

1. **Request Metrics**
   - `mcp_requests_total` - Total requests by tool and status
   - `mcp_request_duration_seconds` - Request latency
   - `mcp_concurrent_requests` - Active concurrent requests

2. **Music Analysis Metrics**
   - `music21_analysis_duration_seconds` - Analysis performance
   - `music21_files_processed_total` - Processing throughput
   - `music21_analysis_queue_size` - Queue depth

3. **Security Metrics**
   - `music21_auth_attempts_total` - Authentication attempts
   - `music21_security_violations_total` - Security violations
   - `music21_blocked_requests_total` - Blocked requests

4. **System Metrics**
   - `music21_memory_usage_bytes` - Memory consumption
   - `music21_cpu_usage_percent` - CPU utilization
   - `music21_open_file_descriptors` - File descriptor usage

### Alerting Rules

```yaml
# alert_rules.yml
groups:
  - name: music21_mcp_alerts
    rules:
      - alert: HighErrorRate
        expr: |
          (
            sum(rate(mcp_requests_total{status="error"}[5m])) by (instance)
            /
            sum(rate(mcp_requests_total[5m])) by (instance)
          ) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} for instance {{ $labels.instance }}"

      - alert: HighMemoryUsage
        expr: music21_memory_usage_bytes{type="rss"} / 1024 / 1024 / 1024 > 6
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanize }}GB"

      - alert: CircuitBreakerOpen
        expr: increase(music21_circuit_breaker_state_changes[1h]) > 0
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Circuit breaker opened"
          description: "Circuit breaker {{ $labels.circuit_name }} has opened"
```

## Backup & Recovery

### Database Backup

```bash
#!/bin/bash
# backup_database.sh
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/var/backups/music21_mcp"
DB_NAME="music21_mcp"

# Create backup
pg_dump -h localhost -U music21_user $DB_NAME | gzip > "$BACKUP_DIR/db_backup_$TIMESTAMP.sql.gz"

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "db_backup_*.sql.gz" -mtime +30 -delete
```

### Configuration Backup

```bash
#!/bin/bash
# backup_config.sh
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/var/backups/music21_mcp"
CONFIG_DIR="/opt/music21-mcp"

# Create configuration backup
tar -czf "$BACKUP_DIR/config_backup_$TIMESTAMP.tar.gz" \
  -C "$CONFIG_DIR" \
  configs/ \
  docker-compose.enterprise.yml \
  monitoring/
```

### Recovery Procedures

1. **Database Recovery**
```bash
# Stop the service
sudo systemctl stop music21-mcp

# Restore database
gunzip -c /var/backups/music21_mcp/db_backup_TIMESTAMP.sql.gz | \
  psql -h localhost -U music21_user music21_mcp

# Start the service
sudo systemctl start music21-mcp
```

2. **Configuration Recovery**
```bash
# Extract configuration backup
tar -xzf /var/backups/music21_mcp/config_backup_TIMESTAMP.tar.gz \
  -C /opt/music21-mcp/

# Restart services
docker-compose -f docker-compose.enterprise.yml restart
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Check for memory leaks in music analysis
   - Adjust `MAX_SCORES_IN_MEMORY` setting
   - Implement score caching strategy

2. **Authentication Failures**
   - Verify OAuth configuration
   - Check client credentials
   - Review audit logs for security violations

3. **Circuit Breaker Trips**
   - Check underlying service health
   - Review failure patterns
   - Adjust circuit breaker thresholds

4. **Performance Issues**
   - Monitor request queue depth
   - Check CPU and memory usage
   - Review slow query logs

### Log Analysis

```bash
# View application logs
sudo journalctl -u music21-mcp -f

# View security logs
tail -f /var/log/music21_mcp/audit/audit.log

# View error logs
grep ERROR /var/log/music21_mcp/server.log
```

### Performance Tuning

1. **Memory Optimization**
```python
# Adjust memory settings
MAX_MEMORY_GB=16.0
MAX_SCORES_IN_MEMORY=2000
ENABLE_AGGRESSIVE_GC=true
```

2. **Concurrency Tuning**
```python
# Adjust concurrency limits
MAX_CONCURRENT_REQUESTS=100
BH_MAX_CONCURRENT_ANALYSIS=20
THREAD_POOL_SIZE=50
```

## Compliance

### SOC2 Compliance

The enterprise edition includes SOC2 Type II compliance features:

- **Security**: OAuth 2.1, input validation, security monitoring
- **Availability**: Circuit breakers, health checks, redundancy
- **Processing Integrity**: Input validation, audit trails
- **Confidentiality**: Encryption at rest and in transit
- **Privacy**: Data anonymization, retention policies

### GDPR Compliance

GDPR compliance features include:

- **Data Protection**: Encryption, access controls
- **Audit Trails**: Comprehensive logging of data operations
- **Data Retention**: Automated data purging
- **Data Subject Rights**: Export and deletion capabilities
- **Privacy by Design**: Default privacy-preserving settings

### Compliance Reporting

```bash
# Generate compliance report
curl -X POST http://localhost:8080/compliance/report \
  -H "Content-Type: application/json" \
  -d '{
    "report_type": "soc2",
    "start_date": "2024-01-01",
    "end_date": "2024-12-31"
  }'
```

## Production Checklist

Before going to production:

- [ ] Security configuration reviewed and hardened
- [ ] SSL/TLS certificates installed and configured
- [ ] OAuth 2.1 provider configured with production credentials
- [ ] Monitoring and alerting configured
- [ ] Backup procedures tested
- [ ] Disaster recovery plan documented
- [ ] Performance testing completed
- [ ] Security penetration testing completed
- [ ] Compliance requirements verified
- [ ] Documentation updated
- [ ] Team trained on operations procedures

## Support

For enterprise support:

- Technical Documentation: `/docs/`
- API Reference: `/docs/api/`
- Security Guide: `/docs/SECURITY.md`
- Performance Tuning: `/docs/PERFORMANCE.md`
- Troubleshooting: `/docs/TROUBLESHOOTING.md`

## License

This enterprise edition includes additional enterprise features and requires appropriate licensing for production use.