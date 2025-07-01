# Production Deployment Guide

This guide covers deploying the Music21 Remote MCP Server in production environments with high availability, security, and scalability.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Cloud Platform Deployments](#cloud-platform-deployments)
5. [Security Configuration](#security-configuration)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Backup and Disaster Recovery](#backup-and-disaster-recovery)
8. [Performance Optimization](#performance-optimization)

## Prerequisites

### System Requirements

- **CPU**: Minimum 2 cores, recommended 4+ cores
- **Memory**: Minimum 2GB RAM, recommended 4GB+ RAM
- **Storage**: Minimum 10GB, recommended 50GB+ for data persistence
- **Network**: HTTPS/TLS support, load balancer capability

### Software Requirements

- Docker 20.10+ and Docker Compose 2.0+
- Kubernetes 1.24+ (for K8s deployments)
- Helm 3.8+ (for Helm deployments)
- Redis 6.0+ (for session storage)
- SSL/TLS certificates

## Docker Deployment

### Basic Docker Deployment

```bash
# Clone the repository
git clone https://github.com/Bright-L01/music21-mcp-server
cd music21-mcp-server

# Build the production image
docker build -t music21-mcp-server:latest .

# Run with basic configuration
docker run -d \
  --name music21-mcp \
  -p 8000:8000 \
  -e REDIS_URL=redis://redis:6379 \
  -e LOG_LEVEL=INFO \
  music21-mcp-server:latest
```

### Docker Compose Production Setup

```bash
# Start with Redis and Nginx
docker-compose up -d

# Check service health
docker-compose ps
docker-compose logs -f music21-mcp
```

### Production Docker Configuration

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'
services:
  music21-mcp:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    image: music21-mcp-server:1.0.0
    restart: unless-stopped
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      - REDIS_URL=redis://redis:6379/0
      - LOG_LEVEL=INFO
      - ENABLE_DEMO_USERS=false
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 1G
          cpus: '1.0'
        reservations:
          memory: 256M
          cpus: '0.2'
    volumes:
      - music21_data:/app/data
      - music21_logs:/app/logs
    networks:
      - music21_network

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: >
      redis-server
      --appendonly yes
      --maxmemory 512mb
      --maxmemory-policy allkeys-lru
      --requirepass ${REDIS_PASSWORD}
    environment:
      - REDIS_PASSWORD=${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    networks:
      - music21_network
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'

  nginx:
    image: nginx:alpine
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/nginx.prod.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - music21-mcp
    networks:
      - music21_network
    deploy:
      resources:
        limits:
          memory: 256M
          cpus: '0.2'

volumes:
  music21_data:
    driver: local
  music21_logs:
    driver: local
  redis_data:
    driver: local

networks:
  music21_network:
    driver: bridge
```

## Kubernetes Deployment

### Using Kubectl

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n music21-mcp
kubectl get services -n music21-mcp
kubectl logs -f deployment/music21-mcp-server -n music21-mcp
```

### Using Kustomize

```bash
# Deploy with Kustomize
kubectl apply -k k8s/

# Update image tag
cd k8s
kustomize edit set image music21-mcp-server=music21-mcp-server:v1.0.1
kubectl apply -k .
```

### Using Helm

```bash
# Add the repository (if published)
helm repo add music21-mcp https://charts.example.com/music21-mcp
helm repo update

# Install with custom values
helm install music21-mcp music21-mcp/music21-mcp \
  --namespace music21-mcp \
  --create-namespace \
  --values values.production.yaml

# Upgrade deployment
helm upgrade music21-mcp music21-mcp/music21-mcp \
  --namespace music21-mcp \
  --values values.production.yaml
```

Create `values.production.yaml`:

```yaml
replicaCount: 3

image:
  tag: "1.0.0"

resources:
  requests:
    memory: "512Mi"
    cpu: "300m"
  limits:
    memory: "2Gi"
    cpu: "2000m"

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70

ingress:
  enabled: true
  className: "nginx"
  hosts:
  - host: music21-mcp.yourdomain.com
    paths:
    - path: /
      pathType: Prefix
  tls:
  - secretName: music21-mcp-tls
    hosts:
    - music21-mcp.yourdomain.com

config:
  demo:
    enabled: false
  oauth2:
    requirePkce: true
    allowPublicClients: false

redis:
  enabled: true
  auth:
    enabled: true
    password: "your-secure-password"
  master:
    persistence:
      enabled: true
      size: 10Gi

monitoring:
  enabled: true
```

## Cloud Platform Deployments

### AWS Deployment

#### EKS with ALB

```yaml
# values.aws.yaml
ingress:
  enabled: true
  className: "alb"
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:region:account:certificate/cert-id
    alb.ingress.kubernetes.io/ssl-policy: ELBSecurityPolicy-TLS-1-2-2017-01
    alb.ingress.kubernetes.io/backend-protocol: HTTP

persistence:
  storageClass: "gp3"

redis:
  master:
    persistence:
      storageClass: "gp3"
```

#### ECS Deployment

```json
{
  "family": "music21-mcp-server",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "music21-mcp",
      "image": "music21-mcp-server:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "REDIS_URL",
          "value": "redis://redis-cluster.cache.amazonaws.com:6379"
        }
      ],
      "secrets": [
        {
          "name": "JWT_SECRET",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:music21-mcp/jwt-secret"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/music21-mcp",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 10,
        "retries": 3
      }
    }
  ]
}
```

### Google Cloud Platform (GCP)

#### GKE Deployment

```yaml
# values.gcp.yaml
ingress:
  enabled: true
  className: "gce"
  annotations:
    kubernetes.io/ingress.global-static-ip-name: "music21-mcp-ip"
    networking.gke.io/managed-certificates: "music21-mcp-ssl-cert"
    kubernetes.io/ingress.allow-http: "false"

persistence:
  storageClass: "ssd"

redis:
  master:
    persistence:
      storageClass: "ssd"
```

#### Cloud Run Deployment

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: music21-mcp-server
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/execution-environment: gen2
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "100"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 80
      containers:
      - image: gcr.io/project-id/music21-mcp-server:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-memorystore-ip:6379"
        resources:
          limits:
            cpu: "2"
            memory: "2Gi"
        startupProbe:
          httpGet:
            path: /health
          initialDelaySeconds: 10
          periodSeconds: 5
          failureThreshold: 3
```

### Azure Deployment

#### AKS Deployment

```yaml
# values.azure.yaml
ingress:
  enabled: true
  className: "azure/application-gateway"
  annotations:
    appgw.ingress.kubernetes.io/ssl-redirect: "true"
    appgw.ingress.kubernetes.io/backend-protocol: "http"

persistence:
  storageClass: "managed-premium"

redis:
  master:
    persistence:
      storageClass: "managed-premium"
```

#### Container Instances

```json
{
  "apiVersion": "2021-07-01",
  "type": "Microsoft.ContainerInstance/containerGroups",
  "name": "music21-mcp-server",
  "location": "West US 2",
  "properties": {
    "containers": [
      {
        "name": "music21-mcp",
        "properties": {
          "image": "music21-mcp-server:latest",
          "ports": [
            {
              "port": 8000,
              "protocol": "TCP"
            }
          ],
          "environmentVariables": [
            {
              "name": "REDIS_URL",
              "value": "redis://redis-cache.redis.cache.windows.net:6380"
            }
          ],
          "resources": {
            "requests": {
              "cpu": 1,
              "memoryInGB": 2
            }
          }
        }
      }
    ],
    "osType": "Linux",
    "ipAddress": {
      "type": "Public",
      "ports": [
        {
          "port": 8000,
          "protocol": "TCP"
        }
      ]
    },
    "restartPolicy": "Always"
  }
}
```

## Security Configuration

### TLS/SSL Setup

```bash
# Generate self-signed certificates (development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/key.pem \
  -out ssl/cert.pem \
  -subj "/CN=music21-mcp.local"

# For production, use Let's Encrypt with cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.2/cert-manager.yaml

# Create ClusterIssuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@yourdomain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### OAuth2 Security Hardening

```yaml
# Production OAuth2 configuration
config:
  oauth2:
    requirePkce: true
    allowPublicClients: false
    accessTokenExpireMinutes: 15  # Shorter expiry
    refreshTokenExpireDays: 7     # Shorter refresh window
    supportedScopes:
    - read
    - write
    # Remove admin scope for most deployments
  
  session:
    ttlMinutes: 15               # Shorter sessions
    enableSlidingExpiration: false  # Disable for strict timeout
  
  cors:
    origins:
    - "https://yourdomain.com"   # Only allowed origins
```

### Network Security

```yaml
# Network policy example
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: music21-mcp-netpol
  namespace: music21-mcp
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: music21-mcp-server
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app.kubernetes.io/name: redis
    ports:
    - protocol: TCP
      port: 6379
  - to: []  # DNS and external APIs
    ports:
    - protocol: UDP
      port: 53
```

## Monitoring and Observability

### Prometheus Monitoring

```yaml
# ServiceMonitor for Prometheus
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: music21-mcp-monitor
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: music21-mcp-server
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Music21 MCP Server",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"music21-mcp-server\"}[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"music21-mcp-server\"}[5m]))"
          }
        ]
      }
    ]
  }
}
```

### Logging Configuration

```yaml
# Fluentd DaemonSet for log collection
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
spec:
  selector:
    matchLabels:
      name: fluentd
  template:
    spec:
      containers:
      - name: fluentd
        image: fluent/fluentd-kubernetes-daemonset:v1-debian-elasticsearch
        env:
        - name: FLUENT_ELASTICSEARCH_HOST
          value: "elasticsearch.logging.svc.cluster.local"
        volumeMounts:
        - name: varlog
          mountPath: /var/log
        - name: varlibdockercontainers
          mountPath: /var/lib/docker/containers
          readOnly: true
```

## Backup and Disaster Recovery

### Data Backup Strategy

```bash
# Redis backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

# Create Redis backup
kubectl exec -n music21-mcp redis-master-0 -- redis-cli BGSAVE
kubectl cp music21-mcp/redis-master-0:/data/dump.rdb $BACKUP_DIR/redis_$DATE.rdb

# Upload to cloud storage
aws s3 cp $BACKUP_DIR/redis_$DATE.rdb s3://your-backup-bucket/redis/
```

### Kubernetes Backup

```yaml
# Velero backup schedule
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: music21-mcp-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  template:
    includedNamespaces:
    - music21-mcp
    storageLocation: default
    volumeSnapshotLocations:
    - default
    ttl: 720h0m0s  # 30 days
```

## Performance Optimization

### Resource Tuning

```yaml
# Production resource allocation
resources:
  requests:
    memory: "512Mi"
    cpu: "300m"
  limits:
    memory: "2Gi"
    cpu: "2000m"

# JVM tuning for better performance
env:
- name: PYTHONOPTIMIZE
  value: "2"
- name: PYTHONUNBUFFERED
  value: "1"
```

### Database Optimization

```yaml
# Redis optimization
redis:
  master:
    configuration: |
      maxmemory 1gb
      maxmemory-policy allkeys-lru
      tcp-keepalive 60
      timeout 300
      save 900 1
      save 300 10
      save 60 10000
```

### Load Testing

```bash
# Using k6 for load testing
k6 run - <<EOF
import http from 'k6/http';
import { check } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 100 },
    { duration: '5m', target: 100 },
    { duration: '2m', target: 200 },
    { duration: '5m', target: 200 },
    { duration: '2m', target: 0 },
  ],
};

export default function () {
  let response = http.get('https://music21-mcp.yourdomain.com/health');
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
}
EOF
```

## Troubleshooting

### Common Issues

1. **Pod CrashLoopBackOff**
   ```bash
   kubectl describe pod -n music21-mcp
   kubectl logs -f deployment/music21-mcp-server -n music21-mcp --previous
   ```

2. **Redis Connection Issues**
   ```bash
   kubectl exec -it deployment/music21-mcp-server -n music21-mcp -- python -c "
   import redis
   r = redis.Redis(host='redis-service', port=6379)
   print(r.ping())
   "
   ```

3. **Ingress Issues**
   ```bash
   kubectl describe ingress -n music21-mcp
   kubectl logs -f deployment/nginx-controller -n ingress-nginx
   ```

### Health Checks

```bash
# Application health
curl -f https://music21-mcp.yourdomain.com/health

# OAuth2 metadata
curl https://music21-mcp.yourdomain.com/auth/.well-known/oauth-authorization-server

# Resource usage
kubectl top pods -n music21-mcp
kubectl top nodes
```

This comprehensive guide covers all aspects of production deployment for the Music21 Remote MCP Server, ensuring high availability, security, and scalability in enterprise environments.