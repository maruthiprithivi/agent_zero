# Agent Zero Deployment Guide

Complete guide for deploying Agent Zero ClickHouse MCP Server from development to enterprise production environments.

## Overview

Agent Zero supports multiple deployment patterns designed for different use cases and environments:

| Deployment Mode | Use Case | Transport | Scalability | Best For |
|----------------|----------|-----------|-------------|----------|
| **Local** | Development, IDE integration | stdio | Single user | Developers, testing |
| **Standalone** | Team deployments, testing | HTTP/WebSocket | Multi-user | Small teams, staging |
| **Enterprise** | Production, high availability | HTTP/WebSocket/gRPC | Highly scalable | Production, enterprise |

## Quick Start

### Installation

```bash
# Install with uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install ch-agent-zero

# Or install with pip
pip install ch-agent-zero
```

### Configuration

```bash
# Set ClickHouse connection
export CLICKHOUSE_HOST=your-clickhouse-host
export CLICKHOUSE_USER=your-username
export CLICKHOUSE_PASSWORD=your-password

# Test connection
ch-agent-zero --test-connection
```

## Local Development

### IDE Integration

**Claude Desktop** - Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "agent-zero": {
      "command": "ch-agent-zero",
      "env": {
        "CLICKHOUSE_HOST": "localhost",
        "CLICKHOUSE_USER": "default",
        "CLICKHOUSE_PASSWORD": "password"
      }
    }
  }
}
```

**Cursor IDE** - Auto-configure:
```bash
bash <(curl -sSL https://raw.githubusercontent.com/maruthiprithivi/agent_zero/main/scripts/install.sh) --ide cursor
```

**Windsurf IDE** - Auto-configure:
```bash
bash <(curl -sSL https://raw.githubusercontent.com/maruthiprithivi/agent_zero/main/scripts/install.sh) --ide windsurf
```

**VS Code** - Manual configuration:
```json
{
  "mcp.servers": {
    "agent-zero": {
      "command": "ch-agent-zero",
      "args": ["--deployment-mode", "local"],
      "env": {
        "CLICKHOUSE_HOST": "localhost",
        "CLICKHOUSE_USER": "default",
        "CLICKHOUSE_PASSWORD": "password"
      }
    }
  }
}
```

### Local Server Mode

```bash
# Run standalone server for local development
ch-agent-zero --deployment-mode standalone --host 127.0.0.1 --port 8505

# Access health check
curl http://localhost:8505/health
```

## Docker Deployment

### Single Container

```bash
# Quick start with Docker
docker run -d \
  --name agent-zero \
  -p 8505:8505 \
  -e CLICKHOUSE_HOST=your-host \
  -e CLICKHOUSE_USER=your-user \
  -e CLICKHOUSE_PASSWORD=your-password \
  -e DEPLOYMENT_MODE=standalone \
  ghcr.io/maruthiprithivi/agent-zero:latest
```

### Docker Compose

**Development Environment:**
```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  agent-zero:
    image: ghcr.io/maruthiprithivi/agent-zero:latest
    container_name: agent-zero-dev
    ports:
      - "8505:8505"
    environment:
      - DEPLOYMENT_MODE=standalone
      - LOG_LEVEL=DEBUG
      - CLICKHOUSE_HOST=clickhouse
      - CLICKHOUSE_USER=default
      - CLICKHOUSE_PASSWORD=
      - CLICKHOUSE_SECURE=false
    depends_on:
      - clickhouse
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8505/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  clickhouse:
    image: clickhouse/clickhouse-server:latest
    container_name: clickhouse-dev
    ports:
      - "8123:8123"
      - "9000:9000"
    environment:
      - CLICKHOUSE_DB=default
      - CLICKHOUSE_USER=default
      - CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1
    volumes:
      - clickhouse-data:/var/lib/clickhouse
    restart: unless-stopped

volumes:
  clickhouse-data:
```

**Production Environment:**
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  agent-zero:
    image: ghcr.io/maruthiprithivi/agent-zero:latest
    deploy:
      replicas: 3
      restart_policy:
        condition: on-failure
        delay: 10s
        max_attempts: 3
      update_config:
        parallelism: 1
        delay: 30s
        failure_action: rollback
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
    ports:
      - "8505:8505"
    environment:
      - DEPLOYMENT_MODE=enterprise
      - LOG_LEVEL=INFO
      - LOG_FORMAT=json
      - ENABLE_METRICS=true
      - ENABLE_HEALTH_CHECK=true
      - ENABLE_TRACING=true
      - SSL_ENABLE=true
      - CLICKHOUSE_HOST=${CLICKHOUSE_HOST}
      - CLICKHOUSE_USER=${CLICKHOUSE_USER}
      - CLICKHOUSE_PASSWORD_FILE=/run/secrets/clickhouse_password
      - CLICKHOUSE_SECURE=true
    secrets:
      - clickhouse_password
      - ssl_cert
      - ssl_key
    volumes:
      - prod-logs:/app/logs
      - prod-backup:/app/backup
    networks:
      - prod-network
    healthcheck:
      test: ["CMD", "curl", "-k", "-f", "https://localhost:8505/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Load balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - agent-zero
    networks:
      - prod-network

secrets:
  clickhouse_password:
    file: ./secrets/clickhouse_password.txt
  ssl_cert:
    file: ./secrets/ssl_cert.pem
  ssl_key:
    file: ./secrets/ssl_key.pem

volumes:
  prod-logs:
  prod-backup:

networks:
  prod-network:
    driver: overlay
    attachable: true
```

### Custom Docker Build

```dockerfile
# Dockerfile.production
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -s /bin/bash -u 1000 agent_zero

# Set working directory
WORKDIR /app

# Install Agent Zero
RUN pip install ch-agent-zero[production]

# Security hardening
RUN find /usr/local -type d -name __pycache__ -exec rm -rf {} + || true
RUN find /usr/local -type f -name "*.pyc" -delete || true

# Switch to app user
USER agent_zero

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8505/health || exit 1

EXPOSE 8505

# Production command
CMD ["ch-agent-zero", \
     "--deployment-mode", "enterprise", \
     "--host", "0.0.0.0", \
     "--enable-metrics", \
     "--enable-health-check"]
```

## Kubernetes Deployment

### Helm Chart Installation

```bash
# Add Helm repository (when available)
helm repo add agent-zero https://charts.agent-zero.example.com
helm repo update

# Install with custom values
helm install agent-zero agent-zero/agent-zero \
  --namespace agent-zero \
  --create-namespace \
  --set clickhouse.host=your-clickhouse-host \
  --set clickhouse.user=your-user \
  --set clickhouse.password=your-password \
  --set deployment.mode=enterprise \
  --set replicaCount=3
```

### Manual Kubernetes Deployment

**Namespace and ConfigMap:**
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: agent-zero
  labels:
    name: agent-zero

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-zero-config
  namespace: agent-zero
data:
  production.json: |
    {
      "service_name": "agent-zero-mcp",
      "environment": "production",
      "host": "0.0.0.0",
      "port": 8505,
      "enable_metrics": true,
      "enable_tracing": true,
      "enable_performance_monitoring": true,
      "enable_backups": true,
      "performance_config": {
        "connection_pool_size": 20,
        "query_timeout_seconds": 300,
        "cache_size_mb": 256
      }
    }
```

**Secrets:**
```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: clickhouse-secret
  namespace: agent-zero
type: Opaque
stringData:
  host: "your-clickhouse-host"
  username: "your-username"
  password: "your-password"
```

**Deployment:**
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-zero
  namespace: agent-zero
  labels:
    app: agent-zero
    version: v1.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: agent-zero
  template:
    metadata:
      labels:
        app: agent-zero
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8505"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: agent-zero
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
      containers:
      - name: agent-zero
        image: ghcr.io/maruthiprithivi/agent-zero:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8505
          name: http
          protocol: TCP
        env:
        - name: DEPLOYMENT_MODE
          value: "enterprise"
        - name: LOG_LEVEL
          value: "INFO"
        - name: LOG_FORMAT
          value: "json"
        - name: ENABLE_METRICS
          value: "true"
        - name: ENABLE_HEALTH_CHECK
          value: "true"
        - name: CLICKHOUSE_HOST
          valueFrom:
            secretKeyRef:
              name: clickhouse-secret
              key: host
        - name: CLICKHOUSE_USER
          valueFrom:
            secretKeyRef:
              name: clickhouse-secret
              key: username
        - name: CLICKHOUSE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: clickhouse-secret
              key: password
        - name: CLICKHOUSE_SECURE
          value: "true"
        livenessProbe:
          httpGet:
            path: /health/live
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: http
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 3
        startupProbe:
          httpGet:
            path: /health/live
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 10
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
          readOnly: true
        - name: logs-volume
          mountPath: /app/logs
        - name: backup-volume
          mountPath: /app/backup
      volumes:
      - name: config-volume
        configMap:
          name: agent-zero-config
      - name: logs-volume
        emptyDir: {}
      - name: backup-volume
        persistentVolumeClaim:
          claimName: agent-zero-backup-pvc
      nodeSelector:
        kubernetes.io/os: linux
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - agent-zero
              topologyKey: kubernetes.io/hostname
```

**Service and Ingress:**
```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: agent-zero-service
  namespace: agent-zero
  labels:
    app: agent-zero
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8505"
    prometheus.io/path: "/metrics"
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8505
    protocol: TCP
    name: http
  selector:
    app: agent-zero

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: agent-zero-ingress
  namespace: agent-zero
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/backend-protocol: "HTTP"
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "600"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.agent-zero.example.com
    secretName: agent-zero-tls
  rules:
  - host: api.agent-zero.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: agent-zero-service
            port:
              number: 80
```

### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-zero-hpa
  namespace: agent-zero
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-zero
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Pods
        value: 1
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 2
        periodSeconds: 60
```

## Cloud Provider Deployments

### AWS ECS/Fargate

```json
{
  "family": "agent-zero",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "agent-zero",
      "image": "ghcr.io/maruthiprithivi/agent-zero:latest",
      "portMappings": [
        {
          "containerPort": 8505,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "DEPLOYMENT_MODE", "value": "enterprise"},
        {"name": "LOG_LEVEL", "value": "INFO"},
        {"name": "LOG_FORMAT", "value": "json"}
      ],
      "secrets": [
        {
          "name": "CLICKHOUSE_HOST",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:agent-zero/clickhouse:host::"
        },
        {
          "name": "CLICKHOUSE_PASSWORD",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:agent-zero/clickhouse:password::"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/agent-zero",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8505/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

### Google Cloud Run

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: agent-zero
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 80
      containers:
      - image: ghcr.io/maruthiprithivi/agent-zero:latest
        ports:
        - containerPort: 8505
        env:
        - name: DEPLOYMENT_MODE
          value: "enterprise"
        - name: CLICKHOUSE_HOST
          value: "your-clickhouse-host"
        - name: CLICKHOUSE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: clickhouse-secrets
              key: password
        resources:
          limits:
            cpu: "1"
            memory: "512Mi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8505
          initialDelaySeconds: 30
          periodSeconds: 10
```

### Azure Container Instances

```yaml
apiVersion: 2019-12-01
location: East US
name: agent-zero-aci
properties:
  containers:
  - name: agent-zero
    properties:
      image: ghcr.io/maruthiprithivi/agent-zero:latest
      ports:
      - port: 8505
        protocol: TCP
      environmentVariables:
      - name: DEPLOYMENT_MODE
        value: enterprise
      - name: CLICKHOUSE_HOST
        value: your-clickhouse-host
      - name: CLICKHOUSE_PASSWORD
        secureValue: your-password
      resources:
        requests:
          cpu: 0.5
          memoryInGb: 1
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 8505
    dnsNameLabel: agent-zero-aci
tags:
  environment: production
  project: agent-zero
type: Microsoft.ContainerInstance/containerGroups
```

## Environment Configuration

### Required Variables

```bash
# ClickHouse Connection
CLICKHOUSE_HOST=your-clickhouse-host
CLICKHOUSE_USER=your-username
CLICKHOUSE_PASSWORD=your-password
```

### Optional Configuration

```bash
# ClickHouse Connection Details
CLICKHOUSE_PORT=8443                    # Default: 8443 for HTTPS, 8123 for HTTP
CLICKHOUSE_SECURE=true                  # Use SSL/TLS
CLICKHOUSE_VERIFY=true                  # Verify SSL certificates
CLICKHOUSE_CONNECT_TIMEOUT=30           # Connection timeout in seconds
CLICKHOUSE_SEND_RECEIVE_TIMEOUT=300     # Query timeout in seconds
CLICKHOUSE_DATABASE=default             # Default database

# Server Configuration
MCP_SERVER_HOST=127.0.0.1              # Server bind address
MCP_SERVER_PORT=8505                   # Server port
MCP_DEPLOYMENT_MODE=local              # local/standalone/enterprise
MCP_TRANSPORT=stdio                    # stdio/http/websocket

# Feature Flags
MCP_ENABLE_METRICS=false               # Prometheus metrics
MCP_ENABLE_HEALTH_CHECK=true           # Health check endpoints
MCP_RATE_LIMIT_ENABLED=false           # Rate limiting
MCP_TOOL_LIMIT=100                     # Max concurrent tools
MCP_RESOURCE_LIMIT=50                  # Max concurrent resources

# Security
MCP_SSL_ENABLE=false                   # Enable SSL/TLS
MCP_SSL_CERTFILE=/path/to/cert.pem     # SSL certificate file
MCP_SSL_KEYFILE=/path/to/key.pem       # SSL private key file
MCP_AUTH_USERNAME=                     # Basic auth username
MCP_AUTH_PASSWORD_FILE=                # Basic auth password file
MCP_OAUTH_ENABLE=false                 # OAuth2 authentication

# Performance
MCP_WORKER_THREADS=4                   # Worker thread count
MCP_MAX_CONCURRENT_REQUESTS=100        # Max concurrent requests
MCP_CACHE_SIZE=1000                    # Cache size limit
MCP_CACHE_TTL=300                      # Cache TTL in seconds

# Logging
LOG_LEVEL=INFO                         # DEBUG/INFO/WARNING/ERROR
LOG_FORMAT=json                        # json/text
LOG_FILE=/var/log/agent_zero.log       # Log file path

# Monitoring
ENABLE_TRACING=false                   # Distributed tracing
TRACING_ENDPOINT=http://jaeger:14268   # Tracing collector endpoint
METRICS_PORT=9090                      # Metrics server port
```

## Monitoring and Health Checks

### Health Check Endpoints

```bash
# Basic health check
curl http://localhost:8505/health

# Detailed health status
curl http://localhost:8505/health?detailed=true

# Readiness check (for load balancers)
curl http://localhost:8505/ready

# Liveness check (for orchestrators)
curl http://localhost:8505/health/live

# Metrics endpoint (if enabled)
curl http://localhost:8505/metrics
```

### Health Check Response

```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00Z",
  "uptime_seconds": 3600,
  "version": "1.0.0",
  "deployment_mode": "enterprise",
  "services": [
    {
      "name": "system",
      "status": "healthy",
      "message": "System resources within normal limits",
      "response_time_ms": 5.2,
      "details": {
        "cpu_percent": 45.2,
        "memory_percent": 68.1,
        "memory_available_mb": 2048
      }
    },
    {
      "name": "clickhouse",
      "status": "healthy",
      "message": "ClickHouse connectivity verified",
      "response_time_ms": 12.8,
      "details": {
        "server_version": "23.12.1.1",
        "server_uptime_seconds": 86400
      }
    }
  ],
  "metrics": {
    "checks_total": 2,
    "healthy_checks": 2,
    "degraded_checks": 0,
    "unhealthy_checks": 0
  }
}
```

### Prometheus Monitoring

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'agent-zero'
    static_configs:
      - targets: ['agent-zero:8505']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'agent-zero-health'
    static_configs:
      - targets: ['agent-zero:8505']
    metrics_path: /health
    scrape_interval: 60s
```

### Key Metrics

- `agent_zero_http_requests_total` - Total HTTP requests
- `agent_zero_http_request_duration_seconds` - Request duration histogram
- `agent_zero_active_connections` - Active connections by type
- `agent_zero_clickhouse_queries_total` - ClickHouse queries executed
- `agent_zero_clickhouse_query_duration_seconds` - Query execution time
- `agent_zero_errors_total` - Total errors by type
- `agent_zero_mcp_tool_calls_total` - MCP tool invocations
- `agent_zero_process_memory_bytes` - Process memory usage
- `agent_zero_process_cpu_usage_percent` - CPU usage percentage

## Security Considerations

### SSL/TLS Configuration

```bash
# Generate self-signed certificate for development
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Configure SSL
export MCP_SSL_ENABLE=true
export MCP_SSL_CERTFILE=/path/to/cert.pem
export MCP_SSL_KEYFILE=/path/to/key.pem

# Run with SSL
ch-agent-zero --deployment-mode enterprise --enable-ssl
```

### Authentication

```bash
# Basic authentication
export MCP_AUTH_USERNAME=admin
echo "secure-password" > /path/to/password.txt
export MCP_AUTH_PASSWORD_FILE=/path/to/password.txt

# OAuth2 (when supported)
export MCP_OAUTH_ENABLE=true
export MCP_OAUTH_CLIENT_ID=your-client-id
export MCP_OAUTH_CLIENT_SECRET=your-client-secret
```

### Network Security

- Use VPCs and security groups
- Implement firewall rules
- Enable rate limiting
- Monitor access patterns
- Use secrets management

## Troubleshooting

### Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Connection Failed** | Can't connect to ClickHouse | Verify CLICKHOUSE_HOST and credentials |
| **Port Binding** | Port already in use | Change MCP_SERVER_PORT or kill conflicting process |
| **SSL Errors** | Certificate validation failed | Check certificate paths and validity |
| **Memory Issues** | Container OOM killed | Increase memory limits |
| **Permission Denied** | File access errors | Check file permissions and user context |

### Debugging Commands

```bash
# Check configuration
ch-agent-zero --show-config

# Debug mode
ch-agent-zero --log-level DEBUG

# Validate environment
ch-agent-zero --validate-config

# Connection test
ch-agent-zero --test-connection

# Container logs
docker logs agent-zero

# Kubernetes logs
kubectl logs -f deployment/agent-zero -n agent-zero
```

### Performance Optimization

```bash
# Resource optimization
export MCP_WORKER_THREADS=4
export MCP_MAX_CONCURRENT_REQUESTS=100
export MCP_CACHE_SIZE=1000
export MCP_CACHE_TTL=300

# Connection optimization
export CLICKHOUSE_POOL_SIZE=10
export CLICKHOUSE_MAX_OVERFLOW=20
export CLICKHOUSE_SEND_RECEIVE_TIMEOUT=300
```

## Best Practices

### Production Deployment

1. **High Availability**
   - Deploy multiple replicas
   - Use health checks and readiness probes
   - Implement circuit breakers
   - Set up load balancing

2. **Security**
   - Use secrets management
   - Enable SSL/TLS for all communication
   - Implement proper authentication
   - Regular security updates

3. **Monitoring**
   - Set up comprehensive metrics collection
   - Implement alerting on key metrics
   - Use distributed tracing
   - Monitor logs for errors and patterns

4. **Performance**
   - Monitor resource utilization
   - Optimize database queries
   - Implement caching strategies
   - Use auto-scaling

5. **Reliability**
   - Regular backup and disaster recovery testing
   - Implement zero-downtime deployments
   - Monitor dependency health
   - Have incident response procedures

### Scaling Guidelines

- **Vertical Scaling**: Increase CPU/memory for single instance
- **Horizontal Scaling**: Add more replicas behind load balancer
- **Database Scaling**: Use ClickHouse clustering
- **Caching**: Implement distributed caching with Redis

## Backup and Recovery

### Automated Backups

```bash
# Create configuration backup
curl -X POST http://localhost:8505/backup/create \
  -H "Content-Type: application/json" \
  -d '{
    "type": "configuration",
    "description": "Automated daily backup",
    "storage_type": "s3",
    "tags": {"environment": "production", "automated": "true"}
  }'

# Check backup status
curl http://localhost:8505/backup/status

# List available backups
curl http://localhost:8505/backup/list
```

### Disaster Recovery

```bash
# Restore from backup
curl -X POST http://localhost:8505/backup/restore \
  -H "Content-Type: application/json" \
  -d '{
    "backup_id": "configuration_1642234567",
    "create_backup_before_restore": true,
    "verify_integrity": true
  }'

# Monitor restore progress
curl http://localhost:8505/backup/restore/status/configuration_1642234567
```

Ready to deploy? Choose the deployment option that best fits your environment and follow the step-by-step instructions above.

---

For user documentation, see [USER_GUIDE.md](USER_GUIDE.md).
For development and contributing, see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).
