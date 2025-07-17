# Docker Deployment Guide for Music21 MCP Server

## Overview

This guide covers deploying the Music21 MCP Server using Docker containers. The MCP server uses stdio transport for communication with Claude and other MCP clients.

## Quick Start

### Prerequisites

- Docker 20.10+ 
- Docker Compose 2.0+
- At least 2GB RAM available
- 5GB disk space for images and data

### Development Deployment

```bash
# Clone and navigate to project
git clone <repository-url>
cd music21-mcp-server

# Build and start services
docker-compose up --build

# Check health
docker-compose exec music21-mcp python /app/src/music21_mcp/health_check.py
```

### Production Deployment

```bash
# Use production configuration
docker-compose -f docker-compose.prod.yml up -d --build

# Monitor logs
docker-compose -f docker-compose.prod.yml logs -f music21-mcp
```

## Docker Configuration

### Dockerfile Features

- **Multi-stage build**: Optimized for production with minimal attack surface
- **Non-root user**: Runs as `music21` user for security
- **Lightweight health check**: Fast startup and monitoring
- **Poetry dependency management**: Isolated virtual environment
- **Tini init system**: Proper signal handling

### Transport Configuration

**Important**: MCP servers use stdio transport, not HTTP. Key differences:

- ❌ No port exposure (8000, 80, 443)
- ❌ No HTTP health checks with curl
- ✅ Uses stdin/stdout for communication
- ✅ Health check via Python script
- ✅ TTY and stdin_open enabled

## Service Architecture

### Core Services

1. **music21-mcp**: Main MCP server container
2. **redis**: Optional caching and session storage
3. **mcp-monitor**: Resource monitoring and alerting

### Health Checks

```python
# Lightweight health check (/app/src/music21_mcp/health_check.py)
- Python version validation
- MCP package availability  
- File system permissions
- Basic imports verification
```

## Environment Variables

### Development

```bash
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
LOG_LEVEL=INFO
REDIS_URL=redis://redis:6379/0
```

### Production

```bash
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
LOG_LEVEL=WARNING
REDIS_URL=redis://redis:6379/0
MCP_PRODUCTION=true
MEMORY_LIMIT_MB=512
MAX_SCORES=100
```

## Usage Examples

### Building Images

```bash
# Build development image
docker build -t music21-mcp:dev .

# Build production image
docker build -t music21-mcp:prod --target production .

# Multi-platform build
docker buildx build --platform linux/amd64,linux/arm64 -t music21-mcp:latest .
```

### Running MCP Server

```bash
# Interactive mode (for testing)
docker run -it --rm music21-mcp:dev python -m music21_mcp.server

# With Claude Desktop (stdio transport)
docker run -i music21-mcp:dev python -m music21_mcp.server

# With volumes for persistence
docker run -i -v ./data:/app/data -v ./logs:/app/logs music21-mcp:dev
```

### Testing Health

```bash
# Test health check
docker run --rm music21-mcp:dev python /app/src/music21_mcp/health_check.py

# Expected output: "healthy"
```

## Troubleshooting

### Common Issues

#### 1. Health Check Failing

```bash
# Check container logs
docker-compose logs music21-mcp

# Manual health check
docker-compose exec music21-mcp python /app/src/music21_mcp/health_check.py

# Check file permissions
docker-compose exec music21-mcp ls -la /app/data /app/logs
```

#### 2. Memory Issues

```bash
# Check memory usage
docker stats

# Monitor with dedicated service
docker-compose logs mcp-monitor

# Adjust memory limits in docker-compose.prod.yml
```

#### 3. MCP Import Errors

```bash
# Verify MCP package installation
docker-compose exec music21-mcp python -c "import mcp; print(mcp.__version__)"

# Check virtual environment
docker-compose exec music21-mcp which python
docker-compose exec music21-mcp pip list | grep mcp
```

#### 4. Permission Denied

```bash
# Check user context
docker-compose exec music21-mcp whoami
docker-compose exec music21-mcp id

# Fix volume permissions
sudo chown -R 1000:1000 ./data ./logs
```

### Performance Optimization

#### Memory Management

```bash
# Enable memory cleanup tool
docker-compose exec music21-mcp python -c "
import asyncio
from music21_mcp.server import cleanup_memory
print(asyncio.run(cleanup_memory()))
"
```

#### Container Resources

```yaml
# docker-compose.prod.yml
deploy:
  resources:
    limits:
      memory: 1G      # Adjust based on usage
      cpus: '0.5'     # Limit CPU usage
    reservations:
      memory: 256M    # Minimum guaranteed
      cpus: '0.1'
```

## Monitoring and Logging

### Log Management

```bash
# View logs
docker-compose logs -f music21-mcp

# Log rotation (automatic in production)
max-size: "10m"
max-file: "3"

# Export logs
docker-compose logs music21-mcp > music21-mcp.log
```

### Resource Monitoring

```bash
# Real-time stats
docker stats $(docker-compose ps -q)

# Custom monitoring
docker-compose logs mcp-monitor
```

## Security Considerations

### Container Security

- Runs as non-root user (`music21:music21`)
- Read-only source code mounting in production
- Network isolation with custom bridge
- No unnecessary port exposure
- Memory and CPU limits enforced

### Data Protection

```bash
# Backup volumes
docker run --rm -v music21_data:/data -v $(pwd):/backup alpine tar czf /backup/data.tar.gz /data

# Restore volumes  
docker run --rm -v music21_data:/data -v $(pwd):/backup alpine tar xzf /backup/data.tar.gz -C /
```

## Integration with Claude Desktop

### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "music21": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "./music21-data:/app/data",
        "music21-mcp:latest",
        "python", "-m", "music21_mcp.server"
      ],
      "env": {
        "LOG_LEVEL": "WARNING"
      }
    }
  }
}
```

### Testing MCP Integration

```bash
# Test MCP server manually
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {"roots": {"listChanged": false}}, "clientInfo": {"name": "test", "version": "1.0.0"}}}' | \
docker run -i --rm music21-mcp:latest python -m music21_mcp.server
```

## Deployment Checklist

### Pre-deployment

- [ ] Docker and Docker Compose installed
- [ ] Sufficient system resources (2GB RAM, 5GB disk)
- [ ] Network connectivity for image pulls
- [ ] Volume permissions configured

### Development Deployment

- [ ] `docker-compose up --build` succeeds
- [ ] Health checks pass
- [ ] Logs show no errors
- [ ] MCP tools respond correctly

### Production Deployment

- [ ] Use `docker-compose.prod.yml`
- [ ] Configure resource limits
- [ ] Set up log rotation
- [ ] Monitor resource usage
- [ ] Backup strategy in place
- [ ] Security review completed

## Support

For deployment issues:

1. Check this troubleshooting guide
2. Review container logs
3. Verify system requirements
4. Test with minimal configuration
5. Open GitHub issue with deployment details

## Version Compatibility

- Docker: 20.10+
- Docker Compose: 2.0+
- Python: 3.11+ (in container)
- MCP Protocol: 2024-11-05
- Memory: 256MB minimum, 1GB recommended