# Multi-stage build for optimized production image
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install UV for faster dependency installation
RUN pip install --no-cache-dir uv

# Copy dependency files
WORKDIR /app
COPY pyproject.toml uv.lock ./

# Install dependencies using UV
RUN uv pip install --no-cache-dir -r <(uv pip compile pyproject.toml)

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    # For music21 functionality
    lilypond \
    musescore3 \
    # For health checks
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash music21user

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=music21user:music21user . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/temp && \
    chown -R music21user:music21user /app

# Environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MUSIC21_MCP_HOST=0.0.0.0 \
    MUSIC21_MCP_PORT=8000 \
    MUSIC21_MAX_MEMORY_MB=512 \
    MUSIC21_GC_THRESHOLD_MB=100 \
    MUSIC21_MCP_TIMEOUT=30 \
    MUSIC21_LOG_LEVEL=INFO \
    MUSIC21_CORPUS_PATH=/app/data/corpus

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app/src'); from music21_mcp.health_check import check_health; exit(0 if check_health() else 1)"

# Switch to non-root user
USER music21user

# Expose ports
EXPOSE 8000

# Default command - can be overridden
CMD ["python", "-m", "music21_mcp.launcher", "--mode", "http"]