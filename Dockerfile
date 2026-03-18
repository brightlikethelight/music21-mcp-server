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
RUN uv pip compile pyproject.toml > /tmp/requirements.txt && \
    uv pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    lilypond \
    curl \
    && (apt-get install -y musescore || true) \
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
    MUSIC21_MCP_TIMEOUT=30

# Health check - uses the HTTP health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER music21user

# Expose ports
EXPOSE 8000

# Default command - can be overridden
CMD ["python", "-m", "music21_mcp.launcher", "http"]