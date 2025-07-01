# Multi-stage Docker build for Music21 MCP Server
# Production-optimized with security hardening and performance optimization

# Build stage
FROM python:3.11-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_CACHE_DIR=/opt/poetry-cache
ENV POETRY_VENV_IN_PROJECT=1
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VERSION=1.7.1

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Configure poetry and install dependencies
RUN poetry config virtualenvs.create true \
    && poetry config virtualenvs.in-project true \
    && poetry install --no-dev --no-root \
    && rm -rf $POETRY_CACHE_DIR

# Production stage
FROM python:3.11-slim as production

# Create non-root user for security
RUN groupadd -r music21 && useradd -r -g music21 music21

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set work directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY --chown=music21:music21 src/ ./src/
COPY --chown=music21:music21 pyproject.toml ./

# Install the package
RUN pip install -e .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs \
    && chown -R music21:music21 /app

# Switch to non-root user
USER music21

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000
ENV HOST=0.0.0.0

# Expose port
EXPOSE 8000

# Use tini as init system
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command
CMD ["python", "-m", "music21_mcp.server_remote", "--host", "0.0.0.0", "--port", "8000"]

# Labels for metadata
LABEL maintainer="music21-mcp-server"
LABEL version="1.0.0"
LABEL description="Production-ready Music21 MCP Server with OAuth2 authentication"
LABEL org.opencontainers.image.source="https://github.com/Bright-L01/music21-mcp-server"
LABEL org.opencontainers.image.documentation="https://music21-mcp-server.readthedocs.io"
LABEL org.opencontainers.image.licenses="MIT"