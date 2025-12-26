# ==============================================================================
# DOCKERFILE - Multi-Database Backend System
# ==============================================================================
# Production-ready container with security best practices
# Multi-stage build for optimized image size
# ==============================================================================

# ------------------------------------------------------------------------------
# STAGE 1: Builder - Install dependencies and build wheels
# ------------------------------------------------------------------------------
FROM python:3.11-slim as builder

# Set build-time environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and build wheels
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt

# ------------------------------------------------------------------------------
# STAGE 2: Production - Minimal runtime image
# ------------------------------------------------------------------------------
FROM python:3.11-slim as production

# Security: Run as non-root user
ARG UID=1000
ARG GID=1000

# Runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_HOME=/app

WORKDIR $APP_HOME

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy wheels from builder and install
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

# Create non-root user for security
RUN groupadd --gid $GID appgroup \
    && useradd --uid $UID --gid $GID --shell /bin/bash --create-home appuser \
    && chown -R appuser:appgroup $APP_HOME

# Copy application code
COPY --chown=appuser:appgroup . .

# Switch to non-root user
USER appuser

# Expose application port
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - Production server with optimal settings
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
