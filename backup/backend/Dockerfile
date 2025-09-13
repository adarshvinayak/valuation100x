# Optimized Railway Dockerfile - Slim build without heavy ML dependencies
FROM python:3.11-slim as builder

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Set environment variables for build optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set work directory
WORKDIR /app

# Copy optimized requirements
COPY requirements.railway.txt .

# Install Python dependencies with optimization
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.railway.txt

# Production stage
FROM python:3.11-slim

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

# Create app user for security
RUN useradd --create-home --shell /bin/bash app

# Set work directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p data/outputs logs models/embeddings && \
    chown -R app:app /app

# Copy application code
COPY --chown=app:app . .

# Switch to non-root user
USER app

# Expose port (Railway sets PORT dynamically)
EXPOSE 8000

# Health check (Railway manages healthcheck externally, removing internal healthcheck)
# HEALTHCHECK removed - Railway handles this via external calls

# Run the application
CMD ["python", "api_server.py"]
