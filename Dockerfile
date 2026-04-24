# ── Emotion Intelligence Platform — Dockerfile ───────────────────
# Build:  docker build -t emotion-platform .
# Run:    docker run -p 8000:8000 -v $(pwd)/outputs:/app/outputs emotion-platform
# Dev:    docker-compose up

FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements_platform.txt .
RUN pip install --no-cache-dir -r requirements_platform.txt

# Copy source
COPY src/       ./src/
COPY configs/   ./configs/

# Create runtime directories
RUN mkdir -p outputs/platform outputs/models outputs/results outputs/logs

# Non-root user for security
RUN useradd -m -u 1000 platform && chown -R platform:platform /app
USER platform

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Default: start API without models (health endpoint works, /predict returns 503)
# Override MODEL_PATHS env var to load trained models
CMD ["uvicorn", "src.api.routes:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]