# AI Financial Assistant — Fly.io deployment image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app ./app

# Create the data directory (SQLite + uploaded docs live here via Fly volume)
RUN mkdir -p /data

# The app listens on internal port 8000 (mapped to 80 by fly.io)
EXPOSE 8000

# Healthcheck — Render's /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI app (webhook mode is configured via env vars)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]