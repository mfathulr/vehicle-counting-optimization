# Vehicle Counting & Optimization - Dockerfile (CPU)
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
COPY packages.txt /tmp/packages.txt
RUN set -eux; \
    apt-get update; \
    xargs -a /tmp/packages.txt apt-get install -y --no-install-recommends; \
    apt-get install -y --no-install-recommends curl ca-certificates; \
    rm -rf /var/lib/apt/lists/*

# Python dependencies (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# App source
COPY . .

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Environment defaults
ENV ALLOWED_ORIGINS=http://localhost:8000 \
    MODEL_GDRIVE_ID= \
    API_KEY=

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
