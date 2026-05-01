# AlloyTower AVM Backend — Render deployment
# Python 3.11 slim base — small image, fast cold starts, well-supported wheels.

FROM python:3.11-slim

# Set working directory at the repo root so relative paths in code work.
WORKDIR /app

# System deps required by lightgbm (libgomp) and matplotlib/shap.
# Keeping the install minimal to keep the image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (separate layer = better build cache).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project.
# .dockerignore controls what comes in (so node_modules, venv, etc. are excluded).
COPY . .

# Render injects the PORT env var. Default to 8000 for local docker runs.
ENV PORT=8000

# Bind to 0.0.0.0 so external requests can reach the service.
# Use exec form so SIGTERM reaches uvicorn cleanly during deploys.
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT