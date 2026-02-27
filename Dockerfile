# FORGE Demo UI Dockerfile
# Based on Python 3.11-slim for minimal footprint

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /forge

# Install system dependencies if required (e.g., for certain rust builds or libs)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY demo_ui/backend/requirements.txt demo_ui/backend/requirements.txt
RUN pip install --no-cache-dir -r demo_ui/backend/requirements.txt

# Copy source code and assets
COPY demo_ui/ demo_ui/
COPY examples/ examples/
COPY python/ python/
COPY conftest.py .
COPY demo_results.md .

# Set PYTHONPATH so the backend can find demo_ui and other modules
ENV PYTHONPATH="/forge:${PYTHONPATH}"

# Expose the default port
EXPOSE 8765

# Start the uvicorn server
CMD ["python", "-m", "uvicorn", "demo_ui.backend.main:app", "--host", "0.0.0.0", "--port", "8765"]
