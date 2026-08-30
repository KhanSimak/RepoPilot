FROM python:3.11-slim

# git is needed at runtime by gitpython (cloning repos, incremental diff)
RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so Docker caches this layer — rebuilding after a
# code-only change won't re-download torch/transformers/onnxruntime again.
COPY requirements.txt .

# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.1

# Install the rest
RUN pip install --no-cache-dir \
    -r requirements.txt

COPY . .
# Hits our own /health endpoint. Uses python's stdlib (urllib) instead of
# adding a curl-based check, so no extra image bloat beyond what's above.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

# No --reload here on purpose: file-watching reload is a dev convenience
# and wastes CPU in a deployed container. For local dev with hot-reload,
# run uvicorn directly on your host (see the VS Code setup) instead of
# through this image, or override the command in docker-compose.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
