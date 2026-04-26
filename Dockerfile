# =========================
# Builder Stage
# =========================
ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Build args
ARG BUILD_MODE=in-repo
ARG ENV_NAME=legacyforge

# Copy project
COPY . /app/env
WORKDIR /app/env

# Ensure uv is installed
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh; \
        ln -s /root/.local/bin/uv /usr/local/bin/uv; \
    fi

# Install dependencies (single pass)
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi


# =========================
# Runtime Stage
# =========================
FROM ${BASE_IMAGE}

WORKDIR /app

# Install runtime deps (needed for healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /app/env/.venv /app/.venv

# Copy app code
COPY --from=builder /app/env /app/env

# Environment setup
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]