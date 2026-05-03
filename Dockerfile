FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Install system dependencies needed for XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install uv

# Copy dependency definition first for better Docker layer caching
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --locked --no-dev --no-install-project

ENV PATH="/app/.venv/bin:${PATH}"

# Copy project files
COPY . /app

# Switch to non-root user
USER appuser

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1

# Start the real-time scoring API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
