FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for XGBoost/Optuna
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for blazingly fast dependency management
RUN pip install uv

# Copy project files
COPY . /app

# Install Python dependencies globally within the container via uv
RUN uv pip install --system fastapi uvicorn xgboost shap pandas numpy pydantic google-genai fairlearn optuna scikit-learn matplotlib seaborn evidently

# Expose the API port
EXPOSE 8000

# Start the real-time scoring API
CMD ["uv", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
