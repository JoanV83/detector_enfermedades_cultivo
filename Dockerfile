# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml uv.lock ./

# Set up virtual environment with uv
RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy the entire project
COPY . .

# Install the project with all dependencies in editable mode
RUN uv pip install -e .

# Create necessary directories
RUN mkdir -p /app/checkpoints /app/runs /app/artifacts /app/data

# Create non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose port for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["python", "-m", "streamlit", "run", "src/plant_disease/apps/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
