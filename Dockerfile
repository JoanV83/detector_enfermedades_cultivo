# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/checkpoints /app/runs /app/artifacts /app/data

# Expose port for Streamlit
EXPOSE 8501

# Default command (can be overridden)
CMD ["python", "-m", "streamlit", "run", "src/plant_disease/apps/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]