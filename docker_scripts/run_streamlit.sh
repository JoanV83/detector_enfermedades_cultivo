#!/bin/bash

# Build and run Streamlit app in Docker
echo "Building Docker image..."
docker build -t plant-disease-detector .

echo "Starting Streamlit app..."
docker run -it --rm \
  -p 8501:8501 \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  -v "$(pwd)/runs:/app/runs" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/configs:/app/configs" \
  plant-disease-detector

echo "App available at http://localhost:8501"