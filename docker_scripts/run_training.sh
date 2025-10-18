#!/bin/bash

# Run training in Docker
echo "Building Docker image..."
docker build -t plant-disease-detector .

CONFIG_FILE=${1:-configs/train_vit_gvj.yaml}

echo "Starting training with config: $CONFIG_FILE"
docker run -it --rm \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  -v "$(pwd)/runs:/app/runs" \
  -v "$(pwd)/configs:/app/configs" \
  -v "$(pwd)/data:/app/data" \
  plant-disease-detector \
  python -m plant_disease.training.train --config $CONFIG_FILE

echo "Training completed. Check checkpoints/ and runs/ directories for outputs."