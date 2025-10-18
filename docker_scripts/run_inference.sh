#!/bin/bash

# Run inference in Docker
echo "Building Docker image..."
docker build -t plant-disease-detector .

if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_image> [model_dir] [topk]"
  echo "Example: $0 data/test_images/sample.jpg runs/vit-gvj/final 5"
  exit 1
fi

IMAGE_PATH=$1
MODEL_DIR=${2:-runs/vit-gvj/final}
TOPK=${3:-5}

echo "Running inference on: $IMAGE_PATH"
docker run -it --rm \
  -v "$(pwd)/runs:/app/runs" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/$IMAGE_PATH:/app/input_image.jpg" \
  plant-disease-detector \
  python -m plant_disease.inference.predict \
    --image /app/input_image.jpg \
    --model_dir /app/$MODEL_DIR \
    --topk $TOPK