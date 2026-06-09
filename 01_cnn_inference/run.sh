#!/bin/bash
set -e

echo "=== Job 01: CNN Inference ==="
mkdir -p outputs

IMAGE_PATH="${PARAM_IMAGE_PATH:-/workspace/input_files/sample_image.png}"
MODELS_DIR="${PARAM_MODELS_DIR:-/workspace/models}"

echo "Input image : $IMAGE_PATH"
echo "Models dir  : $MODELS_DIR"

python cnn_inference.py \
    --image_path  "$IMAGE_PATH" \
    --models_dir  "$MODELS_DIR"

echo "=== Job 01 complete ==="
