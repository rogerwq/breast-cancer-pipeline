#!/bin/bash
set -e

echo "=== Job 00: Setup & Verify Models ==="
mkdir -p outputs

MODELS_DIR="${PARAM_MODELS_DIR:-/workspace/models}"
echo "Models directory: $MODELS_DIR"

python verify_models.py --models_dir "$MODELS_DIR"

echo "=== Job 00 complete ==="
