#!/bin/bash

# Script to run ONNX Demo using the Venv

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="$ROOT_DIR/models/mask-object-sam21-small-onnx"
IMAGE_DIR="$ROOT_DIR/images"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found. Run ./setup_env.sh first."
    exit 1
fi

source "$VENV_DIR/bin/activate"

echo "SAM 2.1 Small ONNX segmentation (cat point prompt)"
python3 "$SCRIPT_DIR/demo.py" \
    --encoder "$MODEL_DIR/encoder.onnx" \
    --decoder "$MODEL_DIR/decoder.onnx" \
    --image "$IMAGE_DIR/example_03.jpg" \
    --point 0.28,0.65 \
    --output "$SCRIPT_DIR/output/mask_03.png"
