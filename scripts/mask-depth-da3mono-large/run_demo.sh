#!/bin/bash

# Script to run ONNX Demo using the Venv

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="$ROOT_DIR/models/mask-depth-da3mono-large"
IMAGE_DIR="$ROOT_DIR/images"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found. Run ./setup_env.sh first."
    exit 1
fi

source "$VENV_DIR/bin/activate"

echo "DA3Mono-Large depth estimation"
python3 "$SCRIPT_DIR/demo.py" \
    --model "$MODEL_DIR/model.onnx" \
    --image "$IMAGE_DIR/example_01.jpg" \
    --output "$SCRIPT_DIR/output/depth_01.png"
