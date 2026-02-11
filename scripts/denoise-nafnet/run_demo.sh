#!/bin/bash

# Script to run ONNX Demo using the Venv

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/../../models/denoise-nafnet"
IMAGE_DIR="$SCRIPT_DIR/NAFNet/demo"
VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found. Run ./scripts/setup_env.sh first."
    exit 1
fi

source "$VENV_DIR/bin/activate"

echo "NAFNet-SIDD-width32"
python3 "$SCRIPT_DIR/demo.py" --model "$MODEL_DIR/model.onnx" --input "$IMAGE_DIR/denoise_img.png" --output "$IMAGE_DIR/denoise_img_out_w32.png"