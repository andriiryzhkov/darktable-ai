#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
source "$SCRIPT_DIR/../common.sh"

activate_venv

python3 "$SCRIPT_DIR/export_onnx_model.py" \
    --checkpoint "$ROOT_DIR/checkpoints/denoise-nind/generator_280.pt" \
    --output "$ROOT_DIR/models/denoise-nind/model.onnx" \
    --opset 17 \
    --fp16
