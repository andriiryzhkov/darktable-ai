#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
source "$SCRIPT_DIR/../common.sh"

activate_venv

python3 "$SCRIPT_DIR/export_onnx_model.py" \
    --model-id "$ROOT_DIR/temp/mask-depth-da3mono-large" \
    --output "$ROOT_DIR/models/mask-depth-da3mono-large/model.onnx" \
    --height 504 --width 504 \
    --opset 17
