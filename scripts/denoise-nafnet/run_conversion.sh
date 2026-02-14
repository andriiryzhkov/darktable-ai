#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
source "$SCRIPT_DIR/../common.sh"

activate_venv

python3 "$SCRIPT_DIR/convert_nafnet.py" \
    --config "$SCRIPT_DIR/NAFNet/options/test/SIDD/NAFNet-width32.yml" \
    --checkpoint "$ROOT_DIR/temp/denoise-nafnet/NAFNet-SIDD-width32.pth" \
    --output "$ROOT_DIR/models/denoise-nafnet/model.onnx" \
    --opset 17 \
    --fp16