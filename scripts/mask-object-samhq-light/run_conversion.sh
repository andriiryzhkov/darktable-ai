#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
source "$SCRIPT_DIR/../common.sh"

activate_venv

python3 "$SCRIPT_DIR/convert_to_onnx.py" \
    --model-type vit_tiny \
    --checkpoint "$ROOT_DIR/checkpoints/mask-object-samhq-light/sam_hq_vit_tiny.pth" \
    --output-dir "$ROOT_DIR/models/mask-object-samhq-light" \
    --hq-token-only
