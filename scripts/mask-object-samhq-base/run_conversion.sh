#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
source "$SCRIPT_DIR/../common.sh"

activate_venv

python3 "$SCRIPT_DIR/export_onnx_model.py" \
    --model-type vit_b \
    --checkpoint "$ROOT_DIR/temp/mask-object-samhq-base/sam_hq_vit_b.pth" \
    --output-dir "$ROOT_DIR/models/mask-object-samhq-base" \
    --hq-token-only
