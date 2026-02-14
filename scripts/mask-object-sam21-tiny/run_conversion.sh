#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
source "$SCRIPT_DIR/../common.sh"

activate_venv

python3 "$SCRIPT_DIR/../mask-object-sam21-small/export_onnx_model.py" \
    --model-cfg "configs/sam2.1/sam2.1_hiera_t.yaml" \
    --checkpoint "$ROOT_DIR/checkpoints/mask-object-sam21-tiny/sam2.1_hiera_tiny.pt" \
    --output-dir "$ROOT_DIR/models/mask-object-sam21-tiny"
