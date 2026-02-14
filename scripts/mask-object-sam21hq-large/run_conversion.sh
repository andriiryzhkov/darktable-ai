#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
source "$SCRIPT_DIR/../common.sh"

activate_venv

python3 "$SCRIPT_DIR/export_onnx_model.py" \
    --model-cfg "configs/sam2.1/sam2.1_hq_hiera_l.yaml" \
    --checkpoint "$ROOT_DIR/temp/mask-object-sam21hq-large/sam2.1_hq_hiera_large.pt" \
    --output-dir "$ROOT_DIR/models/mask-object-sam21hq-large"
