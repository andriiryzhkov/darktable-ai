#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
source "$SCRIPT_DIR/model.conf"
source "$SCRIPT_DIR/../common.sh"

activate_venv

CHECKPOINT_DIR="$ROOT_DIR/temp/mask-object-sam21-small-onnx"
MODEL_DIR="$ROOT_DIR/models/mask-object-sam21-small-onnx"

python3 "$SCRIPT_DIR/merge_onnx.py" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "$MODEL_DIR"

echo "Encoder: $MODEL_DIR/encoder.onnx"
echo "Decoder: $MODEL_DIR/decoder.onnx"
