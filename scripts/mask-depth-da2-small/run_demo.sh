#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
source "$SCRIPT_DIR/model.conf"
source "$SCRIPT_DIR/../common.sh"

MODEL_DIR="$ROOT_DIR/models/$MODEL_ID"

echo "Depth Anything V2 Small depth estimation"
run_demo --model "$MODEL_DIR/model.onnx"
