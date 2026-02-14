#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
source "$SCRIPT_DIR/model.conf"
source "$SCRIPT_DIR/../common.sh"

echo "Pre-converted ONNX model — no conversion needed."
echo "ONNX model at: $ROOT_DIR/${ONNX_PATHS[0]}"
