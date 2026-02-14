#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
source "$SCRIPT_DIR/model.conf"
source "$SCRIPT_DIR/../common.sh"

setup_venv
download_onnx_models

echo "Success! Environment ready at $VENV_DIR"
