#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
source "$SCRIPT_DIR/model.conf"
source "$SCRIPT_DIR/../common.sh"

MODEL_DIR="$ROOT_DIR/models/$MODEL_ID"

# Per-image mask rectangles (normalized x1,y1,x2,y2).
demo_args() {
    case "$1" in
        example_01) echo "--mask 0.30,0.15,0.75,0.85" ;;  # boat
        example_02) echo "--mask 0.40,0.35,0.75,0.80" ;;  # red tram
        example_03) echo "--mask 0.10,0.40,0.45,0.95" ;;  # cat
    esac
}

echo "LaMa Big inpainting"
run_demo --model "$MODEL_DIR/model.onnx"
