#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
source "$SCRIPT_DIR/model.conf"
source "$SCRIPT_DIR/../common.sh"

MODEL_DIR="$ROOT_DIR/models/$MODEL_ID"

# Per-image point prompts (normalized x,y coordinates).
demo_args() {
    case "$1" in
        example_01) echo "--point 0.55,0.50" ;;  # boat hull
        example_02) echo "--point 0.55,0.55" ;;  # red tram
        example_03) echo "--point 0.28,0.65" ;;  # cat
    esac
}

echo "SAM 2.1 Small segmentation"
run_demo \
    --encoder "$MODEL_DIR/encoder.onnx" \
    --decoder "$MODEL_DIR/decoder.onnx"
