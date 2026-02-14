#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <model_id>"
    echo ""
    echo "Available models:"
    for dir in "$(dirname "$0")"/*/; do
        if [ -f "$dir/model.conf" ]; then
            echo "  $(basename "$dir")"
        fi
    done
    exit 1
fi

MODEL_DIR="$(cd "$(dirname "$0")/$1" 2>/dev/null && pwd)" || {
    echo "Error: Model '$1' not found in scripts/"
    exit 1
}

if [ ! -f "$MODEL_DIR/model.conf" ]; then
    echo "Error: No model.conf found in $MODEL_DIR"
    exit 1
fi

echo "=== Setup ==="
bash "$MODEL_DIR/setup_env.sh"

echo ""
echo "=== Convert ==="
bash "$MODEL_DIR/run_conversion.sh"

if [ -f "$MODEL_DIR/run_demo.sh" ]; then
    echo ""
    echo "=== Demo ==="
    bash "$MODEL_DIR/run_demo.sh"
fi

echo ""
echo "=== Clean ==="
bash "$MODEL_DIR/clean_env.sh"
