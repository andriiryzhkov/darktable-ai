#!/bin/bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$ROOT_DIR/models"

if [ -z "$1" ]; then
    echo "Usage: $0 <model_id> [setup|convert|validate|demo|clean]"
    echo ""
    echo "Available models:"
    for dir in "$MODELS_DIR"/*/; do
        if [ -f "$dir/model.conf" ]; then
            echo "  $(basename "$dir")"
        fi
    done
    exit 1
fi

MODEL_ID="$1"
ACTION="${2:-all}"

SCRIPT_DIR="$(cd "$MODELS_DIR/$MODEL_ID" 2>/dev/null && pwd)" || {
    echo "Error: Model '$MODEL_ID' not found in models/"
    exit 1
}

if [ ! -f "$SCRIPT_DIR/model.conf" ]; then
    echo "Error: No model.conf found in $SCRIPT_DIR"
    exit 1
fi

VENV_DIR="$SCRIPT_DIR/.venv"
source "$SCRIPT_DIR/model.conf"
source "$MODELS_DIR/common.sh"

case "$ACTION" in
    setup)
        setup_env
        ;;
    convert)
        run_conversion
        ;;
    validate)
        run_validation
        ;;
    demo)
        run_demo_pipeline
        ;;
    clean)
        clean_env
        ;;
    all)
        echo "=== Setup ==="
        setup_env

        echo ""
        echo "=== Convert ==="
        run_conversion

        echo ""
        echo "=== Validate ==="
        run_validation

        echo ""
        echo "=== Demo ==="
        run_demo_pipeline

        echo ""
        echo "=== Clean ==="
        clean_env
        ;;
    *)
        echo "Error: Unknown action '$ACTION'. Use: setup, convert, validate, demo, clean"
        exit 1
        ;;
esac
