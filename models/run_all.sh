#!/bin/bash
set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for dir in "$SCRIPTS_DIR"/*/; do
    if [ -f "$dir/model.conf" ]; then
        model_id="$(basename "$dir")"
        if [ -f "$dir/.skip" ]; then
            echo "Skipping $model_id (.skip file found)"
            continue
        fi
        echo "========================================"
        echo "  $model_id"
        echo "========================================"
        bash "$SCRIPTS_DIR/run.sh" "$model_id"
        echo ""
    fi
done

echo "All models processed."
