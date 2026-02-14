#!/bin/bash
# Shared functions for model setup, conversion, and cleanup scripts.
#
# Usage: source this file after setting SCRIPT_DIR, VENV_DIR, and
# sourcing the model's model.conf.
#
# Expected variables from model.conf:
#   MODEL_ID              - e.g. "mask-object-samhq-light"
#   REPO_URL              - git clone URL
#   REPO_DIR              - local directory name for the cloned repo
#   REPO_INSTALL_CMD      - command to install the repo (run from repo dir)
#   REPO_HAS_REQUIREMENTS - "true" to pip install -r requirements.txt in repo
#   CHECKPOINT_URLS[]     - array of download URLs (supports direct URLs,
#                           Google Drive URLs, or gdrive://FILE_ID)
#   CHECKPOINT_PATHS[]    - array of paths relative to ROOT_DIR
#
# Optional variables for pre-converted ONNX models (no conversion needed):
#   ONNX_URLS[]           - array of pre-converted ONNX download URLs
#   ONNX_PATHS[]          - array of output paths relative to ROOT_DIR

ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

setup_venv() {
    if [ -d "$VENV_DIR" ]; then
        echo "Virtual environment already exists at $VENV_DIR"
    else
        echo "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi

    source "$VENV_DIR/bin/activate"

    echo "Upgrading pip..."
    pip install --upgrade pip

    echo "Installing project dependencies..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
}

clone_and_install_repo() {
    if ! [ -d "$SCRIPT_DIR/$REPO_DIR" ]; then
        echo "Cloning $REPO_DIR..."
        git clone "$REPO_URL" "$SCRIPT_DIR/$REPO_DIR"
    else
        echo "$REPO_DIR already cloned at $SCRIPT_DIR/$REPO_DIR"
    fi

    cd "$SCRIPT_DIR/$REPO_DIR"
    if [ "${REPO_HAS_REQUIREMENTS:-false}" = true ]; then
        pip install -r requirements.txt
    fi
    eval "$REPO_INSTALL_CMD"
    cd "$SCRIPT_DIR"
}

download_gdrive() {
    local file_id="$1"
    local output="$2"
    curl -L -o "$output" \
        "https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t"
}

# Extract Google Drive file ID from various URL formats:
#   https://drive.google.com/file/d/FILE_ID/view
#   https://drive.google.com/uc?id=FILE_ID
#   https://drive.google.com/open?id=FILE_ID
#   gdrive://FILE_ID
gdrive_file_id() {
    local url="$1"
    if [[ "$url" =~ ^gdrive://(.+)$ ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ "$url" =~ /file/d/([^/]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ "$url" =~ [\?\&]id=([^\&]+) ]]; then
        echo "${BASH_REMATCH[1]}"
    fi
}

download_checkpoints() {
    for i in "${!CHECKPOINT_URLS[@]}"; do
        local url="${CHECKPOINT_URLS[$i]}"
        local path="$ROOT_DIR/${CHECKPOINT_PATHS[$i]}"
        if ! [ -f "$path" ]; then
            echo "Downloading checkpoint: $(basename "$path")..."
            mkdir -p "$(dirname "$path")"
            local gdrive_id
            gdrive_id=$(gdrive_file_id "$url")
            if [ -n "$gdrive_id" ]; then
                download_gdrive "$gdrive_id" "$path"
            else
                curl -L "$url" -o "$path"
            fi
        else
            echo "Checkpoint already exists at $path"
        fi
    done
}

download_onnx_models() {
    for i in "${!ONNX_URLS[@]}"; do
        local url="${ONNX_URLS[$i]}"
        local path="$ROOT_DIR/${ONNX_PATHS[$i]}"
        if ! [ -f "$path" ]; then
            echo "Downloading ONNX model: $(basename "$path")..."
            mkdir -p "$(dirname "$path")"
            curl -L "$url" -o "$path"
        else
            echo "ONNX model already exists at $path"
        fi
    done
}

activate_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "Error: Virtual environment not found. Run ./setup_env.sh first."
        exit 1
    fi
    source "$VENV_DIR/bin/activate"
}

clean_env() {
    echo "Cleaning up environment for $MODEL_ID..."

    if [ -d "$VENV_DIR" ]; then
        echo "Removing virtual environment at $VENV_DIR..."
        rm -rf "$VENV_DIR"
    fi

    if [ -n "$REPO_DIR" ] && [ -d "$SCRIPT_DIR/$REPO_DIR" ]; then
        echo "Removing $REPO_DIR repository at $SCRIPT_DIR/$REPO_DIR..."
        rm -rf "$SCRIPT_DIR/$REPO_DIR"
    fi

    echo "Cleanup complete."
}
