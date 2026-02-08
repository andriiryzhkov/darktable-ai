# Darktable AI Models

ONNX models and conversion scripts for darktable image processing.

## Models

| Model                | Task    | Description                                  |
|----------------------|---------|----------------------------------------------|
| `denoise-nafnet`     | denoise | NAFNet denoiser trained on SIDD dataset      |
| `mask-light-hq-sam`  | mask    | Light HQ Segment Anything Model for masking  |

## Repository structure

```
checkpoints/          PyTorch checkpoints (downloaded by setup scripts)
models/               ONNX models + config.json per model
scripts/
  common.sh           Shared shell functions for setup/clean/convert
  <model>/
    model.conf        Model-specific configuration (repo URL, checkpoint, etc.)
    requirements.txt  Python dependencies
    setup_env.sh      Create venv, clone repo, download checkpoint
    run_conversion.sh Convert PyTorch checkpoint to ONNX
    clean_env.sh      Remove venv and cloned repo
    convert_*.py      Model-specific conversion script
```

## Usage

Setup environment and download checkpoint:

```bash
./scripts/<model>/setup_env.sh
```

Convert to ONNX:

```bash
./scripts/<model>/run_conversion.sh
```

Clean up:

```bash
./scripts/<model>/clean_env.sh
```

## Adding a new model

1. Create `scripts/<model>/model.conf` with repo URL, install command, and checkpoint URLs
2. Create `scripts/<model>/requirements.txt` with Python dependencies
3. Create `scripts/<model>/convert_*.py` with model-specific conversion logic
4. Copy `setup_env.sh`, `run_conversion.sh`, `clean_env.sh` from an existing model (they are identical thin wrappers around `scripts/common.sh`)
5. Create `models/<model>/config.json` with model metadata
