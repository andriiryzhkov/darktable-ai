# Darktable AI Models

ONNX models and conversion scripts for darktable image processing.

## Models

| Model               | Task    | Description                             |
|---------------------|---------|-----------------------------------------|
| `denoise-nafnet`    | denoise | NAFNet denoiser trained on SIDD dataset |
| `denoise-nind`      | denoise | UNet denoiser trained on NIND dataset   |
| `mask-samhq-light`  | mask    | SAM-HQ Light (ViT-Tiny) for masking    |
| `mask-samhq-base`   | mask    | SAM-HQ (ViT-B) for masking             |
| `mask-sam21-tiny`   | mask    | SAM 2.1 (Hiera Tiny) for masking       |
| `mask-sam21-small`  | mask    | SAM 2.1 (Hiera Small) for masking      |

## Repository structure

```
checkpoints/          PyTorch checkpoints (downloaded by setup scripts)
models/               ONNX models + config.json per model
scripts/
  common.sh           Shared shell functions for setup/clean/convert
  run.sh              Run full pipeline (setup → convert → clean) for a model
  run_all.sh          Run full pipeline for all models
  <model>/
    model.conf        Model-specific configuration (repo URL, checkpoint, etc.)
    requirements.txt  Python dependencies
    setup_env.sh      Create venv, clone repo, download checkpoint
    run_conversion.sh Convert PyTorch checkpoint to ONNX
    clean_env.sh      Remove venv and cloned repo
    convert_*.py      Model-specific conversion script
```

## Usage

Run the full pipeline (setup, convert, clean) for a single model:

```bash
./scripts/run.sh <model_id>
```

Run the pipeline for all models:

```bash
./scripts/run_all.sh
```

Or run each step individually:

```bash
./scripts/<model>/setup_env.sh        # Create venv, clone repo, download checkpoint
./scripts/<model>/run_conversion.sh   # Convert to ONNX
./scripts/<model>/clean_env.sh        # Remove venv and cloned repo
```

## Adding a new model

1. Create `scripts/<model>/model.conf` with repo URL, install command, and checkpoint URLs
2. Create `scripts/<model>/requirements.txt` with Python dependencies
3. Create `scripts/<model>/convert_*.py` with model-specific conversion logic
4. Copy `setup_env.sh`, `run_conversion.sh`, `clean_env.sh` from an existing model (they are identical thin wrappers around `scripts/common.sh`)
5. Create `models/<model>/config.json` with model metadata
