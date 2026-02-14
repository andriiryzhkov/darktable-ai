# Darktable AI Models

ONNX models and conversion scripts for darktable image processing.

## Models

| Model                                                                            | Task    | Description                                   |
|----------------------------------------------------------------------------------|---------|-----------------------------------------------|
| [`denoise-nafnet`](scripts/denoise-nafnet/README.md)                             | denoise | NAFNet denoiser trained on SIDD dataset       |
| [`denoise-nind`](scripts/denoise-nind/README.md)                                 | denoise | UNet denoiser trained on NIND dataset         |
| [`mask-object-samhq-light`](scripts/mask-object-samhq-light/README.md)           | mask    | HQ-SAM Light (ViT-Tiny) for masking           |
| [`mask-object-samhq-base`](scripts/mask-object-samhq-base/README.md)             | mask    | HQ-SAM (ViT-B) for masking                    |
| [`mask-object-sam21-tiny`](scripts/mask-object-sam21-tiny/README.md)             | mask    | SAM 2.1 (Hiera Tiny) for masking              |
| [`mask-object-sam21-small`](scripts/mask-object-sam21-small/README.md)           | mask    | SAM 2.1 (Hiera Small) for masking             |
| [`mask-object-sam21-small-onnx`](scripts/mask-object-sam21-small-onnx/README.md) | mask    | SAM 2.1 (Hiera Small) pre-converted ONNX      |
| [`mask-object-sam21hq-large`](scripts/mask-object-sam21hq-large/README.md)       | mask    | HQ-SAM 2.1 (Hiera Large) for masking          |
| [`mask-depth-da2-small`](scripts/mask-depth-da2-small/README.md)                 | mask    | Depth Anything V2 Small for depth masking     |
| [`mask-depth-da3mono-large`](scripts/mask-depth-da3mono-large/README.md)         | mask    | Depth Anything 3 Mono-Large for depth masking |

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
