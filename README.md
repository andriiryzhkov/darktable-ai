# Darktable AI Models

ONNX models and conversion scripts for [darktable](https://www.darktable.org/) - an open-source photography workflow application and raw developer ([GitHub](https://github.com/darktable-org/darktable)).

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
| [`mask-depth-da2-small`](scripts/mask-depth-da2-small/README.md)                 | depth    | Depth Anything V2 Small for depth masking     |
| [`mask-depth-da3mono-large`](scripts/mask-depth-da3mono-large/README.md)         | depth    | Depth Anything 3 Mono-Large for depth masking |
| [`erase-lama-big`](scripts/erase-lama-big/README.md)                             | erase   | LaMa Big for object erasure (inpainting)      |

## Repository structure

```
images/               Sample images for demos
models/               ONNX models + config.json per model
temp/                 Downloaded checkpoints (before conversion, gitignored)
scripts/
  common.sh           Shared shell functions (setup, convert, clean, demo)
  run.sh              Run full pipeline (setup -> convert -> clean) for a model
  run_all.sh          Run full pipeline for all models
  <model>/
    model.conf        Model configuration (repo URL, checkpoints, etc.)
    requirements.txt  Python dependencies
    setup_env.sh      Create venv, clone repo, download checkpoints
    run_conversion.sh Convert PyTorch checkpoint to ONNX
    clean_env.sh      Remove venv and cloned repo
    convert_*.py      Model-specific conversion script
    demo.py           Demo inference script
    run_demo.sh       Run demo on all sample images
    README.md         Model documentation and ONNX tensor specs
    .skip             If present, skip this model in run_all.sh and CI
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
./scripts/<model>/run_demo.sh.        # Run a demo
./scripts/<model>/clean_env.sh        # Remove venv and cloned repo
```

## Demos

Each model includes a demo script that runs inference on the sample images in `images/`.

Run a demo for a single model (requires the venv and ONNX model to be present):

```bash
./scripts/<model>/run_demo.sh
```

Demo scripts use the shared `run_demo` function from `common.sh`, which iterates
over all sample images automatically. Models that require per-image input (e.g.
point prompts for object segmentation) define a `demo_args()` function in their
`run_demo.sh`.

Output images are saved to `scripts/<model>/output/`.

## Model selection criteria

Darktable is free software licensed under [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html). All AI models included in this repository are selected with the following principles in mind.

### Open-source and licensing

- **GPL-3.0-compatible licenses only.** Every model must be released under a license compatible with GPL-3.0 (e.g. Apache-2.0, MIT, BSD, GPL-3.0). Proprietary or non-commercial-only models are not accepted.
- **Open weights.** Model weights must be publicly available for download without registration walls or usage restrictions.
- **Published research.** Models should have an accompanying peer-reviewed paper or public technical report describing the architecture and training methodology.

### AI ethics

- **Training data transparency.** We prefer models trained on publicly documented datasets with clear provenance. Models trained on undisclosed or scraped personal data without consent are not accepted.
- **Privacy by design.** All inference runs locally on the user's machine. No data is sent to external services. No telemetry, no cloud dependencies.
- **Purpose-limited scope.** Models are selected for photo editing tasks: denoising, masking, depth estimation, and object removal (inpainting), etc. We do not include models designed for generating, manipulating, or synthesizing human likenesses.
- **Reproducibility.** Conversion scripts, model configurations, and source references are fully documented so that any user can verify and rebuild the ONNX models from the original checkpoints.

## Adding a new model

1. Create `scripts/<model>/model.conf` with repo URL, install command, and checkpoint URLs
2. Create `scripts/<model>/requirements.txt` with Python dependencies
3. Create `scripts/<model>/convert_*.py` with model-specific conversion logic
4. Copy `setup_env.sh`, `run_conversion.sh`, `clean_env.sh` from an existing model (they are identical thin wrappers around `scripts/common.sh`)
5. Create `models/<model>/config.json` with model metadata
