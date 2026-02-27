"""Run demo inference on sample images."""

from __future__ import annotations

import json
from pathlib import Path

from darktable_ai.config import ModelConfig
from darktable_ai.convert import _import_script


def run_demo(config: ModelConfig) -> None:
    """Run the model's demo.py on all sample images for its task."""
    images_dir = config.root_dir / "samples" / config.task
    if not images_dir.is_dir():
        print(f"  No samples directory found: {images_dir}")
        return

    demo_script = config.model_dir / "demo.py"
    if not demo_script.is_file():
        print(f"  No demo.py found in {config.model_dir}")
        return

    demo_output_dir = config.root_dir / "output" / f"{config.id}-demo"
    demo_output_dir.mkdir(parents=True, exist_ok=True)

    module = _import_script(demo_script)
    model_kwargs = _model_type_kwargs(config)

    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        if img.stem.startswith("expected"):
            continue

        name = img.stem
        output_path = demo_output_dir / f"{name}.png"
        extra_kwargs = _image_kwargs(config, name)

        print(f"  {name}")
        module.demo(
            **model_kwargs,
            image=str(img),
            output=str(output_path),
            **extra_kwargs,
        )


def _model_type_kwargs(config: ModelConfig) -> dict:
    """Build model path kwargs for demo() based on model type."""
    output_dir = config.output_dir
    if config.type == "split":
        return {
            "encoder": str(output_dir / "encoder.onnx"),
            "decoder": str(output_dir / "decoder.onnx"),
        }
    elif config.type == "multi":
        return {"model_dir": str(output_dir)}
    else:
        return {"model": str(output_dir / "model.onnx")}


def _image_kwargs(config: ModelConfig, image_name: str) -> dict:
    """Get extra demo kwargs for a specific image.

    Reads from a JSON sidecar file next to the sample image first
    (e.g. ``samples/mask-object/example_01.json``), then falls back
    to ``demo.image_args`` in model.yaml.
    """
    sidecar = config.root_dir / "samples" / config.task / f"{image_name}.json"
    if sidecar.is_file():
        with open(sidecar) as f:
            return json.load(f)
    return config.demo.image_args.get(image_name, {})
