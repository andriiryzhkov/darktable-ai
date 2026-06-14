"""Run BiRefNet Lite on an image and save a red-mask overlay.

Usage:
  python3 models/mask-subject-birefnet-lite/demo.py \
      --model output/mask-subject-birefnet-lite/model.onnx \
      --image samples/mask-object/example_03.jpg \
      --output models/mask-subject-birefnet-lite/output/example_03.png
"""

import argparse
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

MODEL_SIZE = 1024

# Standard ImageNet normalization (RGB in [0, 1]).
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def preprocess(image, ort_type):
    image = image.resize((MODEL_SIZE, MODEL_SIZE), Image.LANCZOS)
    arr = np.array(image).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]   # (1, 3, H, W)
    arr = (arr - MEAN) / STD
    np_dtype = np.float16 if ort_type == "tensor(float16)" else np.float32
    return arr.astype(np_dtype)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def overlay(image, mask, alpha=0.45):
    img_arr = np.array(image).astype(np.float32)
    red = np.array([255.0, 0.0, 0.0])
    mask_3d = mask[:, :, np.newaxis]
    img_arr = img_arr * (1 - mask_3d * alpha) + red * mask_3d * alpha
    return Image.fromarray(img_arr.clip(0, 255).astype(np.uint8))


def run_inference(model_path, image_path, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    t0 = time.perf_counter()
    print(f"Loading model: {model_path}")
    session = ort.InferenceSession(model_path,
                                   providers=["CPUExecutionProvider"])
    in_meta = session.get_inputs()[0]
    print(f"  Load: {time.perf_counter() - t0:.3f}s  "
          f"input={in_meta.name} {in_meta.shape} {in_meta.type}")

    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    orig_size = image.size

    arr = preprocess(image, in_meta.type)

    t1 = time.perf_counter()
    raw = session.run(None, {in_meta.name: arr})[0]
    print(f"  Inference: {time.perf_counter() - t1:.3f}s")

    raw = raw.astype(np.float32)
    # the published ONNX outputs logits; apply sigmoid only if the range
    # actually looks like logits, so re-exports that bake sigmoid in stay
    # correct
    if raw.min() < 0.0 or raw.max() > 1.0:
        raw = sigmoid(raw)
    print(f"  Mask stats: min={raw.min():.4f} max={raw.max():.4f} "
          f"mean={raw.mean():.4f}")

    # raw shape is (1, 1, H, W); collapse to (H, W) then back to the source
    # image's pixel grid
    mask = raw[0, 0]
    mask_img = Image.fromarray((mask * 255.0).clip(0, 255).astype(np.uint8))
    mask_img = mask_img.resize(orig_size, Image.LANCZOS)
    mask_np = np.array(mask_img).astype(np.float32) / 255.0

    out_img = overlay(image, mask_np)
    out_img.save(output_path)
    print(f"Saved {output_path}")


def demo(model, image, output, **kwargs):
    """Pipeline entry point."""
    run_inference(model, image, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    demo(args.model, args.image, args.output)


if __name__ == "__main__":
    main()
