# Demo: run the exported DA3Mono-Large ONNX model on an image and save the
# depth map as a grayscale PNG.
#
# Usage:
#   python3 models/mask-depth-da3mono-large/demo.py \
#       --model output/mask-depth-da3mono-large/model.onnx \
#       --image images/example_01.jpg \
#       --output models/mask-depth-da3mono-large/output/depth_01.png

import argparse
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(
    image: Image.Image, size: int = 504
) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize to fixed square resolution and ImageNet-normalize.

    RoPE positional embeddings are baked at the export resolution, so the
    input must match exactly (504x504 by default).
    """
    orig_size = image.size  # (W, H)
    image = image.resize((size, size), Image.LANCZOS)

    arr = np.array(image).astype(np.float32) / 255.0  # (H, W, 3)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # (3, H, W)
    return arr[np.newaxis], orig_size


def postprocess(depth: np.ndarray) -> np.ndarray:
    """Normalize depth to 0-255 uint8."""
    d = depth.squeeze()  # (H, W)
    d_min, d_max = d.min(), d.max()
    if d_max - d_min > 1e-6:
        d = (d - d_min) / (d_max - d_min)
    else:
        d = np.zeros_like(d)
    return (d * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="DA3Mono-Large ONNX depth inference demo.")
    parser.add_argument("--model", type=str, required=True, help="Path to model.onnx")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, required=True, help="Output depth PNG path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    t0 = time.perf_counter()

    print(f"Loading model: {args.model}")
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    t_model = time.perf_counter()
    print(f"  Load model:    {t_model - t0:.3f}s")

    print(f"Loading image: {args.image}")
    image = Image.open(args.image)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    input_tensor, orig_size = preprocess(image)
    t_image = time.perf_counter()
    print(f"  Original size: {orig_size[0]}x{orig_size[1]}")
    print(f"  Model input:   {input_tensor.shape[3]}x{input_tensor.shape[2]}")
    print(f"  Load image:    {t_image - t_model:.3f}s")

    print("Running inference...")
    [depth] = session.run(None, {"image": input_tensor})
    t_infer = time.perf_counter()
    print(f"  Output shape:  {depth.shape}")
    print(f"  Inference:     {t_infer - t_image:.3f}s")

    depth_img = Image.fromarray(postprocess(depth))
    depth_img = depth_img.resize(orig_size, Image.LANCZOS)
    depth_img.save(args.output)
    print(f"Saved depth map: {args.output}")
    print(f"  Total:         {t_infer - t0:.3f}s")


if __name__ == "__main__":
    main()
