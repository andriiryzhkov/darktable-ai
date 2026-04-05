# Demo: run the pre-converted Depth Anything V2 Small ONNX model on an image
# and save the depth map as a grayscale PNG.
#
# Usage:
#   python3 models/mask-depth-da2-small/demo.py \
#       --model output/mask-depth-da2-small/model.onnx \
#       --image images/example_01.jpg \
#       --output models/mask-depth-da2-small/output/depth_01.png

import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(
    image: Image.Image, size: int = 518
) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize to fixed square resolution and ImageNet-normalize.

    DAv2-Small uses 518x518 input with dimensions divisible by 14
    (ViT patch size).
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


def demo(model, image, output, **kwargs):
    """Run depth inference on a single image."""
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    t0 = time.perf_counter()

    session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    t_model = time.perf_counter()
    print(f"    Load model:    {t_model - t0:.3f}s")

    img = Image.open(image)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    input_tensor, orig_size = preprocess(img)
    t_image = time.perf_counter()
    print(f"    Input:         {input_tensor.shape[3]}x{input_tensor.shape[2]}")
    print(f"    Load image:    {t_image - t_model:.3f}s")

    [depth] = session.run(None, {input_name: input_tensor})
    t_infer = time.perf_counter()
    print(f"    Inference:     {t_infer - t_image:.3f}s")

    depth_img = Image.fromarray(postprocess(depth))
    depth_img = depth_img.resize(orig_size, Image.LANCZOS)
    depth_img.save(output)
    print(f"    Total:         {t_infer - t0:.3f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Depth Anything V2 Small ONNX depth inference demo.")
    parser.add_argument("--model", type=str, required=True, help="Path to model.onnx")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, required=True, help="Output depth PNG path")
    args = parser.parse_args()
    demo(args.model, args.image, args.output)
