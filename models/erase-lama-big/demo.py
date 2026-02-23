# Demo: run the LaMa Big ONNX inpainting model on an image with a
# rectangular mask and save the inpainted result.
#
# Usage:
#   python3 models/erase-lama-big/demo.py \
#       --model output/erase-lama-big/model.onnx \
#       --image images/example_03.jpg \
#       --mask 0.15,0.45,0.42,0.90 \
#       --output models/erase-lama-big/output/example_03.png

import argparse
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

MODEL_SIZE = 512


def main():
    parser = argparse.ArgumentParser(description="LaMa Big ONNX inpainting demo.")
    parser.add_argument("--model", type=str, required=True, help="Path to model.onnx")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--mask", type=str, required=True,
                        help="Rectangular mask as normalized x1,y1,x2,y2 (e.g. 0.15,0.45,0.42,0.90)")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Parse mask rectangle (normalized coordinates)
    x1, y1, x2, y2 = [float(v) for v in args.mask.split(",")]

    t0 = time.perf_counter()

    print(f"Loading model: {args.model}")
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    t_model = time.perf_counter()
    print(f"  Load model:    {t_model - t0:.3f}s")

    print(f"Loading image: {args.image}")
    image = Image.open(args.image)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    orig_w, orig_h = image.size
    t_image = time.perf_counter()
    print(f"  Original size: {orig_w}x{orig_h}")
    print(f"  Load image:    {t_image - t_model:.3f}s")

    # Resize to model input size
    image_resized = image.resize((MODEL_SIZE, MODEL_SIZE), Image.LANCZOS)

    # Preprocess image: RGB [0, 1], BCHW
    img_arr = np.array(image_resized).astype(np.float32) / 255.0
    img_arr = img_arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 512, 512)

    # Create binary mask: 1 = inpaint, 0 = keep
    mask_arr = np.zeros((1, 1, MODEL_SIZE, MODEL_SIZE), dtype=np.float32)
    mx1 = int(x1 * MODEL_SIZE)
    my1 = int(y1 * MODEL_SIZE)
    mx2 = int(x2 * MODEL_SIZE)
    my2 = int(y2 * MODEL_SIZE)
    mask_arr[0, 0, my1:my2, mx1:mx2] = 1.0
    print(f"  Mask rect:     ({mx1}, {my1}) -> ({mx2}, {my2})")

    # Run inference
    print("Running inference...")
    [output] = session.run(None, {"image": img_arr, "mask": mask_arr})
    t_infer = time.perf_counter()
    print(f"  Inference:     {t_infer - t_image:.3f}s")

    # Postprocess: BCHW -> HWC, clip, uint8
    output = output[0].astype(np.float32).transpose(1, 2, 0)
    output = np.clip(output, 0, 1)
    result = Image.fromarray((output * 255).astype(np.uint8))

    # Resize back to original size
    result = result.resize((orig_w, orig_h), Image.LANCZOS)
    result.save(args.output)
    print(f"Saved: {args.output}")
    print(f"  Total:         {time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    main()
