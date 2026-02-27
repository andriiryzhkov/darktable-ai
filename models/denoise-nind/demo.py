# Demo: run the NIND UNet denoiser ONNX model on an image.

import argparse
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


def run_inference(model_path, image_path, output_path, max_size=1024):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    t0 = time.perf_counter()

    print(f"Loading model: {model_path}")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    input_name = model_input.name
    input_is_fp16 = model_input.type == "tensor(float16)"
    t_model = time.perf_counter()
    print(f"  Input name:    {input_name}")
    print(f"  FP16:          {input_is_fp16}")
    print(f"  Load model:    {t_model - t0:.3f}s")

    print(f"Loading image: {image_path}")
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    t_image = time.perf_counter()
    print(f"  Original size: {image.size[0]}x{image.size[1]}")
    if max_size > 0:
        image.thumbnail((max_size, max_size), Image.LANCZOS)
        print(f"  Resized to:    {image.size[0]}x{image.size[1]}")
    print(f"  Load image:    {t_image - t_model:.3f}s")

    # Preprocess: RGB [0, 1], BCHW
    arr = np.array(image).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]

    # Pad all edges to hide UNet border artifacts, then align to multiple of 16
    _, _, h, w = arr.shape
    border = 16
    ph = border + (16 - (h + 2 * border) % 16) % 16
    pw = border + (16 - (w + 2 * border) % 16) % 16
    arr = np.pad(arr, ((0, 0), (0, 0), (border, ph), (border, pw)), mode="reflect")
    print(f"  Padded to:     {arr.shape[3]}x{arr.shape[2]} (border={border})")

    if input_is_fp16:
        arr = arr.astype(np.float16)

    print("Running inference...")
    [output] = session.run(None, {input_name: arr})
    t_infer = time.perf_counter()
    print(f"  Inference:     {t_infer - t_image:.3f}s")

    # Postprocess: crop padding, BCHW -> HWC, clip, uint8
    output = output[:, :, border:border + h, border:border + w]
    output = output[0].astype(np.float32).transpose(1, 2, 0)
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)

    Image.fromarray(output).save(output_path)
    print(f"Saved: {output_path}")
    print(f"  Total:         {time.perf_counter() - t0:.3f}s")


def demo(model, image, output, **kwargs):
    """Entry point for programmatic demo."""
    run_inference(model, image, output, max_size=kwargs.get("max_size", 1024))


def main():
    parser = argparse.ArgumentParser(description="NIND UNet ONNX denoising demo.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-size", type=int, default=1024)
    args = parser.parse_args()

    demo(args.model, args.image, args.output, max_size=args.max_size)


if __name__ == "__main__":
    main()
