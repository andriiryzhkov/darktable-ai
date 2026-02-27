# ------------------------------------------------------------------------
# Darktable AI Upscale - ONNX Demo
# ------------------------------------------------------------------------

import argparse
import sys
import os
import cv2
import numpy as np
import onnxruntime as ort
import time


def run_inference(model_path, input_path, output_path, max_size=512):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found at {input_path}")

    start_time = time.time()
    print(f"Loading ONNX model: {model_path}")
    try:
        session = ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    model_input = session.get_inputs()[0]
    input_name = model_input.name
    input_is_fp16 = model_input.type == 'tensor(float16)'

    # 1. Read image
    print(f"Reading image: {input_path}")
    img = cv2.imread(input_path)  # BGR
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")
    h, w = img.shape[:2]
    print(f"  Original size: {w}x{h}")
    if max_size > 0 and max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        print(f"  Resized to:    {img.shape[1]}x{img.shape[0]}")

    # 2. Preprocessing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = img[None, :, :, :]

    if input_is_fp16:
        img = img.astype(np.float16)

    # 3. Run Inference
    print("Running inference...")
    try:
        output = session.run(None, {input_name: img})[0]
    except Exception as e:
        print(f"Inference failed: {e}")
        sys.exit(1)

    # 4. Postprocessing
    output = output[0].astype(np.float32)
    output = output.transpose(1, 2, 0)
    output = np.clip(output, 0, 1)
    output = (output * 255.0).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    out_h, out_w = output.shape[:2]
    print(f"  Output size: {out_w}x{out_h}")

    print(f"Saving result to: {output_path}")
    cv2.imwrite(output_path, output)
    print("Done.")
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.4f} seconds")


def demo(model_dir, image, output, **kwargs):
    """Entry point for programmatic demo. Runs all *.onnx in model_dir."""
    import glob

    max_size = kwargs.get("max_size", 512)
    models = sorted(glob.glob(os.path.join(model_dir, "*.onnx")))
    if not models:
        raise FileNotFoundError(f"No ONNX models found in {model_dir}")
    base, ext = os.path.splitext(output)
    for model_path in models:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = f"{base}_{model_name}{ext}"
        print(f"\n--- {model_name} ---")
        run_inference(model_path, image, output_path, max_size=max_size)


def main():
    parser = argparse.ArgumentParser(description='Run BSRGAN ONNX Demo')
    parser.add_argument('--model', type=str)
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--max-size', type=int, default=512)
    args = parser.parse_args()

    if args.model_dir:
        demo(args.model_dir, args.image, args.output, max_size=args.max_size)
    elif args.model:
        run_inference(args.model, args.image, args.output, max_size=args.max_size)
    else:
        print("Error: provide --model or --model-dir")
        sys.exit(1)


if __name__ == '__main__':
    main()
