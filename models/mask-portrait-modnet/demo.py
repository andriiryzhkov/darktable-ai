"""Run MODNet on an image and save a red-mask overlay.

The preprocessing pipeline (normalization, resize policy, output activation)
is driven entirely by `config.json` written next to `model.onnx` by the
`dtai run` pipeline. If `config.json` is missing the demo falls back to the
upstream MODNet defaults so the script still works against a raw download.

Usage:
  python3 models/mask-portrait-modnet/demo.py \
      --model output/mask-portrait-modnet/model.onnx \
      --image samples/mask-object/example_portrait.jpg \
      --output models/mask-portrait-modnet/output/example_portrait.png
"""

import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


# Upstream MODNet defaults, used only when config.json is absent.
DEFAULTS = {
    "input_sizes": [512],
    "resize_mode": "longest_side",
    "size_multiple": 32,
    "color_space": "rgb",
    "norm_mean": [127.5, 127.5, 127.5],
    "norm_std":  [127.5, 127.5, 127.5],
    "output_kind": "alpha_matte",
    "output_activation": "none",
}


def _load_attributes(model_path):
    """Read attributes from config.json next to the model, else defaults."""
    config_path = os.path.join(os.path.dirname(model_path), "config.json")
    if not os.path.isfile(config_path):
        print(f"  (no config.json next to {model_path}; using defaults)")
        return dict(DEFAULTS)
    with open(config_path) as f:
        data = json.load(f)
    attrs = dict(DEFAULTS)
    attrs.update(data.get("attributes", {}))
    return attrs


def fit_dims(width, height, attrs):
    """Compute the resized (w, h) per the configured resize policy."""
    ref = attrs["input_sizes"][0]
    mode = attrs["resize_mode"]
    mult = attrs["size_multiple"]

    if mode == "longest_side":
        if width >= height:
            new_w, new_h = ref, max(int(round(ref * height / width)), 1)
        else:
            new_w, new_h = max(int(round(ref * width / height)), 1), ref
    elif mode == "shortest_side":
        if width <= height:
            new_w, new_h = ref, max(int(round(ref * height / width)), 1)
        else:
            new_w, new_h = max(int(round(ref * width / height)), 1), ref
    elif mode == "square":
        new_w = new_h = ref
    else:
        raise ValueError(f"unsupported resize_mode {mode!r}")

    new_w = max(((new_w + mult - 1) // mult) * mult, mult)
    new_h = max(((new_h + mult - 1) // mult) * mult, mult)
    return new_w, new_h


def preprocess(image, attrs, ort_type):
    if attrs["color_space"] != "rgb":
        raise ValueError(f"unsupported color_space {attrs['color_space']!r}")

    w, h = fit_dims(*image.size, attrs)
    image = image.resize((w, h), Image.LANCZOS)
    arr = np.array(image).astype(np.float32)

    mean = np.array(attrs["norm_mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.array(attrs["norm_std"], dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std

    arr = arr.transpose(2, 0, 1)[np.newaxis]   # (1, 3, H, W)
    np_dtype = np.float16 if ort_type == "tensor(float16)" else np.float32
    return arr.astype(np_dtype)


def apply_output_activation(raw, attrs):
    activation = attrs["output_activation"]
    if activation == "none":
        return raw
    if activation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-raw))
    raise ValueError(f"unsupported output_activation {activation!r}")


def overlay(image, mask, alpha=0.45):
    img_arr = np.array(image).astype(np.float32)
    red = np.array([255.0, 0.0, 0.0])
    mask_3d = mask[:, :, np.newaxis]
    img_arr = img_arr * (1 - mask_3d * alpha) + red * mask_3d * alpha
    return Image.fromarray(img_arr.clip(0, 255).astype(np.uint8))


def run_inference(model_path, image_path, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    attrs = _load_attributes(model_path)

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

    arr = preprocess(image, attrs, in_meta.type)

    t1 = time.perf_counter()
    raw = session.run(None, {in_meta.name: arr})[0]
    print(f"  Inference: {time.perf_counter() - t1:.3f}s")

    raw = raw.astype(np.float32)
    raw = apply_output_activation(raw, attrs)
    raw = np.clip(raw, 0.0, 1.0)
    print(f"  Mask stats: min={raw.min():.4f} max={raw.max():.4f} "
          f"mean={raw.mean():.4f}  kind={attrs['output_kind']}")

    matte = raw[0, 0]
    matte_img = Image.fromarray((matte * 255.0).clip(0, 255).astype(np.uint8))
    matte_img = matte_img.resize(orig_size, Image.LANCZOS)
    matte_np = np.array(matte_img).astype(np.float32) / 255.0

    out_img = overlay(image, matte_np)
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
