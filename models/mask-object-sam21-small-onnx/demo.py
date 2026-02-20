# Demo: run the pre-converted SAM 2.1 Small ONNX model on an image with a
# single point prompt and save the source image with a red mask overlay.
#
# Usage:
#   python3 models/mask-object-sam21-small-onnx/demo.py \
#       --encoder output/mask-object-sam21-small-onnx/encoder.onnx \
#       --decoder output/mask-object-sam21-small-onnx/decoder.onnx \
#       --image images/example_03.jpg \
#       --point 0.28,0.65 \
#       --output models/mask-object-sam21-small-onnx/output/mask_03.png

import argparse
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

MODEL_SIZE = 1024
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize to 1024x1024 and ImageNet-normalize."""
    image = image.resize((MODEL_SIZE, MODEL_SIZE), Image.LANCZOS)
    arr = np.array(image).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # (3, H, W)
    return arr[np.newaxis]


def make_overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay a red mask on the source image."""
    img_arr = np.array(image).astype(np.float32)
    red = np.array([255.0, 0.0, 0.0])
    mask_3d = mask[:, :, np.newaxis]
    img_arr = img_arr * (1 - mask_3d * alpha) + red * mask_3d * alpha
    return Image.fromarray(img_arr.clip(0, 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="SAM 2.1 Small ONNX segmentation demo.")
    parser.add_argument("--encoder", type=str, required=True, help="Path to encoder.onnx")
    parser.add_argument("--decoder", type=str, required=True, help="Path to decoder.onnx")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--point", type=str, required=True,
                        help="Foreground point as normalized x,y (e.g. 0.28,0.65)")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Parse point
    px, py = [float(v) for v in args.point.split(",")]

    t0 = time.perf_counter()

    # Load models
    print(f"Loading encoder: {args.encoder}")
    enc_session = ort.InferenceSession(args.encoder, providers=["CPUExecutionProvider"])
    print(f"Loading decoder: {args.decoder}")
    dec_session = ort.InferenceSession(args.decoder, providers=["CPUExecutionProvider"])
    t_model = time.perf_counter()
    print(f"  Load models:   {t_model - t0:.3f}s")

    # Load and preprocess image
    print(f"Loading image: {args.image}")
    image = Image.open(args.image)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    orig_w, orig_h = image.size
    pixel_values = preprocess_image(image)
    t_image = time.perf_counter()
    print(f"  Original size: {orig_w}x{orig_h}")
    print(f"  Load image:    {t_image - t_model:.3f}s")

    # Run encoder
    # Outputs: image_embeddings.0 (high-res 32ch), .1 (mid-res 64ch), .2 (low-res 256ch)
    print("Running encoder...")
    enc_outputs = enc_session.run(None, {"pixel_values": pixel_values})
    image_emb_0, image_emb_1, image_emb_2 = enc_outputs
    t_enc = time.perf_counter()
    print(f"  Encoder:       {t_enc - t_image:.3f}s")

    # Prepare point prompt (normalized coords → 1024x1024 space)
    # Decoder expects input_points [B, 1, N, 2] and input_labels [B, 1, N]
    input_points = np.array([[[[px * MODEL_SIZE, py * MODEL_SIZE]]]], dtype=np.float32)
    input_labels = np.array([[[1]]], dtype=np.int64)
    # Empty box prompt [B, 0, 4]
    input_boxes = np.zeros((1, 0, 4), dtype=np.float32)
    print(f"  Point (1024):  ({input_points[0,0,0,0]:.0f}, {input_points[0,0,0,1]:.0f})")

    # Run decoder
    print("Running decoder...")
    dec_outputs = dec_session.run(None, {
        "image_embeddings.0": image_emb_0,
        "image_embeddings.1": image_emb_1,
        "image_embeddings.2": image_emb_2,
        "input_points": input_points,
        "input_labels": input_labels,
        "input_boxes": input_boxes,
    })
    iou_scores, pred_masks, obj_scores = dec_outputs
    t_dec = time.perf_counter()
    print(f"  Decoder:       {t_dec - t_enc:.3f}s")
    print(f"  Masks shape:   {pred_masks.shape}")
    print(f"  IoU scores:    {iou_scores[0, 0]}")

    # pred_masks shape: [B, num_prompts, num_masks, H, W]
    # Select best mask (highest IoU) from the point prompt
    best_idx = int(np.argmax(iou_scores[0, 0]))
    mask_logits = pred_masks[0, 0, best_idx]
    print(f"  Best mask:     #{best_idx} (IoU={iou_scores[0, 0, best_idx]:.4f})")

    # Upscale mask to original image size
    mask_h, mask_w = mask_logits.shape
    mask_img = Image.fromarray((mask_logits > 0).astype(np.uint8) * 255)
    mask_full = mask_img.resize((orig_w, orig_h), Image.LANCZOS)
    mask_binary = (np.array(mask_full) > 127).astype(np.float32)

    # Save overlay
    result = make_overlay(image, mask_binary)
    result.save(args.output)
    t_total = time.perf_counter()
    print(f"Saved: {args.output}")
    print(f"  Total:         {t_total - t0:.3f}s")


if __name__ == "__main__":
    main()
