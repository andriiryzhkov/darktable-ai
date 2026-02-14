# Depth Anything V2 Small

Monocular depth estimation — small variant. Pre-converted ONNX model from
HuggingFace (no local conversion needed).

## Source

- Repository: https://github.com/DepthAnything/Depth-Anything-V2
- ONNX source: https://huggingface.co/onnx-community/depth-anything-v2-small
- Paper: [Depth Anything V2](https://arxiv.org/abs/2406.09414) (2024)
- License: Apache-2.0

## Architecture

DINOv2-Small encoder + DPT (Dense Prediction Transformer) decoder.

## ONNX Model

| Property    | Value                                     |
|-------------|-------------------------------------------|
| File        | `model.onnx`                              |
| Input       | `pixel_values` — float32 [1, 3, 518, 518] |
| Output      | `predicted_depth` — float32 [1, 518, 518] |
| Resolution  | 518x518 (divisible by 14, ViT patch size) |
| Normalize   | ImageNet mean/std                         |
| Tiling      | No                                        |

## Notes

- Outputs **disparity** (inverse depth): close objects are bright (high values), far objects are dark (low values).
- Pre-converted ONNX from HuggingFace — no PyTorch conversion step.
- Full precision (float32).
- Input/output tensor names may vary — use `session.get_inputs()[0].name` at runtime.
