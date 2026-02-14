# Depth Anything 3 Mono-Large

Monocular depth estimation — large variant from Depth Anything 3. Converted
from PyTorch (safetensors) to ONNX.

## Source

- Repository: https://github.com/ByteDance-Seed/Depth-Anything-3
- Model: https://huggingface.co/depth-anything/DA3MONO-LARGE
- Paper: [Depth Anything 3](https://arxiv.org/abs/2511.10647) (2025)
- License: Apache-2.0

## Architecture

DINOv2-Large encoder + DualDPT decoder. The model predicts 2 channels
(depth + confidence); only the depth channel is exported.

## ONNX Model

| Property    | Value                                     |
|-------------|-------------------------------------------|
| File        | `model.onnx`                              |
| Input       | `image` — float32 [1, 3, 504, 504]       |
| Output      | `depth` — float32 [1, 1, 504, 504]       |
| Resolution  | Fixed 504x504 (RoPE embeddings baked)     |
| Normalize   | ImageNet mean/std                         |
| Tiling      | No                                        |

## Notes

- Outputs **depth** (not disparity): close objects are dark (low values), far objects are bright (high values). Opposite convention from DAv2.
- Resolution is fixed at 504x504 — RoPE positional embeddings are baked at export time, so input must match exactly.
- Requires `bfloat16 → float16` patch during export (ONNX does not support bfloat16).
- Heavy dependencies (xformers, open3d, etc.) are stubbed out during export — only the core model is needed.

## Selection criteria compliance

- **License:** Apache-2.0 -- permissive, GPL-3.0 compatible.
- **Open weights:** Publicly available on HuggingFace, no registration required.
- **Published research:** Published with technical report (arXiv 2025).
- **Training data:** Synthetic datasets + large-scale pseudo-labeled real images with scale-and-shift-invariant training. All training sources are publicly documented.
- **Purpose:** Depth estimation for masking - photo editing task. Runs locally, no external services.
