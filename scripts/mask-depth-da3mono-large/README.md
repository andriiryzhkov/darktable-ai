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

## Selection Criteria

| Property                 | Value                                                                              |
|--------------------------|------------------------------------------------------------------------------------|
| Model license            | Apache-2.0                                                                         |
| OSAID v1.0               | Open Weights                                                                       |
| MOF                      | Class II (Open Tooling)                                                            |
| Training data license    | Mixed — all public academic datasets (per authors)                                 |
| Training data provenance | ~69 datasets (synthetic + sensor + SfM/stereo). DINOv2-Large backbone on LVD-142M |
| Training code            | [Apache-2.0](https://github.com/ByteDance-Seed/Depth-Anything-3)                  |
| Known limitations        | LVD-142M: 142M web-crawled images, not released by Meta, not auditable. ~69 constituent datasets have varying individual licenses |
| Published research       | [Depth Anything 3](https://arxiv.org/abs/2511.10647) (2025)                       |
| Inference                | Local only, no cloud dependencies                                                  |
| Scope                    | Depth estimation                                                                   |
| Reproducibility          | Full pipeline (setup, convert, clean, demo)                                        |
