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

## Selection Criteria

| Property                 | Value                                                                                                      |
|--------------------------|------------------------------------------------------------------------------------------------------------|
| Model license            | Apache-2.0                                                                                                 |
| OSAID v1.0               | Open Weights                                                                                               |
| MOF                      | Class II (Open Tooling)                                                                                    |
| Training data license    | Mixed (SA-1B: Meta research-only; Places365: non-commercial; others: various open)                         |
| Training data provenance | Teacher: [Hypersim](https://github.com/apple/ml-hypersim), [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/), [TartanAir](https://theairlab.org/tartanair-dataset/), [BlendedMVS](https://github.com/YoYo000/BlendedMVS), IRS (595K). Student: [BDD100K](https://www.bdd100k.com/), [Google Landmarks](https://github.com/cvdfoundation/google-landmark), [ImageNet-21K](https://www.image-net.org/), [LSUN](https://www.yf.io/p/lsun), [Objects365](https://www.objects365.org/), [Open Images V7](https://storage.googleapis.com/openimages/web/index.html), [Places365](http://places2.csail.mit.edu/), [SA-1B](https://ai.meta.com/datasets/segment-anything/) (62M). DINOv2-Small on LVD-142M (not public) |
| Training code            | [Apache-2.0](https://github.com/DepthAnything/Depth-Anything-V2)                                          |
| Known limitations        | LVD-142M: 142M web-crawled images, not released by Meta, not auditable. SA-1B: research-only (not OSI). Places365: non-commercial research terms |
| Published research       | [Depth Anything V2](https://arxiv.org/abs/2406.09414) (2024)                                              |
| Inference                | Local only, no cloud dependencies                                                                          |
| Scope                    | Depth estimation                                                                                           |
| Reproducibility          | Pre-converted ONNX; full demo pipeline provided                                                            |
