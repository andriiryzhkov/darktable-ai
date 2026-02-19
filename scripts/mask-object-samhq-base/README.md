# HQ-SAM (ViT-B)

Segment Anything in High Quality — base variant with ViT-B encoder.
Exported in HQ-token-only single-mask mode.

## Source

- Repository: https://github.com/SysCV/sam-hq
- Paper: [Segment Anything in High Quality](https://arxiv.org/abs/2306.01567) (NeurIPS 2023)
- License: Apache-2.0

## Architecture

ViT-B encoder + HQ-SAM prompt encoder and mask decoder with HQ token
refinement for improved boundary quality.

## ONNX Models

### Encoder (`encoder.onnx`)

| Property    | Value                                          |
|-------------|-------------------------------------------------|
| Input       | `image` — float32 [1, 3, 1024, 1024]           |
| Output 1    | `image_embeddings` — float32 [1, 256, 64, 64]  |
| Output 2    | `interm_embeddings` — float32 [4, 1, 64, 64, 768] |
| Resolution  | Fixed 1024x1024                                 |

### Decoder (`decoder.onnx`)

| Property    | Value                                          |
|-------------|-------------------------------------------------|
| Input 1     | `image_embeddings` — float32 [1, 256, 64, 64]  |
| Input 2     | `interm_embeddings` — float32 [4, 1, 64, 64, 768] |
| Input 3     | `point_coords` — float32 [1, N, 2]             |
| Input 4     | `point_labels` — float32 [1, N]                |
| Input 5     | `mask_input` — float32 [1, 1, 256, 256]        |
| Input 6     | `has_mask_input` — float32 [1]                  |
| Output 1    | `masks` — float32 [1, 1, 1024, 1024]            |
| Output 2    | `iou_predictions` — float32 [1, 1]             |
| Output 3    | `low_res_masks` — float32 [1, 1, 256, 256]     |

## Notes

- Exported with `--hq-token-only` (single HQ mask output, no multi-mask).
- Output masks are always 1024x1024 (resize to original image size at runtime).
- Point labels: 0 = background, 1 = foreground, 2 = top-left box corner, 3 = bottom-right box corner.
- 4 intermediate embeddings from ViT-B global attention blocks (dim 768).

## Selection Criteria

| Property                 | Value                                                                                       |
|--------------------------|---------------------------------------------------------------------------------------------|
| Model license            | Apache-2.0                                                                                  |
| OSAID v1.0               | Open Weights                                                                                |
| MOF                      | Class II (Open Tooling)                                                                     |
| Training data license    | SA-1B: custom Meta research-only; HQ-Seg44K: mixed                                         |
| Training data provenance | [SA-1B](https://ai.meta.com/datasets/segment-anything/) (11M stock images) + HQ-Seg44K (44K images, 6 datasets) |
| Training code            | [Apache-2.0](https://github.com/SysCV/sam-hq)                                              |
| Known limitations        | SA-1B: unnamed stock provider, research-only license (not OSI), prohibits commercial use/redistribution. HQ-Seg44K: ThinObject-5K (CC BY-NC 4.0), DUT-OMRON (all rights reserved), FSS-1000/ECSSD (no license) |
| Published research       | [Segment Anything in High Quality](https://arxiv.org/abs/2306.01567) (NeurIPS 2023)        |
| Inference                | Local only, no cloud dependencies                                                           |
| Scope                    | Object segmentation                                                                         |
| Reproducibility          | Full pipeline (setup, convert, clean, demo)                                                 |
