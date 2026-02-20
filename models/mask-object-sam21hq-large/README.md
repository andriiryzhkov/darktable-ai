# HQ-SAM 2.1 (Hiera Large)

Segment Anything Model 2.1 with HQ token refinement — large variant with
Hiera encoder. Multi-mask output mode (3 SAM masks + 1 HQ mask).

## Source

- Repository: https://github.com/SysCV/sam-hq
- Paper: [Segment Anything in High Quality](https://arxiv.org/abs/2306.01567) (NeurIPS 2023)
- License: Apache-2.0

## Architecture

Hiera Large encoder with SAM 2.1 HQ mask decoder. The HQ decoder adds a
refinement token that produces a high-quality mask with better boundary
precision than the standard SAM masks.

## ONNX Models

### Encoder (`encoder.onnx`)

| Property    | Value                                          |
|-------------|-------------------------------------------------|
| Input       | `image` — float32 [1, 3, 1024, 1024]           |
| Output 1    | `high_res_feats_0` — float32 [1, 32, 256, 256] |
| Output 2    | `high_res_feats_1` — float32 [1, 64, 128, 128] |
| Output 3    | `image_embed` — float32 [1, 256, 64, 64]       |
| Resolution  | Fixed 1024x1024                                 |

### Decoder (`decoder.onnx`)

| Property    | Value                                          |
|-------------|-------------------------------------------------|
| Input 1     | `image_embed` — float32 [1, 256, 64, 64]       |
| Input 2     | `high_res_feats_0` — float32 [1, 32, 256, 256] |
| Input 3     | `high_res_feats_1` — float32 [1, 64, 128, 128] |
| Input 4     | `point_coords` — float32 [1, N, 2]             |
| Input 5     | `point_labels` — float32 [1, N]                |
| Input 6     | `mask_input` — float32 [1, 1, 256, 256]        |
| Input 7     | `has_mask_input` — float32 [1]                  |
| Output 1    | `masks` — float32 [1, 4, 1024, 1024]           |
| Output 2    | `iou_predictions` — float32 [1, 3]             |
| Output 3    | `low_res_masks` — float32 [1, 4, 256, 256]     |

## Notes

- Outputs 4 masks: indices 0-2 are standard SAM multi-masks, index 3 is the HQ refinement mask.
- IoU predictions cover only the 3 SAM masks (no IoU for HQ mask).
- The HQ mask (index 3) typically has the best boundary quality.
- Output masks are always 1024x1024 (resize to original image size at runtime).
- `low_res_masks` (256x256) can be fed back as `mask_input` for iterative refinement.

## Selection Criteria

| Property                 | Value                                                                                       |
|--------------------------|---------------------------------------------------------------------------------------------|
| Model license            | Apache-2.0                                                                                  |
| OSAID v1.0               | Open Weights                                                                                |
| MOF                      | Class II (Open Tooling)                                                                     |
| Training data license    | SA-V: CC BY 4.0; SA-1B: custom Meta research-only; HQ-Seg44K: mixed                        |
| Training data provenance | [SA-V](https://ai.meta.com/datasets/segment-anything-video/) + [SA-1B](https://ai.meta.com/datasets/segment-anything/) (SAM 2.1 base) + HQ-Seg44K (44K images, HQ tuning) |
| Training code            | [Apache-2.0](https://github.com/SysCV/sam-hq)                                              |
| Known limitations        | SA-1B: unnamed stock provider, research-only license (not OSI). HQ-Seg44K: ThinObject-5K (CC BY-NC 4.0), DUT-OMRON (all rights reserved), FSS-1000/ECSSD (no license) |
| Published research       | [Segment Anything in High Quality](https://arxiv.org/abs/2306.01567) (NeurIPS 2023)        |
| Inference                | Local only, no cloud dependencies                                                           |
| Scope                    | Object segmentation                                                                         |
| Reproducibility          | Full pipeline (setup, convert, clean, demo)                                                 |
