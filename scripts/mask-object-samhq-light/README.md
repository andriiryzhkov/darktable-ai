# HQ-SAM Light (ViT-Tiny)

Segment Anything in High Quality — lightweight variant with TinyViT encoder.
Exported in HQ-token-only single-mask mode.

## Source

- Repository: https://github.com/SysCV/sam-hq
- Paper: [Segment Anything in High Quality](https://arxiv.org/abs/2306.01567) (NeurIPS 2023)
- License: Apache-2.0

## Architecture

TinyViT encoder + HQ-SAM prompt encoder and mask decoder with HQ token
refinement for improved boundary quality.

## ONNX Models

### Encoder (`encoder.onnx`)

| Property    | Value                                          |
|-------------|-------------------------------------------------|
| Input       | `image` — float32 [1, 3, 1024, 1024]           |
| Output 1    | `image_embeddings` — float32 [1, 256, 64, 64]  |
| Output 2    | `interm_embeddings` — float32 [1, 1, 64, 64, 160] |
| Resolution  | Fixed 1024x1024                                 |

### Decoder (`decoder.onnx`)

| Property    | Value                                          |
|-------------|-------------------------------------------------|
| Input 1     | `image_embeddings` — float32 [1, 256, 64, 64]  |
| Input 2     | `interm_embeddings` — float32 [1, 1, 64, 64, 160] |
| Input 3     | `point_coords` — float32 [1, N, 2]             |
| Input 4     | `point_labels` — float32 [1, N]                |
| Input 5     | `mask_input` — float32 [1, 1, 256, 256]        |
| Input 6     | `has_mask_input` — float32 [1]                  |
| Input 7     | `orig_im_size` — float32 [1, 2]                |
| Output 1    | `masks` — float32 [1, 1, H, W]                 |
| Output 2    | `iou_predictions` — float32 [1, 1]             |
| Output 3    | `low_res_masks` — float32 [1, 1, 256, 256]     |

## Notes

- Exported with `--hq-token-only` (single HQ mask output, no multi-mask).
- Output mask resolution matches `orig_im_size`.
- Point labels: 0 = background, 1 = foreground, 2 = top-left box corner, 3 = bottom-right box corner.
- 1 intermediate embedding (TinyViT has 1 global attention stage vs 4 for standard ViT).
