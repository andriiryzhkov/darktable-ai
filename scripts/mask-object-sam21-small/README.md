# SAM 2.1 (Hiera Small)

Segment Anything Model 2.1 — small variant with Hiera encoder.
Multi-mask output mode (3 masks per prompt).

## Source

- Repository: https://github.com/facebookresearch/sam2
- Paper: [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) (2024)
- License: Apache-2.0

## Architecture

Hiera Small encoder with SAM 2 mask decoder and high-resolution feature
projections (conv_s0, conv_s1).

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
| Input 4     | `point_coords` — float32 [B, N, 2]             |
| Input 5     | `point_labels` — float32 [B, N]                |
| Input 6     | `mask_input` — float32 [B, 1, 256, 256]        |
| Input 7     | `has_mask_input` — float32 [1]                  |
| Output 1    | `masks` — float32 [B, 3, 1024, 1024]           |
| Output 2    | `iou_predictions` — float32 [B, 3]             |
| Output 3    | `low_res_masks` — float32 [B, 3, 256, 256]     |

## Notes

- Multi-mask output: 3 candidate masks per prompt, select by highest IoU score.
- Output masks are always 1024x1024 (resize to original image size at runtime).
- `low_res_masks` (256x256) can be fed back as `mask_input` for iterative refinement.
