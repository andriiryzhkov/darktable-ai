# SAM 2.1 (Hiera Small) -- Pre-converted ONNX

Segment Anything Model 2.1 -- small variant with Hiera encoder.
Pre-converted ONNX model from HuggingFace (no local conversion needed).

## Source

- Repository: <https://github.com/facebookresearch/sam2>
- ONNX source: <https://huggingface.co/onnx-community/sam2.1-hiera-small-ONNX>
- Paper: [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) (2024)
- License: Apache-2.0

## Architecture

Hiera Small encoder with SAM 2 mask decoder and high-resolution feature
projections. Same architecture as `mask-object-sam21-small` but downloaded
as pre-converted ONNX.

## ONNX Models

### Encoder (`encoder.onnx`)

| Property   | Value                                             |
|------------|---------------------------------------------------|
| Input      | `pixel_values` -- float32 [B, 3, 1024, 1024]      |
| Output 1   | `image_embeddings.0` -- float32 [B, 32, 256, 256] |
| Output 2   | `image_embeddings.1` -- float32 [B, 64, 128, 128] |
| Output 3   | `image_embeddings.2` -- float32 [B, 256, 64, 64]  |
| Resolution | Fixed 1024x1024                                   |

### Decoder (`decoder.onnx`)

| Property | Value                                             |
|----------|---------------------------------------------------|
| Input 1  | `image_embeddings.0` -- float32 [B, 32, 256, 256] |
| Input 2  | `image_embeddings.1` -- float32 [B, 64, 128, 128] |
| Input 3  | `image_embeddings.2` -- float32 [B, 256, 64, 64]  |
| Input 4  | `input_points` -- float32 [B, 1, N, 2]            |
| Input 5  | `input_labels` -- int64 [B, 1, N]                 |
| Input 6  | `input_boxes` -- float32 [B, M, 4]                |
| Output 1 | `iou_scores` -- float32 [B, P, 3]                 |
| Output 2 | `pred_masks` -- float32 [B, P, 3, H, W]           |
| Output 3 | `object_score_logits` -- float32 [B, P, 1]        |

## Notes

- Pre-converted ONNX from HuggingFace, merged from external data format into single files.
- Tensor names differ from our custom export (`pixel_values` vs `image`, `input_points` vs `point_coords`, etc.).
- Decoder outputs low-res masks (upscale to original image size at runtime).
- Full precision (float32).
