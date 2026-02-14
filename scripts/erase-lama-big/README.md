# LaMa Big (Large Mask Inpainting)

Image inpainting model for object erasure - large variant. Pre-converted
ONNX model from HuggingFace (no local conversion needed).

## Source

- Repository: <https://github.com/advimman/lama>
- ONNX source: <https://huggingface.co/Carve/LaMa-ONNX>
- Paper: [Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161) (WACV 2022)
- License: Apache-2.0

## Architecture

Fourier convolution-based encoder-decoder with fast Fourier convolution
(FFC) layers that provide a global receptive field. The "big" variant is
trained with a larger training schedule on Places365.

## ONNX Model

| Property    | Value                                     |
|-------------|-------------------------------------------|
| File        | `model.onnx`                              |
| Input 1     | `image` -- float32 [1, 3, 512, 512]      |
| Input 2     | `mask` -- float32 [1, 1, 512, 512]       |
| Output      | `output` -- float32 [1, 3, 512, 512]     |
| Resolution  | 512x512 (pad to multiple of 8)            |
| Normalize   | [0, 1] range (divide by 255)              |
| Tiling      | No                                        |

## Notes

- Input image is RGB in [0, 1] range. No ImageNet normalization.
- Mask is binary: 1 = area to inpaint, 0 = keep.
- Output is the inpainted image in [0, 1] range.
- Fixed 512x512 resolution. Resize input and mask before inference, resize output back to original size.
- Pre-converted ONNX from HuggingFace. Full precision (float32).

## Selection criteria compliance

- **License:** Apache-2.0 -- permissive, GPL-3.0 compatible.
- **Open weights:** Publicly available on HuggingFace, no registration required.
- **Published research:** Peer-reviewed at WACV 2022.
- **Training data:** Places365 dataset (scene recognition dataset with ~1.8M images). Publicly available and well-documented.
- **Purpose:** Image inpainting (object erasure) - photo editing task. Runs locally, no external services.
