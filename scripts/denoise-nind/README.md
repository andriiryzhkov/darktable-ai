# NIND UNet Denoiser

Image denoiser trained on the Natural Image Noise Dataset (NIND) from
Wikimedia Commons.

## Source

- Repository: https://github.com/trougnouf/nind-denoise
- Paper: [Natural Image Noise Dataset](https://arxiv.org/abs/1906.00270) (CVPR Workshops 2019)
- Dataset: https://commons.wikimedia.org/wiki/Natural_Image_Noise_Dataset
- License: GPL-3.0

## Architecture

Standard U-Net encoder-decoder: 3 → 64 → 128 → 256 → 512 → 512 (bottleneck),
with skip connections and sigmoid output activation.

## ONNX Model

| Property    | Value                                     |
|-------------|-------------------------------------------|
| File        | `model.onnx`                              |
| Input       | `input` — float32 [1, 3, H, W]           |
| Output      | `output` — float32 [1, 3, H, W]          |
| Resolution  | Dynamic (any H, W)                        |
| Normalize   | [0, 1] range (divide by 255)              |
| Tiling      | Yes                                       |

## Notes

- No ImageNet normalization — input is simply RGB in [0, 1] range.
- Output is clamped to [0, 1] by the sigmoid activation.
- Exported with FP16 precision.
