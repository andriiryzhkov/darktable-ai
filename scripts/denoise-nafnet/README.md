# NAFNet SIDD Width-32

Image denoiser trained on the SIDD (Smartphone Image Denoising Dataset).
Lightweight variant with width=32.

## Source

- Repository: https://github.com/megvii-research/NAFNet
- Paper: [Simple Baselines for Image Restoration](https://arxiv.org/abs/2204.04676) (ECCV 2022)
- License: MIT

## Architecture

NAFNet (Nonlinear Activation Free Network) — encoder-decoder with 4 stages,
channel widths [32, 64, 128, 256], 12 middle blocks.

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

- Input and output are both RGB images in [0, 1] range.
- Output should be clipped to [0, 1] before converting back to uint8.
- Exported with FP16 precision.

## Selection criteria compliance

- **License:** MIT -- permissive, GPL-3.0 compatible.
- **Open weights:** Publicly available on Google Drive, no registration required.
- **Published research:** Peer-reviewed at ECCV 2022.
- **Training data:** SIDD (Smartphone Image Denoising Dataset) -- academic dataset of real smartphone noisy/clean image pairs from 10 scenes captured with 5 smartphones. Publicly documented and available.
- **Purpose:** Image denoising - photo editing task. Runs locally, no external services.
