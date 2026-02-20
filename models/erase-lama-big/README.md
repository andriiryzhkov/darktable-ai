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

## Selection Criteria

| Property                 | Value                                                                                                                 |
|--------------------------|-----------------------------------------------------------------------------------------------------------------------|
| Model license            | Apache-2.0                                                                                                            |
| OSAID v1.0               | Open Weights                                                                                                          |
| MOF                      | Class II (Open Tooling)                                                                                               |
| Training data license    | Custom non-commercial (MIT CSAIL terms)                                                                               |
| Training data provenance | [Places365](http://places2.csail.mit.edu/) (~1.8M scene images, MIT CSAIL, sourced from internet)                     |
| Training code            | [Apache-2.0](https://github.com/advimman/lama)                                                                       |
| Known limitations        | Places365: non-commercial research/education terms (not OSI), prohibits redistribution. Individual image copyrights belong to original uploaders, not auditable |
| Published research       | [Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://arxiv.org/abs/2109.07161) (WACV 2022)    |
| Inference                | Local only, no cloud dependencies                                                                                     |
| Scope                    | Object erasure (inpainting)                                                                                           |
| Reproducibility          | Pre-converted ONNX; full demo pipeline provided                                                                       |
