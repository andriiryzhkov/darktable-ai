# MAN-Base

Multi-scale Attention Network, "Base" variant — a mid-weight pure-CNN
super-resolution architecture combining multi-scale large kernel attention
(MLKA) with a gated spatial attention unit (GSAU). Heavier than RealPLKSR,
lighter than DAT-2; quality typically sits in between.

Includes 2x and 4x upscaling variants.

## Source

- Repository: [icandle/MAN](https://github.com/icandle/MAN) (Apache-2.0, vendored at `vendor/man`)
- Architecture: [`vendor/man/archs/MAN_arch.py`](../../vendor/man/archs/MAN_arch.py)
- Paper: [Multi-scale Attention Network for Single Image Super-Resolution](https://arxiv.org/abs/2209.14145)
- Checkpoints: [Google Drive folder](https://drive.google.com/drive/folders/1sARYFkVeTIFVCa2EnZg9TjZvirDvUNOL) — `MAN-base.zip` (185 MB) bundles the MAN-Base weights. Setup downloads and unpacks automatically.

## Architecture

MAN-Base interleaves a sequence of Multi-scale Attention Blocks (MABs) where
each block applies:

- **Multi-scale Large Kernel Attention (MLKA)** — depth-wise large-kernel
  conv decomposed across three scales (7×7 / 9×9 / 13×13 effective with
  dilation) to balance receptive field against parameter cost.
- **Gated Spatial Attention Unit (GSAU)** — lightweight spatial gating that
  modulates feature maps without a full attention computation.

MAN is not supported by Spandrel, so the architecture is imported directly
from the vendored repo at `vendor/man/archs/MAN_arch.py`. The conversion
script provides a minimal stub for `basicsr.utils.registry` so the model
loads without pulling in the BasicSR training framework.

| Property     | Value                                 |
|--------------|---------------------------------------|
| Architecture | MAN-Base                              |
| n_resblocks  | 36                                    |
| n_feats      | 180                                   |
| Parameters   | ~8.7M                                 |
| Receptive    | Large (7×7 / 9×9 / 13×13 multi-scale) |
| Upsampler    | PixelShuffle                          |

## ONNX Models

| Property   | model_x2.onnx                         | model_x4.onnx                         |
|------------|---------------------------------------|---------------------------------------|
| Input      | `input` — float32 [1, 3, 512, 512]    | `input` — float32 [1, 3, 256, 256]    |
| Output     | `output` — float32 [1, 3, 1024, 1024] | `output` — float32 [1, 3, 1024, 1024] |
| Resolution | Static, baked at 512×512              | Static, baked at 256×256              |
| Opset      | 20                                    | 20                                    |
| Precision  | FP32                                  | FP32                                  |
| Normalize  | [0, 1] range (divide by 255)          | [0, 1] range (divide by 255)          |
| Tiling     | Yes (`model_x2.input_sizes: [512]`)   | Yes (`model_x4.input_sizes: [256]`)   |

Per-stem tile sizes are declared in the manifest so darktable picks the
right size for each variant at runtime:

```yaml
attributes:
  model_x2:
    input_sizes: [512]
  model_x4:
    input_sizes: [256]
```

## Notes

- Input and output are RGB images in [0, 1] range; output should be
  clipped to [0, 1] before converting back to uint8.
- Exported with FP32 precision. FP16 export is supported via `--fp16` in
  convert args but off by default.
- Inputs are baked into the graph so JIT-compiling EPs (CoreML,
  MIGraphX) only pay the compile cost once. Callers must tile at
  exactly the declared size.

## Selection Criteria

| Property                 | Value                                                                                              |
|--------------------------|----------------------------------------------------------------------------------------------------|
| Model license            | Apache-2.0                                                                                         |
| OSAID v1.0               | Open Source AI                                                                                     |
| MOF                      | Class II (Open Tooling)                                                                            |
| Training data license    | DIV2K (CC0), Flickr2K — standard SR research datasets                                              |
| Training data provenance | Public image restoration benchmarks (DF2K)                                                         |
| Training code            | [Apache-2.0](https://github.com/icandle/MAN)                                                       |
| Known limitations        | Training dataset Flickr2K does not carry an explicit open-source license                           |
| Published research       | [Multi-scale Attention Network for Image Super-Resolution](https://arxiv.org/abs/2209.14145)       |
| Inference                | Local only, no cloud dependencies                                                                  |
| Scope                    | Image upscaling (2x and 4x super-resolution)                                                       |
| Reproducibility          | Full pipeline (setup, convert, clean, demo)                                                        |
