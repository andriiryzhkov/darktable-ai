# DAT-2

Dual Aggregation Transformer (DAT) for image super-resolution. DAT-2
is the rectangular-window variant of the architecture, balancing fidelity
and parameter count (~11.2M).

Includes both 2x and 4x upscaling variants.

## Source

- Repository: [zhengchen1999/DAT](https://github.com/zhengchen1999/DAT)
- Paper: [Dual Aggregation Transformer for Image Super-Resolution](https://arxiv.org/abs/2308.03364) (ICCV 2023)
- License: Apache-2.0
- Checkpoints: mirrored on Hugging Face at [`silveroxides/DAT_upscale_models`](https://huggingface.co/silveroxides/DAT_upscale_models/tree/main/DAT-2) (Apache-2.0)

## Architecture

DAT applies spatial and channel self-attention in consecutive transformer
blocks, with an adaptive interaction module (AIM) coordinating them and
a spatial-gate feed-forward network (SGFN) replacing the standard FFN.
DAT-2 specifically uses *rectangular* attention windows with
`split_size=[8, 32]` — meaning inputs must be a multiple of 32 in both
dimensions.

| Property         | Value              |
|------------------|--------------------|
| embed_dim        | 180                |
| depth            | [6, 6, 6, 6, 6, 6] |
| num_heads        | [6, 6, 6, 6, 6, 6] |
| split_size       | [8, 32]            |
| expansion_factor | 2                  |
| parameters       | ~11.21M            |

Reported quality (paper, Urban100 ×4): 27.86 dB PSNR / 0.8341 SSIM.

## ONNX Models

| Property   | model_x2.onnx                       | model_x4.onnx                       |
|------------|-------------------------------------|-------------------------------------|
| Input      | `input` — float32 [1, 3, 256, 256]  | `input` — float32 [1, 3, 128, 128]  |
| Output     | `output` — float32 [1, 3, 512, 512] | `output` — float32 [1, 3, 512, 512] |
| Resolution | Static, baked at 256×256            | Static, baked at 128×128            |
| Opset      | 20                                  | 20                                  |
| Normalize  | [0, 1] range (divide by 255)        | [0, 1] range (divide by 255)        |
| Tiling     | Yes (`model_x2.input_sizes: [256]`) | Yes (`model_x4.input_sizes: [128]`) |

Both variants produce a 512×512 output tile — x2 from a 256×256 input,
x4 from a 128×128 input. Per-stem tile sizes are declared in the manifest
so darktable picks the right size for each variant at runtime:

```yaml
attributes:
  model_x2:
    input_sizes: [256]
  model_x4:
    input_sizes: [128]
```

Tile sizes are kept smaller than BSRGAN's (which uses 512/256) because
DAT-2's window-attention trace captures the full graph at the deployment
dim and OOMs on consumer-class memory at 512×512 input. The smaller
tiles cost more inference calls per image but trace cleanly and still
emit a useful 512×512 output per call.

## Notes

- Input and output are RGB images in [0, 1] range.
- Output should be clipped to [0, 1] before converting back to uint8.
- Exported with FP16 precision (halves file size and accelerates inference
  on EPs with native FP16 support; quality difference is negligible for SR).
  Override with `--fp16=false` in convert args if FP32 is needed.
- Inputs are baked into the graph so JIT-compiling EPs (CoreML,
  MIGraphX) only pay the compile cost once. Callers must tile at
  exactly the declared size — and at a multiple of 32 to satisfy the
  rectangular-window attention.
- Conversion uses [Spandrel](https://github.com/chaiNNer-org/spandrel)
  to auto-detect the DAT-2 variant from the checkpoint's state_dict,
  avoiding the need to clone the official BasicSR-derived training
  scaffolding.

## Selection Criteria

| Property                 | Value                                                                                              |
|--------------------------|----------------------------------------------------------------------------------------------------|
| Model license            | Apache-2.0                                                                                         |
| OSAID v1.0               | Open Source AI                                                                                     |
| MOF                      | Class II (Open Tooling)                                                                            |
| Training data license    | DIV2K (CC0), Flickr2K — standard SR research datasets                                              |
| Training data provenance | Public image restoration benchmarks (DF2K)                                                         |
| Training code            | [Apache-2.0](https://github.com/zhengchen1999/DAT)                                                 |
| Known limitations        | Training dataset Flickr2K does not carry an explicit open-source license                           |
| Published research       | [Dual Aggregation Transformer for Image Super-Resolution](https://arxiv.org/abs/2308.03364)        |
| Inference                | Local only, no cloud dependencies                                                                  |
| Scope                    | Image upscaling (2x and 4x super-resolution)                                                       |
| Reproducibility          | Full pipeline (setup, convert, clean, demo)                                                        |
