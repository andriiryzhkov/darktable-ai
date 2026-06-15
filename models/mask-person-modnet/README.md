# MODNet

Trimap-free portrait matting. Produces a continuous 0–1 **alpha matte** for
people in the frame – hair, fly-aways and soft edges come out as transparency
rather than a hard cut. Specialised for portraits; complements the generic
subject-extraction BiRefNet variants and the click-based SAM 2.1 / SegNext
models in this repo.

## Source

- Repository: <https://github.com/ZHKKKe/MODNet>
- Paper: [MODNet: Real-Time Trimap-Free Portrait Matting via Objective Decomposition](https://arxiv.org/abs/2011.11961) (AAAI 2022)
- License: Apache-2.0
- ONNX weights: [Xenova/modnet](https://huggingface.co/Xenova/modnet) (Apache-2.0, derived from the official MODNet checkpoint)

## Architecture

MobileNetV2 backbone with three task-specific branches – semantic estimation,
detail prediction, semantic-detail fusion – decomposed under the MODNet
"objective decomposition" framework. Compact (~7 MiB image-matting checkpoint
upstream); the ONNX export here is ~13 MiB at FP16.

## ONNX Model

| Direction | Tensor | Shape                         | Type                          |
| --------- | ------ | ----------------------------- | ----------------------------- |
| in        | input  | 1 x 3 x H x W (dynamic)       | float16 (default) / float32   |
| out       | output | 1 x 1 x H x W (dynamic)       | float16 (default) / float32   |

Default ship is FP16 (~13 MiB). The FP32 variant (~26 MiB) lives at the same
HF path under `onnx/model.onnx`; swap the URL in `model.yaml` to fetch it
instead.

### Preprocessing (client-side)

- match the upstream MODNet pipeline: resize so the longer side is 512 px
  while preserving aspect ratio, then round each dimension up to the next
  multiple of 32
- normalize RGB to `[-1, 1]` via `(rgb - 127.5) / 127.5`
- transpose to NCHW and cast to the model's expected dtype

### Postprocessing

- the output is already a `[0, 1]` alpha matte – no sigmoid, no threshold
- resize the matte back to original image dimensions

## Selection Criteria

| Property                 | Value                                                                                              |
| ------------------------ | -------------------------------------------------------------------------------------------------- |
| Model license            | Apache-2.0                                                                                         |
| OSAID v1.0               | Open Source AI                                                                                     |
| MOF                      | Class I (Open Science)                                                                             |
| Training data license    | PPM-100: Apache-2.0 alongside the codebase. Adobe Composition-1k: research-only, not redistributable. AISegment portrait dataset: provenance not fully documented |
| Training data provenance | [PPM-100](https://github.com/ZHKKKe/PPM): 100 high-resolution portraits collected and annotated by the MODNet authors. Adobe Composition-1k and AISegment supply additional supervised pre-training data |
| Training code            | [Apache-2.0](https://github.com/ZHKKKe/MODNet)                                                     |
| Known limitations        | Adobe Composition-1k training portion uses research-only data; AISegment provenance is not fully documented. Worth flagging under the strictest reading of the AI policy. The model is portrait-specialised – tends to under-segment groups, partial occlusion, and non-human subjects |
| Published research       | [MODNet](https://arxiv.org/abs/2011.11961) (AAAI 2022)                                             |
| Inference                | Local only, no cloud dependencies                                                                  |
| Scope                    | Automatic portrait / person matting; no clicks required; alpha matte output (not binary mask)      |
| Reproducibility          | Full pipeline – ONNX is downloaded from the Xenova mirror of the official MODNet checkpoint        |
