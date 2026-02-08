# Export SAM-HQ image encoder and decoder (prompt encoder + mask decoder) to
# separate ONNX files:  encoder.onnx  and  decoder.onnx.
#
# Based on sam-hq/scripts/export_onnx_model.py with fixes for Light HQ-SAM
# (vit_tiny).
#
# The original script fails for vit_tiny because:
# 1. encoder_embed_dim_dict has no "vit_tiny" entry -> KeyError
# 2. interm_embeddings dummy shape assumes 4 intermediate embeddings (standard ViT),
#    but TinyViT only produces 1.

import argparse
import os
import warnings

import torch
import torch.nn as nn

from segment_anything import sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel

try:
    import onnxruntime  # type: ignore

    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False

# Per-model configuration:
#   encoder_embed_dim: dimension of intermediate ViT features passed to compress_vit_feat
#     - Standard ViT: same as the encoder embed_dim
#     - TinyViT: 160 (output of stage 1 after PatchMerging, matches vit_dim in MaskDecoderHQ)
#   num_interm: number of intermediate embeddings the image encoder produces
#     - Standard ViT: 4 (one per global-attention block)
#     - TinyViT: 1 (captured after layer 1 only)
MODEL_CONFIG = {
    "vit_b":    {"encoder_embed_dim": 768,  "num_interm": 4},
    "vit_l":    {"encoder_embed_dim": 1024, "num_interm": 4},
    "vit_h":    {"encoder_embed_dim": 1280, "num_interm": 4},
    "vit_tiny": {"encoder_embed_dim": 160,  "num_interm": 1},
}

parser = argparse.ArgumentParser(
    description="Export SAM-HQ encoder and decoder to separate ONNX files."
)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to the SAM-HQ model checkpoint (.pth).",
)
parser.add_argument(
    "--output-dir",
    type=str,
    required=True,
    help="Output directory for encoder.onnx and decoder.onnx.",
)
parser.add_argument(
    "--model-type",
    type=str,
    default="vit_tiny",
    choices=list(MODEL_CONFIG.keys()),
    help="Model variant to export (default: vit_tiny).",
)
parser.add_argument(
    "--hq-token-only",
    action="store_true",
    help=(
        "Use HQ output only (best for single-object images). "
        "When False (default), HQ output corrects the SAM output "
        "(better for multi-object images like COCO)."
    ),
)
parser.add_argument(
    "--multimask-output",
    action="store_true",
    help="Use multi-mask output mode and select the best mask.",
)
parser.add_argument(
    "--opset",
    type=int,
    default=17,
    help="ONNX opset version (must be >= 11, default: 17).",
)
parser.add_argument(
    "--quantize-out",
    type=str,
    default=None,
    help="If set, quantize the decoder model (uint8) and save to this path.",
)
parser.add_argument(
    "--gelu-approximate",
    action="store_true",
    help="Replace GELU with tanh approximation (useful for runtimes with slow erf).",
)
parser.add_argument(
    "--use-stability-score",
    action="store_true",
    help="Replace predicted mask quality score with stability score.",
)
parser.add_argument(
    "--return-extra-metrics",
    action="store_true",
    help="Return (masks, scores, stability_scores, areas, low_res_logits).",
)


class SamEncoderOnnxModel(nn.Module):
    """Wraps the SAM-HQ image encoder for ONNX export.

    Outputs image_embeddings and interm_embeddings stacked into a single
    tensor so they can be fed directly to the decoder ONNX model.
    """

    def __init__(self, sam_model):
        super().__init__()
        self.image_encoder = sam_model.image_encoder

    @torch.no_grad()
    def forward(self, image):
        image_embeddings, interm_embeddings = self.image_encoder(image)
        # Stack list of (B, H, W, C) tensors -> (num_interm, B, H, W, C)
        interm_embeddings = torch.stack(interm_embeddings, dim=0)
        return image_embeddings, interm_embeddings


def run_encoder_export(sam, model_type: str, output: str, opset: int):
    """Export the image encoder to ONNX."""
    encoder_model = SamEncoderOnnxModel(sam)
    encoder_model.eval()

    dummy_image = torch.randn(1, 3, 1024, 1024, dtype=torch.float)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        print(f"Exporting encoder to {output}...")
        torch.onnx.export(
            encoder_model,
            dummy_image,
            output,
            export_params=True,
            verbose=False,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["image_embeddings", "interm_embeddings"],
            dynamo=False,
        )

    if onnxruntime_exists:
        ort_inputs = {"image": dummy_image.cpu().numpy()}
        providers = ["CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(output, providers=providers)
        _ = ort_session.run(None, ort_inputs)
        print("Encoder has successfully been run with ONNXRuntime.")


def run_decoder_export(
    sam,
    model_type: str,
    output: str,
    opset: int,
    hq_token_only: bool = False,
    multimask_output: bool = False,
    gelu_approximate: bool = False,
    use_stability_score: bool = False,
    return_extra_metrics: bool = False,
):
    """Export the prompt encoder + mask decoder to ONNX."""
    onnx_model = SamOnnxModel(
        model=sam,
        hq_token_only=hq_token_only,
        multimask_output=multimask_output,
        use_stability_score=use_stability_score,
        return_extra_metrics=return_extra_metrics,
    )

    if gelu_approximate:
        for _, m in onnx_model.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    cfg = MODEL_CONFIG[model_type]
    encoder_embed_dim = cfg["encoder_embed_dim"]
    num_interm = cfg["num_interm"]

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]

    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "interm_embeddings": torch.randn(
            num_interm, 1, *embed_size, encoder_embed_dim, dtype=torch.float
        ),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }

    _ = onnx_model(**dummy_inputs)

    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(output, "wb") as f:
            print(f"Exporting decoder to {output}...")
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                dynamo=False,  # use legacy TorchScript exporter (new torch.export can't handle data-dependent slicing in mask_postprocessing)
            )

    if onnxruntime_exists:
        ort_inputs = {k: v.cpu().numpy() for k, v in dummy_inputs.items()}
        providers = ["CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(output, providers=providers)
        _ = ort_session.run(None, ort_inputs)
        print("Decoder has successfully been run with ONNXRuntime.")


if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    encoder_path = os.path.join(args.output_dir, "encoder.onnx")
    decoder_path = os.path.join(args.output_dir, "decoder.onnx")

    print(f"Loading model ({args.model_type})...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)

    run_encoder_export(
        sam,
        model_type=args.model_type,
        output=encoder_path,
        opset=args.opset,
    )

    run_decoder_export(
        sam,
        model_type=args.model_type,
        output=decoder_path,
        opset=args.opset,
        hq_token_only=args.hq_token_only,
        multimask_output=args.multimask_output,
        gelu_approximate=args.gelu_approximate,
        use_stability_score=args.use_stability_score,
        return_extra_metrics=args.return_extra_metrics,
    )

    if args.quantize_out is not None:
        assert onnxruntime_exists, "onnxruntime is required to quantize the model."
        from onnxruntime.quantization import QuantType  # type: ignore
        from onnxruntime.quantization.quantize import quantize_dynamic  # type: ignore

        print(f"Quantizing decoder and writing to {args.quantize_out}...")
        quantize_dynamic(
            model_input=decoder_path,
            model_output=args.quantize_out,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        print("Done!")