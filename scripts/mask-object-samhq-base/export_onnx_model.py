# Export SAM-HQ image encoder and decoder (prompt encoder + mask decoder) to
# separate ONNX files:  encoder.onnx  and  decoder.onnx.
#
# Based on sam-hq/scripts/export_onnx_model.py with encoder export added.

import argparse
import os
import warnings

import torch
import torch.nn as nn

from torch.nn import functional as F

from segment_anything import sam_model_registry

try:
    import onnxruntime  # type: ignore

    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False

# Per-model configuration:
#   encoder_embed_dim: dimension of intermediate ViT features
#   num_interm: number of intermediate embeddings the image encoder produces
MODEL_CONFIG = {
    "vit_b": {"encoder_embed_dim": 768,  "num_interm": 4},
    "vit_l": {"encoder_embed_dim": 1024, "num_interm": 4},
    "vit_h": {"encoder_embed_dim": 1280, "num_interm": 4},
}

parser = argparse.ArgumentParser(
    description="Export SAM-HQ encoder and decoder to separate ONNX files."
)

parser.add_argument(
    "--checkpoint", type=str, required=True,
    help="Path to the SAM-HQ model checkpoint (.pth).",
)

parser.add_argument(
    "--output-dir", type=str, required=True,
    help="Output directory for encoder.onnx and decoder.onnx.",
)

parser.add_argument(
    "--model-type", type=str, default="vit_b",
    choices=list(MODEL_CONFIG.keys()),
    help="Model variant to export (default: vit_b).",
)

parser.add_argument(
    "--hq-token-only", action="store_true",
    help=(
        "Use HQ output only (best for single-object images). "
        "When False (default), HQ output corrects the SAM output "
        "(better for multi-object images like COCO)."
    ),
)

parser.add_argument(
    "--multimask-output", action="store_true",
    help="Use multi-mask output mode and select the best mask.",
)

parser.add_argument(
    "--opset", type=int, default=17,
    help="ONNX opset version (must be >= 11, default: 17).",
)

parser.add_argument(
    "--quantize-out", type=str, default=None,
    help="If set, quantize the decoder model (uint8) and save to this path.",
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


def to_numpy(tensor):
    return tensor.cpu().numpy()


class SamHQDecoderOnnxModel(nn.Module):
    """Wraps SAM-HQ prompt encoder + mask decoder for ONNX export.

    Outputs fixed 1024x1024 masks (no orig_im_size input) for a unified
    decoder interface across all SAM variants.

    Inputs:
        image_embeddings:  (1, embed_dim, H, W)
        interm_embeddings: (num_interm, 1, H, W, encoder_embed_dim)
        point_coords:      (1, N, 2)
        point_labels:      (1, N)
        mask_input:        (1, 1, 4*H, 4*W)
        has_mask_input:    (1,)

    Outputs:
        masks:           (1, 1, 1024, 1024)
        iou_predictions: (1, 1)
        low_res_masks:   (1, 1, 256, 256)
    """

    def __init__(self, model, hq_token_only=False, multimask_output=False):
        super().__init__()
        self.mask_decoder = model.mask_decoder
        self.model = model
        self.img_size = model.image_encoder.img_size
        self.hq_token_only = hq_token_only
        self.multimask_output = multimask_output

    def _embed_points(self, point_coords, point_labels):
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.model.prompt_encoder.not_a_point_embed.weight * (
            point_labels == -1
        )

        for i in range(self.model.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.model.prompt_encoder.point_embeddings[
                i
            ].weight * (point_labels == i)

        return point_embedding

    def _embed_masks(self, input_mask, has_mask_input):
        mask_embedding = has_mask_input * self.model.prompt_encoder.mask_downscaling(input_mask)
        mask_embedding = mask_embedding + (
            1 - has_mask_input
        ) * self.model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return mask_embedding

    @torch.no_grad()
    def forward(
        self,
        image_embeddings,
        interm_embeddings,
        point_coords,
        point_labels,
        mask_input,
        has_mask_input,
    ):
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)

        # Compute HQ features from intermediate ViT embeddings
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2)
        hq_features = (
            self.mask_decoder.embedding_encoder(image_embeddings)
            + self.mask_decoder.compress_vit_feat(vit_features)
        )

        masks, scores = self.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            hq_features=hq_features,
        )

        # Select mask output
        if self.multimask_output:
            mask_slice = slice(1, self.mask_decoder.num_mask_tokens - 1)
            scores_multi = scores[:, mask_slice]
            scores_multi, max_iou_idx = torch.max(scores_multi, dim=1)
            scores = scores_multi.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[
                torch.arange(masks_multi.size(0)), max_iou_idx
            ].unsqueeze(1)
        else:
            scores = scores[:, :1]
            masks_sam = masks[:, :1]

        masks_hq = masks[
            :,
            self.mask_decoder.num_mask_tokens - 1 : self.mask_decoder.num_mask_tokens,
        ]

        if self.hq_token_only:
            masks = masks_hq
        else:
            masks = masks_sam + masks_hq

        low_res_masks = masks

        # Upscale to 1024x1024 (unified output resolution)
        masks = F.interpolate(
            masks, size=(1024, 1024), mode="bilinear", align_corners=False
        )

        return masks, scores, low_res_masks


def run_encoder_export(sam, model_type, output, opset):
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
        ort_inputs = {"image": to_numpy(dummy_image)}
        providers = ["CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(output, providers=providers)
        _ = ort_session.run(None, ort_inputs)
        print("Encoder has successfully been run with ONNXRuntime.")


def run_decoder_export(
    sam,
    model_type,
    output,
    opset,
    hq_token_only=False,
    multimask_output=False,
):
    """Export the prompt encoder + mask decoder to ONNX."""
    onnx_model = SamHQDecoderOnnxModel(
        model=sam,
        hq_token_only=hq_token_only,
        multimask_output=multimask_output,
    )

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
                dynamo=False,
            )

    if onnxruntime_exists:
        ort_inputs = {k: to_numpy(v) for k, v in dummy_inputs.items()}
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
