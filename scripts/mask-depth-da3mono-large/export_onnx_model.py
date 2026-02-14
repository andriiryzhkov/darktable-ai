# Export Depth Anything 3 Mono-Large to ONNX format for depth-based masking.
#
# Uses the DA3Mono monocular depth estimation model from:
# https://github.com/ByteDance-Seed/Depth-Anything-3
#
# The model predicts relative depth from a single image using a DINOv2-Large
# encoder + DualDPT decoder.  Output dimension is 2 (depth + confidence);
# only the depth channel (index 0) is exported.
#
# Input:  RGB image [1, 3, H, W]  (float32, ImageNet-normalized)
# Output: depth map [1, 1, H, W]  (float32, relative depth)
#
# H and W must be divisible by 14 (ViT patch size).
# Dynamic spatial dimensions are enabled so any valid resolution works at
# runtime; the tracing resolution (--height/--width) is used only for the
# dummy input during export.
#
# Based on DA3Mono-Large-ONNX-Conversion-Guide.md and
# ika-rwth-aachen/ros2-depth-anything-v3-trt/onnx/export.py

import argparse
import os
import sys
import types
import warnings

# Stub out heavy dependencies that the DA3 import chain pulls in but are not
# needed for ONNX export.  By pre-registering stubs in sys.modules, Python
# skips the real imports of moviepy, open3d, plyfile, evo, etc.
_stub_packages = [
    # third-party packages
    "moviepy", "moviepy.editor", "open3d", "trimesh", "plyfile",
    "e3nn", "pycolmap", "cv2", "imageio", "scipy",
    "scipy.interpolate", "gsplat", "evo", "evo.core", "evo.core.trajectory",
    # DA3 submodules that transitively import the heavy deps
    "depth_anything_3.utils.export",
    "depth_anything_3.utils.export.gs",
    "depth_anything_3.utils.export.video",
    "depth_anything_3.utils.gsply_helpers",
    "depth_anything_3.utils.pose_align",
    "depth_anything_3.model.utils.gs_renderer",
]
for _mod in _stub_packages:
    if _mod not in sys.modules:
        stub = types.ModuleType(_mod)
        if _mod == "depth_anything_3.utils.export":
            stub.export = lambda *a, **kw: None
        if _mod == "depth_anything_3.utils.pose_align":
            stub.align_poses_umeyama = lambda *a, **kw: None
        sys.modules[_mod] = stub

import torch
import torch.nn as nn

try:
    import onnxruntime  # type: ignore

    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False

try:
    import onnxsim

    onnxsim_exists = True
except ImportError:
    onnxsim_exists = False

parser = argparse.ArgumentParser(
    description="Export DA3Mono-Large to ONNX format."
)
parser.add_argument(
    "--model-id", type=str, default="depth-anything/DA3MONO-LARGE",
    help="HuggingFace model ID or local path (default: depth-anything/DA3MONO-LARGE).",
)
parser.add_argument(
    "--output", type=str, required=True,
    help="Output ONNX file path.",
)
parser.add_argument(
    "--height", type=int, default=504,
    help="Input height for tracing, must be divisible by 14 (default: 504).",
)
parser.add_argument(
    "--width", type=int, default=504,
    help="Input width for tracing, must be divisible by 14 (default: 504).",
)
parser.add_argument(
    "--opset", type=int, default=17,
    help="ONNX opset version (default: 17).",
)


class DA3MonoWrapper(nn.Module):
    """Wrapper for DA3Mono ONNX export.

    Strips away multi-view, pose estimation, and API-level preprocessing.
    Extracts only the depth channel from the 2-channel output
    (channel 0 = depth, channel 1 = confidence).

    Input:  [1, 3, H, W] float32 (ImageNet-normalized RGB)
    Output: [1, 1, H, W] float32 (relative depth)
    """

    def __init__(self, model):
        super().__init__()
        self.net = model.model  # underlying DepthAnything3Net

    def forward(self, image):
        # Model expects (B, S, C, H, W) with S=views; add single-view dim
        x = image.unsqueeze(1)
        output = self.net(
            x,
            extrinsics=None,
            intrinsics=None,
            export_feat_layers=[],
            infer_gs=False,
            use_ray_pose=False,
            ref_view_strategy="saddle_balanced",
        )
        # output is addict.Dict; output.depth shape: (B, 1, H, W)
        return output.depth


def to_numpy(tensor):
    return tensor.cpu().numpy()


def simplify_model(model_path):
    """Simplify ONNX model graph if onnxsim is available."""
    if not onnxsim_exists:
        print("Warning: onnx-simplifier not installed, skipping.")
        return

    import onnx

    print("Simplifying ONNX graph...")
    onnx_model = onnx.load(model_path)
    model_simp, check = onnxsim.simplify(onnx_model)
    if check:
        onnx.save(model_simp, model_path)
        print("Graph simplification passed.")
    else:
        print("Warning: simplified model failed validation, keeping original.")


def verify_model(model_path, dummy_inputs, name):
    """Verify exported ONNX model with ONNXRuntime."""
    if not onnxruntime_exists:
        return

    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(model_path, providers=providers)
    _ = ort_session.run(None, dummy_inputs)
    print(f"{name} has successfully been run with ONNXRuntime.")


if __name__ == "__main__":
    args = parser.parse_args()

    assert args.height % 14 == 0, f"Height must be divisible by 14, got {args.height}"
    assert args.width % 14 == 0, f"Width must be divisible by 14, got {args.width}"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Patch bfloat16 -> float16 for ONNX compatibility.
    # DA3 defaults to bfloat16 autocast which ONNX does not support.
    import depth_anything_3.api as api

    original_init = api.DepthAnything3.__init__

    def patched_init(self, *a, **kw):
        original_init(self, *a, **kw)
        self._autocast_dtype = torch.float16

    api.DepthAnything3.__init__ = patched_init

    # Load model (downloads from HuggingFace Hub if not cached)
    from depth_anything_3.api import DepthAnything3

    print(f"Loading model ({args.model_id})...")
    model = DepthAnything3.from_pretrained(args.model_id)
    model.eval()

    wrapper = DA3MonoWrapper(model)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, args.height, args.width, dtype=torch.float32)

    # Verify forward pass works
    with torch.no_grad():
        _ = wrapper(dummy_input)

    print(f"Exporting to {args.output}...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy_input,
                args.output,
                export_params=True,
                verbose=False,
                opset_version=args.opset,
                do_constant_folding=True,
                input_names=["image"],
                output_names=["depth"],
                dynamic_axes={
                    "image": {2: "height", 3: "width"},
                    "depth": {2: "height", 3: "width"},
                },
                dynamo=False,
            )

    simplify_model(args.output)
    verify_model(args.output, {"image": to_numpy(dummy_input)}, "DA3Mono-Large")

    print("Done!")
