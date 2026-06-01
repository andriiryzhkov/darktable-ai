"""Export MAN-Base to ONNX.

MAN is not supported by Spandrel, so we vendor the architecture
(`vendor/man/archs/MAN_arch.py`) and load the state dict directly.

The architecture file imports from `basicsr.utils.registry`; rather than
pull in the full BasicSR runtime (large, training-focused) we stub that
module with a no-op registry. The arch itself is plain PyTorch and
doesn't depend on any BasicSR behaviour beyond the `@ARCH_REGISTRY.register`
decorator.

Checkpoints ship as a single `MAN-base.zip` on Google Drive; this script
extracts the requested .pth from the zip on first run (idempotent across
the three convert steps for x2/x3/x4) and converts it to ONNX (FP32 by
default, FP16 available via `--fp16`).
"""

import argparse
import os
import sys
import types
import zipfile

import torch

try:
    import onnxconverter_common
    HAS_ONNX_CONVERTER = True
except ImportError:
    HAS_ONNX_CONVERTER = False


# ---------------------------------------------------------------------------
# BasicSR stub — MAN_arch.py does
#   from basicsr.utils.registry import ARCH_REGISTRY
# and then decorates the MAN class with @ARCH_REGISTRY.register(). We don't
# need BasicSR's training framework; a no-op registry is enough.
# ---------------------------------------------------------------------------

def _install_basicsr_stub():
    if "basicsr.utils.registry" in sys.modules:
        return
    basicsr = types.ModuleType("basicsr")
    utils = types.ModuleType("basicsr.utils")
    registry = types.ModuleType("basicsr.utils.registry")

    class _Registry:
        def __init__(self, *_args, **_kwargs):
            pass

        def register(self, obj=None, **_kwargs):
            # support both @REGISTRY.register and @REGISTRY.register()
            if obj is None:
                return lambda x: x
            return obj

    registry.ARCH_REGISTRY = _Registry("ARCH")
    basicsr.utils = utils
    utils.registry = registry
    sys.modules["basicsr"] = basicsr
    sys.modules["basicsr.utils"] = utils
    sys.modules["basicsr.utils.registry"] = registry


# ---------------------------------------------------------------------------
# Checkpoint extraction
# ---------------------------------------------------------------------------

def _extract_pth(zip_path, member_name, dest_dir):
    """Extract `member_name` from `zip_path` into `dest_dir` if not present.

    Returns the path to the extracted .pth. The zip member is matched
    case-insensitively against its basename so callers don't need to
    track the zip's internal directory layout.
    """
    dest_path = os.path.join(dest_dir, member_name)
    if os.path.exists(dest_path):
        return dest_path

    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        target = member_name.lower()
        match = None
        for name in zf.namelist():
            if os.path.basename(name).lower() == target:
                match = name
                break
        if match is None:
            raise FileNotFoundError(
                f"{member_name} not found inside {zip_path}; "
                f"available: {[os.path.basename(n) for n in zf.namelist()]}")
        with zf.open(match) as src, open(dest_path, "wb") as dst:
            dst.write(src.read())
    print(f"  Extracted {member_name} -> {dest_path}")
    return dest_path


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_to_onnx(model, output_path, scale, height=256, width=256,
                   dynamic_shapes=True, opset_version=20, fp16=False):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    import onnx

    # MAN-Base is pure CNN (36 MABs with multi-scale depthwise large
    # kernels). Tracing + constant folding at the deployment dim balloons
    # memory (~64 GB at 512x512). Trace at a small dim with dynamic axes,
    # then bake the static shape post-export — same trick BSRGAN uses.
    # This is safe here because MAN has no concrete-dim ops inside.
    trace_dim = 64
    dummy_input = torch.randn(1, 3, trace_dim, trace_dim)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':  {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'},
        },
        verbose=False,
    )
    print(f"Exported: {output_path} (traced at {trace_dim}x{trace_dim})")

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX verification passed.")

    if not dynamic_shapes:
        from onnx.tools import update_model_dims
        from onnx import shape_inference
        onnx_model = update_model_dims.update_inputs_outputs_dims(
            onnx_model,
            {'input':  [1, 3, height, width]},
            {'output': [1, 3, height * scale, width * scale]})
        onnx_model = shape_inference.infer_shapes(onnx_model)
        onnx.save(onnx_model, output_path)
        print(f"  Static dims baked: "
              f"{height}x{width} -> {height * scale}x{width * scale}")

    if fp16:
        if not HAS_ONNX_CONVERTER:
            print("Warning: onnxconverter-common not installed. Skipping FP16 conversion.")
            return
        print("Converting to FP16...")
        from onnxconverter_common import float16
        # keep_io_types=True + Cast block — same defensive recipe DAT-2 uses.
        # MAN doesn't have window-attention Cast nodes, but the recipe is
        # cheap and avoids any analogous internal-Cast surprises.
        fp16_model = float16.convert_float_to_float16(
            onnx_model, keep_io_types=True, op_block_list=['Cast'])
        onnx.save(fp16_model, output_path)
        print(f"FP16 model saved to {output_path}")


# ---------------------------------------------------------------------------
# Conversion entry points
# ---------------------------------------------------------------------------

def _load_man_class():
    """Import the MAN class from the vendored repo with the basicsr stub."""
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DTAI_ROOT = os.environ.get("DTAI_ROOT", os.path.join(SCRIPT_DIR, "../.."))
    sys.path.insert(0, os.path.join(DTAI_ROOT, "vendor", "man", "archs"))
    _install_basicsr_stub()
    from MAN_arch import MAN  # noqa: E402  (after sys.path/stub setup)
    return MAN


class _BroadcastAdd(torch.nn.Module):
    """In-place replacement for MAN's MeanShift `sub_mean` Conv.

    MeanShift is initialised with identity 1x1 weights + bias = -rgb_mean
    (frozen). It's algebraically a constant per-channel add — replacing it
    with an explicit Add op is bit-identical and avoids the CoreML
    MLProgram quirk where the original Conv triggered a phantom
    `cast_to__sub_mean_Conv` feature-input requirement.
    """
    def __init__(self, bias_3):
        super().__init__()
        self.register_buffer('bias', bias_3.view(1, -1, 1, 1).clone())

    def forward(self, x):
        return x + self.bias


def _replace_sub_mean(model):
    """Swap MAN's MeanShift sub_mean for a broadcast Add. Verifies that
    MeanShift's weights are identity (the default — frozen at init)."""
    sm = model.sub_mean
    W_sm = sm.weight.data.squeeze(-1).squeeze(-1)     # (3, 3)
    if not torch.allclose(
        W_sm, torch.eye(W_sm.shape[0], dtype=W_sm.dtype), atol=1e-5):
        raise RuntimeError(
            "sub_mean weight is not identity; cannot replace with broadcast Add. "
            f"diag={W_sm.diag().tolist()}, "
            f"max_offdiag={(W_sm - W_sm.diag().diag()).abs().max().item():.4f}")
    model.sub_mean = _BroadcastAdd(sm.bias.data)


def _verify_fusion_equivalent(orig_model, fused_model, scale):
    """Sanity-check that fusion didn't change model output."""
    orig_model.eval()
    fused_model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out_orig = orig_model(x)
        out_fused = fused_model(x)
    if not torch.allclose(out_orig, out_fused, atol=1e-4, rtol=1e-4):
        max_diff = (out_orig - out_fused).abs().max().item()
        raise RuntimeError(
            f"sub_mean fusion broke output equivalence (max diff = {max_diff:.6f})")
    print(f"  sub_mean fusion verified (max diff {(out_orig - out_fused).abs().max().item():.2e})")


def convert(checkpoint_zip, checkpoint_name, output, scale,
            height=256, width=256,
            dynamic_shapes=True, opset=20, fp16=False, static=False):
    """Entry point for programmatic conversion."""
    scale = int(scale)

    zip_dir = os.path.dirname(checkpoint_zip) or "."
    pth_path = _extract_pth(checkpoint_zip, checkpoint_name, zip_dir)

    MAN = _load_man_class()
    print(f"Loading MAN-Base model (scale={scale}) from {pth_path}")
    # MAN-Base defaults: n_resblocks=36, n_feats=180, n_colors=3, res_scale=1.0
    model = MAN(n_resblocks=36, n_feats=180, scale=scale)
    state = torch.load(pth_path, map_location='cpu', weights_only=False)
    # BasicSR checkpoints wrap weights under 'params' (or 'params_ema')
    if isinstance(state, dict) and 'params_ema' in state:
        state = state['params_ema']
    elif isinstance(state, dict) and 'params' in state:
        state = state['params']
    model.load_state_dict(state, strict=True)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: MAN-Base")
    print(f"  Scale:        x{scale}")
    print(f"  Parameters:   {param_count:,}")

    print("Replacing sub_mean Conv with broadcast Add...")
    import copy
    orig_model = copy.deepcopy(model)
    _replace_sub_mean(model)
    _verify_fusion_equivalent(orig_model, model, scale)
    del orig_model

    print("Exporting to ONNX...")
    export_to_onnx(model, output, scale,
                   height=height, width=width,
                   dynamic_shapes=dynamic_shapes and not static,
                   opset_version=opset, fp16=fp16)


def main():
    parser = argparse.ArgumentParser(description='Export MAN-Base to ONNX')
    parser.add_argument('--checkpoint-zip', required=True,
                        help='Path to MAN-base.zip')
    parser.add_argument('--checkpoint-name', required=True,
                        help='Filename to extract from the zip (e.g. MANx2_DF2K.pth)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--scale', type=int, required=True, choices=[2, 3, 4])
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--opset', type=int, default=20)
    parser.add_argument('--fp16', action='store_true',
                        help='convert weights to FP16 after export')
    parser.add_argument('--static', action='store_true',
                        help='bake input height/width into the graph '
                             '(disables dynamic shape axes)')
    args = parser.parse_args()

    convert(args.checkpoint_zip, args.checkpoint_name, args.output, args.scale,
            height=args.height, width=args.width,
            opset=args.opset, fp16=args.fp16, static=args.static)


if __name__ == '__main__':
    main()
