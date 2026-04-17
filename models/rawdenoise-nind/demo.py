"""Demo: run the RawNIND UtNet2 raw-denoise ONNX models.

This package ships two variants:
  * model_bayer.onnx  – 4ch packed-Bayer input, joint denoise + demosaic
  * model_linear.onnx – 3ch linear Rec.2020 input, denoise only
                        (X-Trans, Foveon, or anywhere demosaic already ran)

Input:  a raw file readable by rawpy (CR2, NEF, ARW, DNG, RAF, ...).
Output: a linear Rec.2020 TIFF (16-bit) or EXR.

Sensor dispatch:
  * `raw.raw_pattern.shape == (2, 2)` → Bayer variant (native raw path)
  * otherwise                         → linear variant, with rawpy demosaicing
                                         to linear Rec.2020 first
"""

import argparse
import os
import time

import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# Bayer path – pack raw mosaic into 4ch [R, G1, G2, B]
# ---------------------------------------------------------------------------

def pack_rggb(bayer: np.ndarray, raw_pattern, color_desc: str) -> np.ndarray:
    """Pack a 2D Bayer mosaic into a 4-channel half-resolution [R, G1, G2, B] tensor."""
    h, w = bayer.shape
    assert h % 2 == 0 and w % 2 == 0, "Bayer dimensions must be even"

    r_plane = g1_plane = g2_plane = b_plane = None
    for i in range(2):
        for j in range(2):
            color = color_desc[int(raw_pattern[i, j])]
            plane = bayer[i::2, j::2]
            if color == "R":
                r_plane = plane
            elif color == "B":
                b_plane = plane
            elif color == "G":
                if g1_plane is None:
                    g1_plane = plane
                else:
                    g2_plane = plane
            else:
                raise ValueError(f"Unsupported Bayer colour: {color!r}")

    if any(p is None for p in (r_plane, g1_plane, g2_plane, b_plane)):
        raise ValueError(
            f"Incomplete RGGB pattern from desc={color_desc!r} "
            f"raw_pattern={raw_pattern.tolist()}"
        )
    return np.stack([r_plane, g1_plane, g2_plane, b_plane], axis=0)


def load_raw_as_packed_bayer(image_path: str):
    """Load a 2×2 Bayer raw, black-level + white-level normalise per channel,
    and pack into a (4, H/2, W/2) tensor. Returns (packed, rgb_xyz_matrix)."""
    import rawpy

    raw = rawpy.imread(image_path)
    assert raw.raw_pattern.shape == (2, 2), "load_raw_as_packed_bayer called on non-Bayer"

    bayer = raw.raw_image_visible.astype(np.float32)
    white = float(raw.white_level)
    color_desc = raw.color_desc.decode("ascii")

    black_per_ch = np.asarray(raw.black_level_per_channel, dtype=np.float32)
    for i in range(2):
        for j in range(2):
            ch_idx = int(raw.raw_pattern[i, j])
            bl = black_per_ch[ch_idx]
            vrange = max(white - bl, 1.0)
            bayer[i::2, j::2] = np.clip((bayer[i::2, j::2] - bl) / vrange, 0.0, 1.0)

    packed = pack_rggb(bayer, raw.raw_pattern, color_desc)
    return packed, np.asarray(raw.rgb_xyz_matrix, dtype=np.float32)


# ---------------------------------------------------------------------------
# Linear path – rawpy demosaic → lin-Rec.2020
# ---------------------------------------------------------------------------

def load_raw_as_lin_rec2020(image_path: str) -> np.ndarray:
    """Demosaic a raw file to a (3, H, W) linear Rec.2020 tensor in [0, 1].

    Uses rawpy's postprocess with a neutral pipeline: linear output, no auto-
    bright, no gamma, no user flip. Output colour space is set to Rec.2020 so
    rawpy applies the camera-matrix + white-balance conversion internally.
    """
    import rawpy

    raw = rawpy.imread(image_path)
    rgb = raw.postprocess(
        output_color=rawpy.ColorSpace.Rec2020,
        output_bps=16,
        no_auto_bright=True,
        use_camera_wb=True,
        gamma=(1.0, 1.0),
        user_flip=0,
    )
    rgb = rgb.astype(np.float32) / 65535.0
    return np.transpose(rgb, (2, 0, 1))  # (3, H, W)


# ---------------------------------------------------------------------------
# camRGB → linear Rec.2020 (Bayer path only; matrix from rawnind_jddc/rawproc.py)
# ---------------------------------------------------------------------------

_XYZ_TO_LIN_REC2020 = np.array(
    [
        [1.71666343, -0.35567332, -0.25336809],
        [-0.66667384, 1.61645574, 0.0157683],
        [0.01764248, -0.04277698, 0.94224328],
    ],
    dtype=np.float32,
)


def cam_rgb_to_lin_rec2020(cam_rgb: np.ndarray, rgb_xyz_matrix: np.ndarray) -> np.ndarray:
    cam_to_xyzd65 = np.linalg.inv(rgb_xyz_matrix[:3, :])
    m = _XYZ_TO_LIN_REC2020 @ cam_to_xyzd65
    h, w, _ = cam_rgb.shape
    out = (m @ cam_rgb.reshape(-1, 3).T).T.reshape(h, w, 3)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _match_gain(other: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Scale `other` so its mean matches `anchor`'s mean (rawproc.match_gain).

    Used at inference to substitute for the training-time match_gain=output
    step, with the input image as the anchor instead of the (unavailable) GT.
    Sign-preserving: if the network output mean is negative, gain is negative
    too, flipping the output back into the anchor's sign convention.
    """
    anchor_mean = float(anchor.mean())
    other_mean = float(other.mean())
    if abs(other_mean) < 1e-12:
        return other  # degenerate — nothing sensible to rescale to
    return other * (anchor_mean / other_mean)


def _crop_mod16(arr: np.ndarray) -> np.ndarray:
    """Crop a (1, C, H, W) tensor so H and W are multiples of 16 (UtNet2 req)."""
    _, _, h, w = arr.shape
    h16 = (h // 16) * 16
    w16 = (w // 16) * 16
    if (h16, w16) != (h, w):
        arr = arr[:, :, :h16, :w16]
        print(f"  Cropped to:    {arr.shape[3]}x{arr.shape[2]} (mod 16)")
    return arr


def _load_session(model_path: str):
    print(f"Loading model: {model_path}")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    return session, model_input.name, model_input.type == "tensor(float16)"


def _save_tiff16(path, rgb: np.ndarray):
    import tifffile
    tifffile.imwrite(path, (rgb * 65535.0).astype(np.uint16))


def _save_exr(path, rgb: np.ndarray):
    import OpenEXR
    import Imath
    h, w, _ = rgb.shape
    header = OpenEXR.Header(w, h)
    half = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    header["channels"] = {"R": half, "G": half, "B": half}
    exr = OpenEXR.OutputFile(path, header)
    r, g, b = (rgb[..., i].astype(np.float16).tobytes() for i in range(3))
    exr.writePixels({"R": r, "G": g, "B": b})
    exr.close()


def _save(output_path: str, rgb_hwc: np.ndarray):
    rgb_hwc = np.clip(rgb_hwc, 0.0, 1.0)
    if os.path.splitext(output_path)[1].lower() == ".exr":
        _save_exr(output_path, rgb_hwc)
    else:
        _save_tiff16(output_path, rgb_hwc)


# ---------------------------------------------------------------------------
# Bayer inference pipeline
# ---------------------------------------------------------------------------

def run_bayer(model_path: str, image_path: str, output_path: str) -> None:
    t0 = time.perf_counter()
    session, input_name, input_is_fp16 = _load_session(model_path)

    print(f"Loading raw (Bayer): {image_path}")
    packed, rgb_xyz_matrix = load_raw_as_packed_bayer(image_path)
    print(f"  Packed shape:  {packed.shape} (C, H, W)")

    arr = packed[np.newaxis]
    arr = _crop_mod16(arr)
    if input_is_fp16:
        arr = arr.astype(np.float16)

    print("Running inference (Bayer)...")
    [output] = session.run(None, {input_name: arr})

    # Bayer model outputs camRGB at an arbitrary learned scale (training used
    # match_gain=output). Gain-match against the input mosaic, then convert
    # camRGB → lin-Rec.2020.
    cam_rgb = output[0].astype(np.float32).transpose(1, 2, 0)
    cam_rgb = _match_gain(cam_rgb, anchor=arr)
    rec2020 = cam_rgb_to_lin_rec2020(cam_rgb, rgb_xyz_matrix)
    print(f"  Output (linear Rec.2020): "
          f"range=[{rec2020.min():.3f}, {rec2020.max():.3f}] mean={rec2020.mean():.3f}")

    _save(output_path, rec2020)
    print(f"Saved: {output_path} (total {time.perf_counter() - t0:.2f}s)")


# ---------------------------------------------------------------------------
# Linear (prgb2prgb) inference pipeline
# ---------------------------------------------------------------------------

def run_linear(model_path: str, image_path: str, output_path: str) -> None:
    t0 = time.perf_counter()
    session, input_name, input_is_fp16 = _load_session(model_path)

    print(f"Loading raw (linear, via rawpy demosaic): {image_path}")
    rec2020_in = load_raw_as_lin_rec2020(image_path)
    print(f"  Demosaicked:   {rec2020_in.shape} (C, H, W)")

    arr = rec2020_in[np.newaxis]
    arr = _crop_mod16(arr)
    if input_is_fp16:
        arr = arr.astype(np.float16)

    print("Running inference (linear)...")
    [output] = session.run(None, {input_name: arr})

    # Like the Bayer variant, the network output is at an arbitrary learned
    # scale (training also used match_gain=output). Gain-match against the
    # input. No colour conversion needed — input and output both live in
    # linear Rec.2020.
    rec2020 = output[0].astype(np.float32).transpose(1, 2, 0)
    rec2020 = _match_gain(rec2020, anchor=arr)
    print(f"  Output (linear Rec.2020): "
          f"range=[{rec2020.min():.3f}, {rec2020.max():.3f}] mean={rec2020.mean():.3f}")

    _save(output_path, rec2020)
    print(f"Saved: {output_path} (total {time.perf_counter() - t0:.2f}s)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _dispatch_variant(image_path: str) -> str:
    """Return 'bayer' or 'linear' based on the raw file's sensor pattern."""
    import rawpy
    with rawpy.imread(image_path) as raw:
        return "bayer" if raw.raw_pattern.shape == (2, 2) else "linear"


def demo(model_dir, image, output, variant=None, **kwargs):
    """Entry point invoked by the framework for type=multi models.

    If `variant` is not given, auto-dispatch based on sensor pattern.
    """
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    if variant is None:
        variant = _dispatch_variant(image)

    if variant == "bayer":
        run_bayer(os.path.join(model_dir, "model_bayer.onnx"), image, output)
    elif variant == "linear":
        run_linear(os.path.join(model_dir, "model_linear.onnx"), image, output)
    else:
        raise ValueError(f"Unknown variant: {variant!r} (expected 'bayer' or 'linear')")


def main():
    parser = argparse.ArgumentParser(description="RawNIND UtNet2 raw-denoise demo.")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing model_bayer.onnx and model_linear.onnx")
    parser.add_argument("--image", type=str, required=True,
                        help="Raw file (CR2, NEF, ARW, DNG, RAF, ...)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output .tif (16-bit linear Rec.2020) or .exr")
    parser.add_argument("--variant", choices=["bayer", "linear"], default=None,
                        help="Force a variant; default: auto-dispatch by sensor")
    args = parser.parse_args()

    demo(args.model_dir, args.image, args.output, variant=args.variant)


if __name__ == "__main__":
    main()
