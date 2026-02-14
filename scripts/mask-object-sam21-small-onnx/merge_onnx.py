#!/usr/bin/env python3
"""Merge ONNX models with external data into single files."""

import argparse
import os

import onnx


def merge_onnx(src_path, dst_path):
    """Load an ONNX model with external data and save as a single file."""
    if os.path.exists(dst_path):
        print(f"  {os.path.basename(dst_path)} already exists, skipping")
        return

    print(f"  {os.path.basename(src_path)} -> {os.path.basename(dst_path)}")
    model = onnx.load(src_path, load_external_data=True)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    onnx.save(model, dst_path)

    # Remove original split files
    os.remove(src_path)
    data_path = src_path + "_data"
    if os.path.exists(data_path):
        os.remove(data_path)


def main():
    parser = argparse.ArgumentParser(description="Merge ONNX external data into single files")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory with downloaded ONNX + external data files")
    parser.add_argument("--output-dir", required=True, help="Directory for merged ONNX output files")
    args = parser.parse_args()

    mappings = [
        ("vision_encoder.onnx", "encoder.onnx"),
        ("prompt_encoder_mask_decoder.onnx", "decoder.onnx"),
    ]

    print("Merging external data into single ONNX files...")
    for src_name, dst_name in mappings:
        src_path = os.path.join(args.checkpoint_dir, src_name)
        dst_path = os.path.join(args.output_dir, dst_name)
        merge_onnx(src_path, dst_path)


if __name__ == "__main__":
    main()
