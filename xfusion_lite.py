#!/usr/bin/env python3
"""XFusion Ultra-Lite CLI for fast demos and validation."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


def create_synthetic_data():
    """Create deterministic synthetic input data for EDVR model testing."""
    # Fixed seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Model expects: sample with 'lq' and 'hq' keys
    # lq: (batch, time_frames, channels, height, width)
    # hq: (batch, hi_time_frames, channels, height*scale, width*scale)

    batch_size = 1
    time_frames = 3  # Low resolution frames
    hi_time_frames = 2  # High resolution frames
    channels = 1  # Grayscale x-ray
    height, width = 64, 64  # Small size for fast testing
    scale = 4  # Upscaling factor

    # Create synthetic low-resolution sequence (simulating camera frames)
    lq_data = torch.randn(batch_size, time_frames, channels, height, width) * 0.5 + 0.5
    lq_data = torch.clamp(lq_data, 0, 1)

    # Create synthetic high-resolution frames (simulating high-speed camera)
    hq_data = (
        torch.randn(batch_size, hi_time_frames, channels, height * scale, width * scale)
        * 0.5
        + 0.5
    )
    hq_data = torch.clamp(hq_data, 0, 1)

    return {"lq": lq_data, "hq": hq_data}


def validate_imports():
    """Validate that all required imports work."""
    try:
        import cv2
        import numpy as np
        import pandas as pd
        import torch
        import torchvision
        import yaml
        from natsort import natsorted
        from PIL import Image
        from tqdm import tqdm

        print("[OK] All imports successful")
        return True
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False


def validate_shapes():
    """Validate synthetic data shapes."""
    data = create_synthetic_data()

    lq_shape = data["lq"].shape
    hq_shape = data["hq"].shape

    expected_lq = (1, 3, 1, 64, 64)
    expected_hq = (1, 2, 1, 256, 256)

    print(f"LQ shape: {lq_shape} (expected: {expected_lq})")
    print(f"HQ shape: {hq_shape} (expected: {expected_hq})")

    if lq_shape == expected_lq and hq_shape == expected_hq:
        print("[OK] Shapes valid")
        return True
    else:
        print("[ERROR] Shape mismatch")
        return False


def run_synthetic_inference():
    """Run inference on synthetic data and measure timing."""
    from xfusion.inference.model.edvr_models import EDVRSTFTempRank

    # Model configuration (minimal for fast testing)
    model_config = {
        "num_in_ch": 1,
        "num_out_ch": 1,
        "num_feat": 64,
        "num_frame": 5,  # Total frames (3 lo + 2 hi)
        "num_frame_hi": 2,
        "deformable_groups": 8,
        "num_extract_block": 5,
        "num_reconstruct_block": 10,
        "center_frame_idx": 1,
        "hr_in": False,
        "with_predeblur": False,
        "with_tsa": True,
        "with_transformer": False,
        "patchsize": None,
        "stack_num": None,
        "fuse_searched_feat_ok": False,
        "num_frame_search": None,
        "downsample_hi_ok": True,
        "num_hidden_feat": None,
        "scale": None,
    }

    print("Creating synthetic input...")
    data = create_synthetic_data()

    print("Initializing model...")
    model = EDVRSTFTempRank(**model_config)
    model.eval()

    # Performance optimization: Enable inference mode
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes

    # Force CPU mode
    device = torch.device("cpu")
    model = model.to(device)
    data["lq"] = data["lq"].to(device)
    data["hq"] = data["hq"].to(device)

    print("Running inference...")
    start_time = time.perf_counter()

    with torch.inference_mode():  # More efficient than no_grad()
        result = model(data)

    end_time = time.perf_counter()
    inference_time = (end_time - start_time) * 1000  # Convert to ms

    print(f"[OK] Inference completed in {inference_time:.2f} ms")
    print(f"Output shape: {result['out'].shape}")

    # Save a small output file
    output_path = Path("synthetic_output.pt")
    torch.save(result["out"], output_path)
    print(f"[OK] Output saved to {output_path}")

    return inference_time


def main():
    parser = argparse.ArgumentParser(description="XFusion Ultra-Lite CLI")
    parser.add_argument(
        "command", choices=["validate", "infer-synthetic"], help="Command to run"
    )

    args = parser.parse_args()

    if args.command == "validate":
        print("=== XFusion Ultra-Lite Validation ===")
        imports_ok = validate_imports()
        shapes_ok = validate_shapes()

        if imports_ok and shapes_ok:
            print("[OK] All validations passed")
            sys.exit(0)
        else:
            print("[ERROR] Validation failed")
            sys.exit(1)

    elif args.command == "infer-synthetic":
        print("=== XFusion Ultra-Lite Synthetic Inference ===")
        try:
            timing = run_synthetic_inference()
            print(f"=== Completed in {timing:.2f} ms ===")
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
