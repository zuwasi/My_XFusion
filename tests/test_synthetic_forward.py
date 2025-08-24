"""Test synthetic forward pass functionality."""


import numpy as np
import pytest
import torch


def create_test_data():
    """Create test data for forward pass."""
    np.random.seed(42)
    torch.manual_seed(42)

    batch_size = 1
    time_frames = 3
    hi_time_frames = 2
    channels = 1
    height, width = 64, 64
    scale = 4

    lq_data = torch.randn(batch_size, time_frames, channels, height, width) * 0.5 + 0.5
    lq_data = torch.clamp(lq_data, 0, 1)

    hq_data = (
        torch.randn(batch_size, hi_time_frames, channels, height * scale, width * scale)
        * 0.5
        + 0.5
    )
    hq_data = torch.clamp(hq_data, 0, 1)

    return {"lq": lq_data, "hq": hq_data}


def test_synthetic_data_shapes():
    """Test that synthetic data has correct shapes."""
    data = create_test_data()

    assert data["lq"].shape == (1, 3, 1, 64, 64)
    assert data["hq"].shape == (1, 2, 1, 256, 256)
    assert data["lq"].dtype == torch.float32
    assert data["hq"].dtype == torch.float32


def test_synthetic_forward_cpu():
    """Test synthetic forward pass on CPU."""
    from xfusion.inference.model.edvr_models import EDVRSTFTempRank

    model_config = {
        "num_in_ch": 1,
        "num_out_ch": 1,
        "num_feat": 64,
        "num_frame": 5,
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

    data = create_test_data()
    model = EDVRSTFTempRank(**model_config)
    model.eval()

    device = torch.device("cpu")
    model = model.to(device)
    data["lq"] = data["lq"].to(device)
    data["hq"] = data["hq"].to(device)

    with torch.no_grad():
        result = model(data)

    # Validate output
    assert "out" in result
    assert result["out"].shape == (1, 1, 256, 256)
    assert result["out"].dtype == torch.float32


@pytest.mark.gpu
def test_synthetic_forward_gpu():
    """Test synthetic forward pass on GPU if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from xfusion.inference.model.edvr_models import EDVRSTFTempRank

    model_config = {
        "num_in_ch": 1,
        "num_out_ch": 1,
        "num_feat": 32,  # Smaller for GPU test
        "num_frame": 5,
        "num_frame_hi": 2,
        "deformable_groups": 4,
        "num_extract_block": 3,
        "num_reconstruct_block": 5,
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

    data = create_test_data()
    model = EDVRSTFTempRank(**model_config)
    model.eval()

    device = torch.device("cuda")
    model = model.to(device)
    data["lq"] = data["lq"].to(device)
    data["hq"] = data["hq"].to(device)

    with torch.no_grad():
        result = model(data)

    assert "out" in result
    assert result["out"].device.type == "cuda"
