"""Test that all required imports work correctly."""



def test_core_imports():
    """Test that core dependencies can be imported."""
    import numpy as np
    import torch

    assert torch.__version__ is not None
    assert np.__version__ is not None


def test_xfusion_imports():
    """Test that xfusion modules can be imported."""
    from xfusion.inference.model.edvr_models import EDVRSTFTempRank

    # Test model can be instantiated (basic config)
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

    model = EDVRSTFTempRank(**model_config)
    assert model is not None


def test_torch_device():
    """Test torch device availability."""
    import torch

    assert torch.cpu.is_available()
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
