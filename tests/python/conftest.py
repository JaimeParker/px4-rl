import os

import pytest


def _cuda_unavailable_reason():
    from px4rl import _lib

    if not _lib.has_cuda_library():
        return "px4rl was built without CUDA"
    try:
        import torch
    except ImportError:
        return "torch is not installed"
    if not torch.cuda.is_available():
        return "no CUDA device is available"
    return None


def pytest_collection_modifyitems(config, items):
    reason = _cuda_unavailable_reason()
    if reason is None:
        return
    if os.environ.get("PX4RL_REQUIRE_CUDA") == "1":
        raise pytest.UsageError(f"PX4RL_REQUIRE_CUDA=1 but the CUDA tests cannot run: {reason}")
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)
