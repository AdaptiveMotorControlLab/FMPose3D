"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0

Shared fixtures, markers, and skip-helpers for the ``fmpose3d_api`` test suite.

Skip logic
----------
* **weights_ready(filename)** – ``True`` when the HuggingFace-cached file
  already exists on disk *or* we can reach ``huggingface.co`` so that
  ``hf_hub_download`` will succeed.
* **has_internet** – evaluated once at collection time via a quick TCP probe.
* **HF_HUB_OFFLINE** – if set to ``"1"`` in the environment the network
  check is skipped entirely (consistent with how ``huggingface_hub``
  itself behaves).
"""

from __future__ import annotations

import os
import socket

import pytest

# ---------------------------------------------------------------------------
# HuggingFace repo & filenames (must match fmpose3d.fmpose3d._HF_REPO_ID)
# ---------------------------------------------------------------------------

HF_REPO_ID: str = "DeepLabCut/FMPose3D"

HUMAN_WEIGHTS_FILENAME: str = "fmpose3d_humans.pth"
ANIMAL_WEIGHTS_FILENAME: str = "fmpose3d_animals.pth"

# ---------------------------------------------------------------------------
# Connectivity helpers
# ---------------------------------------------------------------------------


def _has_internet(host: str = "huggingface.co", port: int = 443, timeout: float = 3) -> bool:
    """Return ``True`` if *host* is reachable via TCP."""
    if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        return False
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except OSError:
        return False


def _weights_cached(filename: str) -> bool:
    """Return ``True`` if *filename* already lives in the local HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache

        result = try_to_load_from_cache(HF_REPO_ID, filename)
        return isinstance(result, str)
    except Exception:
        return False


def weights_ready(filename: str) -> bool:
    """``True`` when we can obtain *filename* — either from cache or network."""
    return _weights_cached(filename) or _has_internet()


# Evaluate once at collection time.
HAS_INTERNET: bool = _has_internet()
HUMAN_WEIGHTS_READY: bool = weights_ready(HUMAN_WEIGHTS_FILENAME)
ANIMAL_WEIGHTS_READY: bool = weights_ready(ANIMAL_WEIGHTS_FILENAME)

try:
    import deeplabcut  # noqa: F401

    DLC_AVAILABLE: bool = True
except ImportError:
    DLC_AVAILABLE = False

# ---------------------------------------------------------------------------
# Reusable skip markers
# ---------------------------------------------------------------------------

requires_network = pytest.mark.skipif(
    not HAS_INTERNET,
    reason="No internet connection (cannot reach huggingface.co)",
)

requires_human_weights = pytest.mark.skipif(
    not HUMAN_WEIGHTS_READY,
    reason="Human weights not cached and no internet connection",
)

requires_animal_weights = pytest.mark.skipif(
    not ANIMAL_WEIGHTS_READY,
    reason="Animal weights not cached and no internet connection",
)

requires_dlc = pytest.mark.skipif(
    not DLC_AVAILABLE,
    reason="DeepLabCut is not installed",
)
