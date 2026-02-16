"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0

Functional integration tests for the human and animal 3D-lifting pipelines.

These tests download pretrained weights from HuggingFace Hub (or use the
local cache) and are marked ``@pytest.mark.functional`` **and**
``@pytest.mark.network`` (the latter because the first run requires
internet access to populate the cache).

Tests are automatically **skipped** when:
* the weights are not in the local HF cache *and* the machine is offline, or
* ``HF_HUB_OFFLINE=1`` is set and the weights are not cached.

Two test tiers
--------------
``TestHuman3DLifting`` / ``TestAnimal3DLifting``
    Test ``pose_3d`` with synthetic 2D keypoints.  Only the model weights
    are required — no images, no HRNet, no DeepLabCut.

``TestHumanEndToEnd`` / ``TestAnimalEndToEnd``
    Full ``prepare_2d`` → ``pose_3d`` pipeline.  These additionally
    require test images on disk and the relevant 2D estimator
    (HRNet / DeepLabCut).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fmpose3d.inference_api.fmpose3d import FMPose3DInference

from .conftest import (
    requires_human_weights,
    requires_animal_weights,
    requires_dlc,
)

# ---------------------------------------------------------------------------
# Synthetic 2D keypoint helpers
# ---------------------------------------------------------------------------

#: Approximate pixel coordinates for a standing human in a 640×480 image
#: (17 H36M joints).  Not anatomically precise — just plausible enough
#: for the normalisation / lifting maths to exercise real code paths.
_HUMAN_KP_17 = np.array(
    [
        [320, 210],  # 0  hip (root)
        [290, 210],  # 1  right hip
        [280, 290],  # 2  right knee
        [275, 370],  # 3  right ankle
        [350, 210],  # 4  left hip
        [360, 290],  # 5  left knee
        [365, 370],  # 6  left ankle
        [320, 170],  # 7  spine
        [320, 130],  # 8  thorax
        [320, 100],  # 9  neck / nose
        [320, 80],   # 10 head
        [280, 130],  # 11 left shoulder
        [250, 170],  # 12 left elbow
        [230, 210],  # 13 left wrist
        [360, 130],  # 14 right shoulder
        [390, 170],  # 15 right elbow
        [410, 210],  # 16 right wrist
    ],
    dtype="float32",
)  # shape (17, 2)

#: Approximate pixel coordinates for a quadruped in a 640×480 image
#: (26 Animal3D joints).
_ANIMAL_KP_26 = np.array(
    [
        [180, 220],  # 0
        [460, 220],  # 1
        [320, 250],  # 2
        [200, 260],  # 3
        [440, 260],  # 4
        [170, 300],  # 5
        [470, 300],  # 6
        [320, 200],  # 7  root
        [320, 170],  # 8
        [300, 350],  # 9
        [160, 350],  # 10
        [480, 350],  # 11
        [240, 190],  # 12
        [400, 190],  # 13
        [150, 400],  # 14
        [490, 400],  # 15
        [160, 430],  # 16
        [480, 430],  # 17
        [320, 160],  # 18
        [320, 140],  # 19
        [200, 200],  # 20
        [440, 200],  # 21
        [190, 180],  # 22
        [450, 180],  # 23
        [320, 140],  # 24
        [320, 190],  # 25
    ],
    dtype="float32",
)  # shape (26, 2)

_IMAGE_SIZE_HW = (480, 640)


def _make_keypoints(kp: np.ndarray) -> np.ndarray:
    """Reshape a (J, 2) template to (1, 1, J, 2) — (persons, frames, joints, xy)."""
    return kp[np.newaxis, np.newaxis, :, :]


# ---------------------------------------------------------------------------
# Paths for end-to-end tests (images must be on disk)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _find_first(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


HUMAN_TEST_IMAGE = _find_first([
    PROJECT_ROOT / "demo" / "images" / "running.png",
])
ANIMAL_TEST_IMAGE = _find_first([
    PROJECT_ROOT / "animals" / "demo" / "images" / "dog.JPEG",
    PROJECT_ROOT / "animals" / "demo" / "images" / "dog.jpeg",
    PROJECT_ROOT / "animals" / "demo" / "images" / "dog.jpg",
])


# =========================================================================
# Tier 1 — 3D lifting with real weights, synthetic 2D keypoints
# =========================================================================


@pytest.mark.functional
@pytest.mark.network
@requires_human_weights
class TestHuman3DLifting:
    """Test ``pose_3d`` with the real human model and synthetic keypoints."""

    @pytest.fixture(scope="class")
    def api(self):
        """Instantiate *without* explicit weights — triggers HF download."""
        return FMPose3DInference(device="cpu")

    @pytest.fixture(scope="class")
    def result_pair(self, api):
        """Lift synthetic 2D keypoints twice (same seed) for reproducibility."""
        kp = _make_keypoints(_HUMAN_KP_17)
        r_a = api.pose_3d(kp, image_size=_IMAGE_SIZE_HW, seed=42)
        r_b = api.pose_3d(kp, image_size=_IMAGE_SIZE_HW, seed=42)
        return r_a, r_b

    def test_3d_shape(self, result_pair):
        r3d, _ = result_pair
        assert r3d.poses_3d.shape == (1, 17, 3)
        assert r3d.poses_3d_world.shape == (1, 17, 3)

    def test_root_joint_zeroed(self, result_pair):
        r3d, _ = result_pair
        np.testing.assert_allclose(r3d.poses_3d[:, 0, :], 0.0, atol=1e-6)

    def test_world_z_floor(self, result_pair):
        r3d, _ = result_pair
        assert np.min(r3d.poses_3d_world[:, :, 2]) >= -1e-6

    def test_poses_finite(self, result_pair):
        r3d, _ = result_pair
        assert np.all(np.isfinite(r3d.poses_3d))
        assert np.all(np.isfinite(r3d.poses_3d_world))

    def test_poses_reasonable_magnitude(self, result_pair):
        r3d, _ = result_pair
        assert np.max(np.abs(r3d.poses_3d)) < 1e4
        assert np.max(np.abs(r3d.poses_3d_world)) < 1e4

    def test_reproducibility(self, result_pair):
        r_a, r_b = result_pair
        np.testing.assert_allclose(r_a.poses_3d, r_b.poses_3d, atol=1e-6)
        np.testing.assert_allclose(r_a.poses_3d_world, r_b.poses_3d_world, atol=1e-6)


@pytest.mark.functional
@pytest.mark.network
@requires_animal_weights
class TestAnimal3DLifting:
    """Test ``pose_3d`` with the real animal model and synthetic keypoints."""

    @pytest.fixture(scope="class")
    def api(self):
        return FMPose3DInference.for_animals(device="cpu")

    @pytest.fixture(scope="class")
    def result_pair(self, api):
        kp = _make_keypoints(_ANIMAL_KP_26)
        r_a = api.pose_3d(kp, image_size=_IMAGE_SIZE_HW, seed=42)
        r_b = api.pose_3d(kp, image_size=_IMAGE_SIZE_HW, seed=42)
        return r_a, r_b

    def test_3d_shape(self, result_pair):
        r3d, _ = result_pair
        assert r3d.poses_3d.shape == (1, 26, 3)
        assert r3d.poses_3d_world.shape == (1, 26, 3)

    def test_poses_finite(self, result_pair):
        r3d, _ = result_pair
        assert np.all(np.isfinite(r3d.poses_3d))
        assert np.all(np.isfinite(r3d.poses_3d_world))

    def test_poses_reasonable_magnitude(self, result_pair):
        r3d, _ = result_pair
        assert np.max(np.abs(r3d.poses_3d)) < 1e4
        assert np.max(np.abs(r3d.poses_3d_world)) < 1e4

    def test_reproducibility(self, result_pair):
        r_a, r_b = result_pair
        np.testing.assert_allclose(r_a.poses_3d, r_b.poses_3d, atol=1e-6)
        np.testing.assert_allclose(r_a.poses_3d_world, r_b.poses_3d_world, atol=1e-6)


# =========================================================================
# Tier 2 — Full end-to-end (prepare_2d → pose_3d)
# =========================================================================


@pytest.mark.functional
@pytest.mark.network
@requires_human_weights
@pytest.mark.skipif(
    HUMAN_TEST_IMAGE is None,
    reason="Human test image not found on disk",
)
class TestHumanEndToEnd:
    """Full pipeline: HRNet 2D detection → 3D lifting with real weights."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        from PIL import Image

        api = FMPose3DInference(device="cpu")
        result_2d = api.prepare_2d(source=str(HUMAN_TEST_IMAGE))

        img = Image.open(str(HUMAN_TEST_IMAGE))
        w, h = img.size
        image_size = (h, w)

        result_3d_a = api.pose_3d(result_2d.keypoints, image_size=image_size, seed=42)
        result_3d_b = api.pose_3d(result_2d.keypoints, image_size=image_size, seed=42)

        return {
            "result_2d": result_2d,
            "image_size": image_size,
            "result_3d_a": result_3d_a,
            "result_3d_b": result_3d_b,
        }

    def test_2d_keypoints_shape(self, pipeline):
        r2d = pipeline["result_2d"]
        P, F, J, C = r2d.keypoints.shape
        assert J == 17
        assert C == 2
        assert F >= 1

    def test_2d_scores_shape(self, pipeline):
        r2d = pipeline["result_2d"]
        assert r2d.scores.ndim == 3
        assert r2d.scores.shape[2] == 17

    def test_2d_image_size(self, pipeline):
        r2d = pipeline["result_2d"]
        h, w = pipeline["image_size"]
        assert r2d.image_size == (h, w)

    def test_3d_poses_shape(self, pipeline):
        r3d = pipeline["result_3d_a"]
        F = pipeline["result_2d"].keypoints.shape[1]
        assert r3d.poses_3d.shape == (F, 17, 3)
        assert r3d.poses_3d_world.shape == (F, 17, 3)

    def test_root_joint_zeroed(self, pipeline):
        r3d = pipeline["result_3d_a"]
        np.testing.assert_allclose(r3d.poses_3d[:, 0, :], 0.0, atol=1e-6)

    def test_world_z_floor(self, pipeline):
        r3d = pipeline["result_3d_a"]
        assert np.min(r3d.poses_3d_world[:, :, 2]) >= -1e-6

    def test_poses_finite(self, pipeline):
        r3d = pipeline["result_3d_a"]
        assert np.all(np.isfinite(r3d.poses_3d))
        assert np.all(np.isfinite(r3d.poses_3d_world))

    def test_reproducibility(self, pipeline):
        r3d_a = pipeline["result_3d_a"]
        r3d_b = pipeline["result_3d_b"]
        np.testing.assert_allclose(r3d_a.poses_3d, r3d_b.poses_3d, atol=1e-6)
        np.testing.assert_allclose(
            r3d_a.poses_3d_world, r3d_b.poses_3d_world, atol=1e-6,
        )


@pytest.mark.functional
@pytest.mark.network
@requires_animal_weights
@requires_dlc
@pytest.mark.skipif(
    ANIMAL_TEST_IMAGE is None,
    reason="Animal test image not found on disk",
)
class TestAnimalEndToEnd:
    """Full pipeline: DeepLabCut 2D detection → 3D lifting with real weights."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        from PIL import Image

        api = FMPose3DInference.for_animals(device="cpu")
        result_2d = api.prepare_2d(source=str(ANIMAL_TEST_IMAGE))

        img = Image.open(str(ANIMAL_TEST_IMAGE))
        w, h = img.size
        image_size = (h, w)

        result_3d_a = api.pose_3d(result_2d.keypoints, image_size=image_size, seed=42)
        result_3d_b = api.pose_3d(result_2d.keypoints, image_size=image_size, seed=42)

        return {
            "result_2d": result_2d,
            "image_size": image_size,
            "result_3d_a": result_3d_a,
            "result_3d_b": result_3d_b,
        }

    def test_2d_keypoints_shape(self, pipeline):
        r2d = pipeline["result_2d"]
        _, F, J, C = r2d.keypoints.shape
        assert J == 26
        assert C == 2

    def test_2d_scores_shape(self, pipeline):
        r2d = pipeline["result_2d"]
        assert r2d.scores.ndim == 3
        assert r2d.scores.shape[2] == 26

    def test_3d_poses_shape(self, pipeline):
        r3d = pipeline["result_3d_a"]
        F = pipeline["result_2d"].keypoints.shape[1]
        assert r3d.poses_3d.shape == (F, 26, 3)
        assert r3d.poses_3d_world.shape == (F, 26, 3)

    def test_poses_finite(self, pipeline):
        r3d = pipeline["result_3d_a"]
        assert np.all(np.isfinite(r3d.poses_3d))
        assert np.all(np.isfinite(r3d.poses_3d_world))

    def test_poses_reasonable_magnitude(self, pipeline):
        r3d = pipeline["result_3d_a"]
        assert np.max(np.abs(r3d.poses_3d)) < 1e4
        assert np.max(np.abs(r3d.poses_3d_world)) < 1e4

    def test_reproducibility(self, pipeline):
        r3d_a = pipeline["result_3d_a"]
        r3d_b = pipeline["result_3d_b"]
        np.testing.assert_allclose(r3d_a.poses_3d, r3d_b.poses_3d, atol=1e-6)
        np.testing.assert_allclose(
            r3d_a.poses_3d_world, r3d_b.poses_3d_world, atol=1e-6,
        )
