"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0

Tests for fmpose3d/fmpose3d.py — the high-level inference API.

Organised into:
  * Unit tests for individual components (no pretrained weights needed).
  * Functional integration tests for the full human and animal pipelines
    (require pretrained weights on disk; marked ``@pytest.mark.functional``).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from fmpose3d.fmpose3d import (
    AnimalPostProcessor,
    FMPose3DInference,
    HRNetEstimator,
    HumanPostProcessor,
    Pose2DResult,
    Pose3DResult,
    SuperAnimalEstimator,
    _default_components,
    _INTERPOLATION_RULES,
    _QUADRUPED80K_TO_ANIMAL3D,
    _ANIMAL_LIMB_CONNECTIONS,
    _DEFAULT_CAM_ROTATION,
    apply_limb_regularization,
    compute_limb_regularization_matrix,
)
from fmpose3d.common.config import FMPose3DConfig, InferenceConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _find_first(candidates: list[Path]) -> Path | None:
    """Return the first path that exists, or ``None``."""
    for p in candidates:
        if p.exists():
            return p
    return None


HUMAN_WEIGHTS = _find_first([
    PROJECT_ROOT / "pre_trained_models" / "fmpose3d_h36m" / "FMpose3D_pretrained_weights.pth",
])
ANIMAL_WEIGHTS = _find_first([
    PROJECT_ROOT / "animals" / "pre_trained_models" / "fmpose3d_animals" / "fmpose3d_animals_pretrained_weights.pth",
    PROJECT_ROOT / "pre_trained_models" / "fmpose3d_animals" / "fmpose3d_animals_pretrained_weights.pth",
])
HUMAN_TEST_IMAGE = _find_first([
    PROJECT_ROOT / "demo" / "images" / "running.png",
])
ANIMAL_TEST_IMAGE = _find_first([
    PROJECT_ROOT / "animals" / "demo" / "images" / "dog.JPEG",
    PROJECT_ROOT / "animals" / "demo" / "images" / "dog.jpeg",
    PROJECT_ROOT / "animals" / "demo" / "images" / "dog.jpg",
])

# Evaluated at collection time — files either exist or they don't.
_human_pipeline_available = HUMAN_WEIGHTS is not None and HUMAN_TEST_IMAGE is not None
_animal_pipeline_available = ANIMAL_WEIGHTS is not None and ANIMAL_TEST_IMAGE is not None

try:
    import deeplabcut  # noqa: F401

    _dlc_available = True
except ImportError:
    _dlc_available = False

_animal_pipeline_available = _animal_pipeline_available and _dlc_available


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ZeroVelocityModel(torch.nn.Module):
    """Trivial model that always predicts zero velocity (for unit tests).

    Because the velocity is zero, the Euler sampler output is just the
    initial random noise — sufficient for testing shapes, seeding, and
    post-processing logic.
    """

    def forward(self, c_2d: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(y)


def _make_ready_api(
    model_type: str = "fmpose3d",
    test_augmentation: bool = False,
) -> FMPose3DInference:
    """Return an ``FMPose3DInference`` with a mock model pre-installed.

    ``setup_runtime`` is replaced by a no-op so ``pose_3d`` can be
    called without real weights on disk.
    """
    inference_cfg = InferenceConfig(test_augmentation=test_augmentation)
    if model_type == "fmpose3d_animals":
        api = FMPose3DInference.for_animals(
            device="cpu",
            inference_cfg=inference_cfg,
        )
    else:
        api = FMPose3DInference(
            inference_cfg=inference_cfg,
            device="cpu",
        )
    api._model_3d = _ZeroVelocityModel()
    api.setup_runtime = lambda: None  # type: ignore[assignment]
    return api


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def human_api() -> FMPose3DInference:
    """Lightweight human API instance (no weights loaded)."""
    return FMPose3DInference(device="cpu")


@pytest.fixture
def animal_api() -> FMPose3DInference:
    """Lightweight animal API instance (no weights loaded)."""
    return FMPose3DInference.for_animals(device="cpu")


@pytest.fixture
def ready_human_api() -> FMPose3DInference:
    """Human API with mock model (TTA disabled)."""
    return _make_ready_api("fmpose3d", test_augmentation=False)


@pytest.fixture
def ready_animal_api() -> FMPose3DInference:
    """Animal API with mock model."""
    return _make_ready_api("fmpose3d_animals", test_augmentation=False)


# =========================================================================
# Unit tests — _map_keypoints
# =========================================================================


class TestMapKeypoints:
    """Tests for ``SuperAnimalEstimator._map_keypoints``."""

    def _source_array(self, num_ind: int = 1, num_src: int = 40) -> np.ndarray:
        """Create a synthetic source array where src[i] = (i*10, i*10+1)."""
        xy = np.zeros((num_ind, num_src, 2), dtype="float32")
        for i in range(num_src):
            xy[:, i, :] = [i * 10, i * 10 + 1]
        return xy

    def test_output_shape(self):
        xy = self._source_array(1, 40)
        mapped = SuperAnimalEstimator._map_keypoints(xy)
        assert mapped.shape == (1, 26, 2)

    def test_direct_mapped_joints(self):
        """Directly-mapped joints land at the correct source position."""
        xy = self._source_array(1, 40)
        mapped = SuperAnimalEstimator._map_keypoints(xy)

        # target[0] ← source[10]
        np.testing.assert_allclose(mapped[0, 0], xy[0, 10])
        # target[1] ← source[5]
        np.testing.assert_allclose(mapped[0, 1], xy[0, 5])
        # target[24] ← source[0]
        np.testing.assert_allclose(mapped[0, 24], xy[0, 0])

    def test_interpolated_joints(self):
        """Interpolated joints are the mean of their two source joints."""
        xy = self._source_array(1, 40)
        mapped = SuperAnimalEstimator._map_keypoints(xy)

        for tgt_idx, (s1, s2) in _INTERPOLATION_RULES.items():
            expected = (xy[0, s1] + xy[0, s2]) / 2.0
            np.testing.assert_allclose(
                mapped[0, tgt_idx],
                expected,
                err_msg=f"target[{tgt_idx}] should be mean of source[{s1}] and source[{s2}]",
            )

    def test_few_source_keypoints_produce_nan(self):
        """Out-of-range source indices leave NaN in the output."""
        # Only 5 source joints → most mappings are out of range.
        xy = self._source_array(1, 5)
        mapped = SuperAnimalEstimator._map_keypoints(xy)

        # target[0] ← source[10], but 10 >= 5, so should be NaN
        assert np.isnan(mapped[0, 0, 0])
        # target[24] ← source[0], 0 < 5, so should be valid
        np.testing.assert_allclose(mapped[0, 24], xy[0, 0])
        # target[2] ← interp(source[3], source[4]), both < 5, valid
        expected = (xy[0, 3] + xy[0, 4]) / 2.0
        np.testing.assert_allclose(mapped[0, 2], expected)

    def test_multiple_individuals(self):
        """Multiple individuals are handled independently."""
        xy = self._source_array(3, 40)
        mapped = SuperAnimalEstimator._map_keypoints(xy)
        assert mapped.shape == (3, 26, 2)


# =========================================================================
# Unit tests — limb regularisation
# =========================================================================


class TestComputeLimbRegularizationMatrix:
    def test_already_vertical_returns_identity(self):
        """Limb vectors along (0, 0, 1) → identity rotation."""
        pose = np.zeros((26, 3))
        for start_idx, end_idx in _ANIMAL_LIMB_CONNECTIONS:
            pose[start_idx] = [0, 0, 1]
            pose[end_idx] = [0, 0, 0]
        R = compute_limb_regularization_matrix(pose)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)

    def test_horizontal_rotated_to_vertical(self):
        """Limb vectors along (1, 0, 0) → R maps (1,0,0) to (0,0,1)."""
        pose = np.zeros((26, 3))
        for start_idx, end_idx in _ANIMAL_LIMB_CONNECTIONS:
            pose[start_idx] = [1, 0, 0]
            pose[end_idx] = [0, 0, 0]
        R = compute_limb_regularization_matrix(pose)
        rotated = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(rotated, [0, 0, 1], atol=1e-6)

    def test_zero_length_limbs_returns_identity(self):
        """All joints coincide (zero-length limbs) → identity."""
        pose = np.zeros((26, 3))
        R = compute_limb_regularization_matrix(pose)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)

    def test_opposite_direction_180_degrees(self):
        """Limb vectors along (0, 0, -1) → R maps (0,0,-1) to (0,0,1)."""
        pose = np.zeros((26, 3))
        for start_idx, end_idx in _ANIMAL_LIMB_CONNECTIONS:
            pose[start_idx] = [0, 0, -1]
            pose[end_idx] = [0, 0, 0]
        R = compute_limb_regularization_matrix(pose)
        rotated = R @ np.array([0.0, 0.0, -1.0])
        np.testing.assert_allclose(rotated, [0, 0, 1], atol=1e-6)

    def test_result_is_valid_rotation(self):
        """R must satisfy R @ R.T ≈ I and det(R) ≈ 1 for arbitrary input."""
        rng = np.random.RandomState(123)
        pose = rng.randn(26, 3).astype("float64")
        R = compute_limb_regularization_matrix(pose)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)

    def test_custom_limb_connections(self):
        """Accepts custom limb connection list."""
        pose = np.zeros((10, 3))
        pose[0] = [0, 1, 0]
        pose[1] = [0, 0, 0]
        R = compute_limb_regularization_matrix(pose, limb_connections=[(0, 1)])
        # (0,1,0) should be rotated to (0,0,1)
        rotated = R @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(rotated, [0, 0, 1], atol=1e-6)


class TestApplyLimbRegularization:
    def test_identity_unchanged(self):
        rng = np.random.RandomState(42)
        pose = rng.randn(26, 3).astype("float64")
        result = apply_limb_regularization(pose, np.eye(3))
        np.testing.assert_allclose(result, pose, atol=1e-12)

    def test_known_rotation(self):
        """90° rotation around z-axis: (x,y,z) → (-y,x,z)."""
        R_z90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype="float64")
        pose = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = apply_limb_regularization(pose, R_z90)
        np.testing.assert_allclose(result[0], [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(result[1], [-1, 0, 0], atol=1e-12)


# =========================================================================
# Unit tests — post-processors
# =========================================================================


class TestHumanPostProcessor:
    def test_no_camera_rotation(self):
        """Without rotation, root is zeroed and pose_world == pose_3d."""
        pp = HumanPostProcessor()
        torch.manual_seed(0)
        raw = torch.randn(1, 1, 17, 3)
        pose_3d, pose_world = pp(raw, camera_rotation=None)

        assert pose_3d.shape == (17, 3)
        assert pose_world.shape == (17, 3)
        np.testing.assert_allclose(pose_3d[0], [0, 0, 0], atol=1e-7)
        np.testing.assert_allclose(pose_3d, pose_world)

    def test_with_camera_rotation(self):
        """With rotation, root is zeroed and min(world_z) == 0."""
        pp = HumanPostProcessor()
        torch.manual_seed(1)
        raw = torch.randn(1, 1, 17, 3)
        pose_3d, pose_world = pp(raw, camera_rotation=_DEFAULT_CAM_ROTATION)

        np.testing.assert_allclose(pose_3d[0], [0, 0, 0], atol=1e-7)
        assert np.min(pose_world[:, 2]) == pytest.approx(0.0, abs=1e-6)
        # Rotation changes the pose
        assert not np.allclose(pose_3d, pose_world)

    def test_mutates_input_tensor(self):
        """The processor zeroes the root joint in-place on the input tensor."""
        pp = HumanPostProcessor()
        raw = torch.ones(1, 1, 17, 3)
        assert raw[0, 0, 0, 0].item() == 1.0  # root is non-zero before

        pp(raw, camera_rotation=None)

        # raw_output[:, :, 0, :] = 0 is applied in-place
        assert torch.all(raw[0, 0, 0] == 0).item()


class TestAnimalPostProcessor:
    def test_basic_output_shape(self):
        pp = AnimalPostProcessor()
        raw = torch.randn(1, 1, 26, 3)
        pose_3d, pose_world = pp(raw, camera_rotation=None)
        assert pose_3d.shape == (26, 3)
        assert pose_world.shape == (26, 3)

    def test_ignores_camera_rotation(self):
        """camera_rotation is accepted but ignored."""
        pp = AnimalPostProcessor()
        raw = torch.randn(1, 1, 26, 3)

        _, world_none = pp(raw, camera_rotation=None)
        _, world_rot = pp(raw, camera_rotation=_DEFAULT_CAM_ROTATION)

        np.testing.assert_allclose(world_none, world_rot, atol=1e-7)

    def test_all_zero_pose(self):
        """All-zero pose → limb reg returns identity → world == 3d."""
        pp = AnimalPostProcessor()
        raw = torch.zeros(1, 1, 26, 3)
        pose_3d, pose_world = pp(raw, camera_rotation=None)

        np.testing.assert_allclose(pose_3d, 0.0)
        np.testing.assert_allclose(pose_world, 0.0)


# =========================================================================
# Unit tests — _default_components
# =========================================================================


class TestDefaultComponents:
    def test_human(self):
        est, pp = _default_components(FMPose3DConfig())
        assert isinstance(est, HRNetEstimator)
        assert isinstance(pp, HumanPostProcessor)

    def test_animal(self):
        est, pp = _default_components(FMPose3DConfig(model_type="fmpose3d_animals"))
        assert isinstance(est, SuperAnimalEstimator)
        assert isinstance(pp, AnimalPostProcessor)


# =========================================================================
# Unit tests — FMPose3DInference construction
# =========================================================================


class TestFMPose3DInferenceInit:
    def test_default_human(self, human_api):
        assert human_api.model_cfg.model_type == "fmpose3d"
        assert human_api._joints_left == [4, 5, 6, 11, 12, 13]
        assert human_api._joints_right == [1, 2, 3, 14, 15, 16]
        assert human_api._root_joint == 0
        assert human_api._pad == 0
        assert isinstance(human_api._estimator_2d, HRNetEstimator)
        assert isinstance(human_api._postprocessor, HumanPostProcessor)
        assert human_api.inference_cfg.test_augmentation is True

    def test_for_animals(self, animal_api):
        assert animal_api.model_cfg.model_type == "fmpose3d_animals"
        assert animal_api.model_cfg.n_joints == 26
        assert isinstance(animal_api._estimator_2d, SuperAnimalEstimator)
        assert isinstance(animal_api._postprocessor, AnimalPostProcessor)
        assert animal_api.inference_cfg.test_augmentation is False

    def test_custom_component_injection(self):
        """estimator_2d and postprocessor kwargs override defaults."""
        custom_est = MagicMock()
        custom_pp = MagicMock()
        api = FMPose3DInference(
            device="cpu",
            estimator_2d=custom_est,
            postprocessor=custom_pp,
        )
        assert api._estimator_2d is custom_est
        assert api._postprocessor is custom_pp

    @pytest.mark.parametrize(
        "frames,expected_pad",
        [(1, 0), (3, 1), (5, 2), (9, 4)],
    )
    def test_resolve_pad(self, frames, expected_pad):
        api = FMPose3DInference(
            model_cfg=FMPose3DConfig(frames=frames),
            device="cpu",
        )
        assert api._pad == expected_pad


# =========================================================================
# Unit tests — _ingest_input
# =========================================================================


class TestIngestInput:
    @pytest.fixture
    def api(self) -> FMPose3DInference:
        return FMPose3DInference(device="cpu")

    # --- happy paths ---

    def test_single_frame_array(self, api):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = api._ingest_input(frame)
        assert result.frames.shape == (1, 480, 640, 3)
        assert result.image_size == (480, 640)

    def test_batch_array(self, api):
        frames = np.random.randint(0, 255, (5, 480, 640, 3), dtype=np.uint8)
        result = api._ingest_input(frames)
        assert result.frames.shape == (5, 480, 640, 3)
        assert result.image_size == (480, 640)

    def test_list_of_arrays(self, api):
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        result = api._ingest_input(frames)
        assert result.frames.shape == (3, 64, 64, 3)

    def test_single_image_path_str(self, api, tmp_path):
        img = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), img)

        result = api._ingest_input(str(path))
        assert result.frames.shape == (1, 100, 120, 3)
        assert result.image_size == (100, 120)

    def test_single_image_path_object(self, api, tmp_path):
        """pathlib.Path objects are accepted."""
        img = np.random.randint(0, 255, (100, 120, 3), dtype=np.uint8)
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), img)

        result = api._ingest_input(path)  # Path, not str
        assert result.frames.shape == (1, 100, 120, 3)

    def test_directory_of_images(self, api, tmp_path):
        for i in range(3):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"img_{i:03d}.png"), img)

        result = api._ingest_input(str(tmp_path))
        assert result.frames.shape == (3, 100, 100, 3)

    def test_list_of_path_strings(self, api, tmp_path):
        paths = []
        for i in range(2):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            p = tmp_path / f"img_{i}.png"
            cv2.imwrite(str(p), img)
            paths.append(str(p))

        result = api._ingest_input(paths)
        assert result.frames.shape == (2, 100, 100, 3)

    # --- error cases ---

    def test_2d_array_raises(self, api):
        with pytest.raises(ValueError, match=r"3 .* or 4 .* dims"):
            api._ingest_input(np.zeros((100, 100)))

    def test_5d_array_raises(self, api):
        with pytest.raises(ValueError, match=r"3 .* or 4 .* dims"):
            api._ingest_input(np.zeros((1, 1, 100, 100, 3)))

    def test_empty_list_raises(self, api):
        with pytest.raises(ValueError, match="Empty source list"):
            api._ingest_input([])

    def test_nonexistent_path_raises(self, api):
        with pytest.raises(FileNotFoundError):
            api._ingest_input("/nonexistent/path/image.png")

    def test_video_path_raises(self, api, tmp_path):
        video = tmp_path / "clip.mp4"
        video.touch()
        with pytest.raises(NotImplementedError, match="Video input"):
            api._ingest_input(str(video))

    def test_unsupported_element_type_raises(self, api):
        with pytest.raises(TypeError, match="Unsupported element type"):
            api._ingest_input([123, 456])

    def test_empty_directory_raises(self, api, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="No image files"):
            api._ingest_input(str(empty))

    def test_corrupt_image_raises(self, api, tmp_path):
        corrupt = tmp_path / "corrupt.png"
        corrupt.write_text("this is not a valid image")
        with pytest.raises(FileNotFoundError, match="Could not read image"):
            api._ingest_input(str(corrupt))


# =========================================================================
# Unit tests — _load_weights error paths
# =========================================================================


class TestLoadWeightsErrors:
    def test_empty_path_raises(self):
        api = FMPose3DInference(model_weights_path="", device="cpu")
        api._model_3d = _ZeroVelocityModel()
        with pytest.raises(ValueError, match="No model weights path"):
            api._load_weights()

    def test_nonexistent_file_raises(self):
        api = FMPose3DInference(
            model_weights_path="/nonexistent/weights.pth",
            device="cpu",
        )
        api._model_3d = _ZeroVelocityModel()
        with pytest.raises(ValueError, match="Model weights file not found"):
            api._load_weights()

    def test_model_not_initialized_raises(self, tmp_path):
        dummy = tmp_path / "dummy.pth"
        torch.save({}, str(dummy))

        api = FMPose3DInference(model_weights_path=str(dummy), device="cpu")
        # _model_3d is None by default → should raise
        with pytest.raises(ValueError, match="Model not initialised"):
            api._load_weights()


# =========================================================================
# Unit tests — pose_3d input handling & edge cases
# =========================================================================


class TestPose3DValidation:
    """Tests for ``FMPose3DInference.pose_3d`` input validation and behaviour."""

    def test_1d_keypoints_raises(self, ready_human_api):
        with pytest.raises(ValueError, match="3 or 4 dims"):
            ready_human_api.pose_3d(np.zeros((34,)), image_size=(480, 640))

    def test_2d_keypoints_raises(self, ready_human_api):
        with pytest.raises(ValueError, match="3 or 4 dims"):
            ready_human_api.pose_3d(np.zeros((17, 2)), image_size=(480, 640))

    def test_5d_keypoints_raises(self, ready_human_api):
        with pytest.raises(ValueError, match="3 or 4 dims"):
            ready_human_api.pose_3d(np.zeros((1, 1, 1, 17, 2)), image_size=(480, 640))

    def test_3d_input_works(self, ready_human_api):
        kpts = np.random.randn(1, 17, 2).astype("float32")
        result = ready_human_api.pose_3d(kpts, image_size=(480, 640), seed=42)
        assert result.poses_3d.shape == (1, 17, 3)

    def test_4d_takes_first_person(self, ready_human_api):
        """4D input (num_persons, num_frames, J, 2) uses first person."""
        kpts_4d = np.random.randn(3, 2, 17, 2).astype("float32")
        kpts_3d = kpts_4d[0]  # first person → (2, 17, 2)

        r4d = ready_human_api.pose_3d(kpts_4d, image_size=(480, 640), seed=42)
        r3d = ready_human_api.pose_3d(kpts_3d, image_size=(480, 640), seed=42)

        np.testing.assert_allclose(r4d.poses_3d, r3d.poses_3d, atol=1e-6)

    def test_zero_frames_raises(self, ready_human_api):
        """Zero-frame input raises (np.stack on empty list)."""
        with pytest.raises(ValueError):
            ready_human_api.pose_3d(
                np.zeros((0, 17, 2), dtype="float32"),
                image_size=(480, 640),
            )

    def test_multiple_frames(self, ready_human_api):
        kpts = np.random.randn(5, 17, 2).astype("float32")
        result = ready_human_api.pose_3d(kpts, image_size=(480, 640), seed=42)
        assert result.poses_3d.shape == (5, 17, 3)
        assert result.poses_3d_world.shape == (5, 17, 3)

    def test_reproducibility_with_seed(self, ready_human_api):
        kpts = np.random.randn(2, 17, 2).astype("float32")
        r1 = ready_human_api.pose_3d(kpts, image_size=(480, 640), seed=42)
        r2 = ready_human_api.pose_3d(kpts, image_size=(480, 640), seed=42)
        np.testing.assert_allclose(r1.poses_3d, r2.poses_3d)
        np.testing.assert_allclose(r1.poses_3d_world, r2.poses_3d_world)

    def test_different_seeds_differ(self, ready_human_api):
        kpts = np.random.randn(1, 17, 2).astype("float32")
        r1 = ready_human_api.pose_3d(kpts, image_size=(480, 640), seed=1)
        r2 = ready_human_api.pose_3d(kpts, image_size=(480, 640), seed=2)
        assert not np.allclose(r1.poses_3d, r2.poses_3d)

    def test_progress_callback(self, ready_human_api):
        calls: list[tuple[int, int]] = []
        kpts = np.random.randn(3, 17, 2).astype("float32")
        ready_human_api.pose_3d(
            kpts,
            image_size=(480, 640),
            progress=lambda cur, tot: calls.append((cur, tot)),
        )
        assert calls == [(0, 3), (1, 3), (2, 3), (3, 3)]

    def test_tta_path_produces_output(self):
        """Test-time augmentation (flip) path produces correct shapes."""
        api = _make_ready_api("fmpose3d", test_augmentation=True)
        kpts = np.random.randn(1, 17, 2).astype("float32")
        result = api.pose_3d(kpts, image_size=(480, 640), seed=42)
        assert result.poses_3d.shape == (1, 17, 3)

    def test_animal_api_shapes(self):
        """Animal pipeline produces 26-joint output."""
        api = _make_ready_api("fmpose3d_animals")
        kpts = np.random.randn(1, 26, 2).astype("float32")
        result = api.pose_3d(kpts, image_size=(480, 640), seed=42)
        assert result.poses_3d.shape == (1, 26, 3)
        assert result.poses_3d_world.shape == (1, 26, 3)

    def test_predict_end_to_end_with_mock_estimator(self):
        """predict() chains prepare_2d → pose_3d correctly."""
        api = _make_ready_api("fmpose3d", test_augmentation=False)

        mock_kpts = np.random.randn(1, 1, 17, 2).astype("float32")
        mock_scores = np.ones((1, 1, 17), dtype="float32")
        api._estimator_2d = MagicMock()
        api._estimator_2d.predict.return_value = (mock_kpts, mock_scores)
        api._estimator_2d.setup_runtime = MagicMock()

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = api.predict(frame, seed=42)

        assert isinstance(result, Pose3DResult)
        assert result.poses_3d.shape == (1, 17, 3)
        api._estimator_2d.predict.assert_called_once()


# =========================================================================
# Unit tests — dataclasses
# =========================================================================


class TestDataclasses:
    def test_pose2d_result(self):
        kpts = np.random.randn(1, 10, 17, 2)
        scores = np.random.randn(1, 10, 17)
        result = Pose2DResult(keypoints=kpts, scores=scores, image_size=(480, 640))
        assert result.keypoints is kpts
        assert result.scores is scores
        assert result.image_size == (480, 640)

    def test_pose2d_result_default_image_size(self):
        result = Pose2DResult(
            keypoints=np.zeros((1, 1, 17, 2)),
            scores=np.zeros((1, 1, 17)),
        )
        assert result.image_size == (0, 0)

    def test_pose3d_result(self):
        p3d = np.random.randn(10, 17, 3)
        pw = np.random.randn(10, 17, 3)
        result = Pose3DResult(poses_3d=p3d, poses_3d_world=pw)
        assert result.poses_3d is p3d
        assert result.poses_3d_world is pw


# =========================================================================
# Unit tests — SuperAnimalEstimator.predict (mocked DLC)
# =========================================================================


class TestSuperAnimalPrediction:
    def test_predict_returns_zeros_when_no_bodyparts(self):
        """When DLC detects nothing, keypoints are zero-filled."""
        pytest.importorskip("deeplabcut")
        estimator = SuperAnimalEstimator()
        frames = np.random.randint(0, 255, (2, 64, 64, 3), dtype=np.uint8)

        with patch(
            "deeplabcut.pose_estimation_pytorch.apis.superanimal_analyze_images",
        ) as mock_fn:
            mock_fn.return_value = {"frame.png": {"bodyparts": None}}
            kpts, scores = estimator.predict(frames)

        assert kpts.shape == (1, 2, 26, 2)
        np.testing.assert_allclose(kpts, 0.0)
        assert scores.shape == (1, 2, 26)
        np.testing.assert_allclose(scores, 1.0)

    def test_predict_maps_valid_bodyparts(self):
        """Valid DLC bodyparts are mapped to Animal3D layout."""
        pytest.importorskip("deeplabcut")
        estimator = SuperAnimalEstimator()
        frames = np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8)

        # Synthetic bodyparts: 1 individual, 40 keypoints, (x, y, score).
        fake_bp = np.arange(120, dtype="float32").reshape(1, 40, 3)

        with patch(
            "deeplabcut.pose_estimation_pytorch.apis.superanimal_analyze_images",
        ) as mock_fn:
            mock_fn.return_value = {"frame.png": {"bodyparts": fake_bp}}
            kpts, scores = estimator.predict(frames)

        assert kpts.shape == (1, 1, 26, 2)
        assert scores.shape == (1, 1, 26)
        # target[24] ← source[0] → (0*3, 0*3+1) = (0.0, 1.0)
        np.testing.assert_allclose(kpts[0, 0, 24], fake_bp[0, 0, :2])


# =========================================================================
# Functional tests — human pipeline
# =========================================================================


@pytest.mark.functional
@pytest.mark.skipif(
    not _human_pipeline_available,
    reason="Human pretrained weights or test image not found",
)
class TestHumanPipeline:
    """End-to-end integration test for the human (17-joint H36M) pipeline."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        """Run the full pipeline once and cache all results."""
        from PIL import Image

        api = FMPose3DInference(
            model_weights_path=str(HUMAN_WEIGHTS),
            device="cpu",
        )

        # 2D estimation
        result_2d = api.prepare_2d(source=str(HUMAN_TEST_IMAGE))

        # Image size
        img = Image.open(str(HUMAN_TEST_IMAGE))
        w, h = img.size
        image_size = (h, w)

        # 3D lifting — two runs with the same seed for reproducibility check
        result_3d_a = api.pose_3d(
            result_2d.keypoints, image_size=image_size, seed=42,
        )
        result_3d_b = api.pose_3d(
            result_2d.keypoints, image_size=image_size, seed=42,
        )

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


# =========================================================================
# Functional tests — animal pipeline
# =========================================================================


@pytest.mark.functional
@pytest.mark.skipif(
    not _animal_pipeline_available,
    reason="Animal pretrained weights, test image, or DeepLabCut not available",
)
class TestAnimalPipeline:
    """End-to-end integration test for the animal (26-joint Animal3D) pipeline."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        """Run the full pipeline once and cache all results."""
        from PIL import Image

        api = FMPose3DInference.for_animals(
            model_weights_path=str(ANIMAL_WEIGHTS),
            device="cpu",
        )

        result_2d = api.prepare_2d(source=str(ANIMAL_TEST_IMAGE))

        img = Image.open(str(ANIMAL_TEST_IMAGE))
        w, h = img.size
        image_size = (h, w)

        result_3d_a = api.pose_3d(
            result_2d.keypoints, image_size=image_size, seed=42,
        )
        result_3d_b = api.pose_3d(
            result_2d.keypoints, image_size=image_size, seed=42,
        )

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
        """Poses should not be excessively large (basic sanity)."""
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
