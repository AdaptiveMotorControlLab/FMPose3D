"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""


from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence, Tuple, Union

import numpy as np
import torch

from fmpose3d.common.camera import camera_to_world, normalize_screen_coordinates
from fmpose3d.common.utils import euler_sample
from fmpose3d.common.config import (
    FMPose3DConfig,
    HRNetConfig,
    InferenceConfig,
    SupportedModel,
    SuperAnimalConfig,
)
from fmpose3d.models import get_model

#: Progress callback signature: ``(current_step, total_steps) -> None``.
ProgressCallback = Callable[[int, int], None]


#: HuggingFace repository hosting the official FMPose3D checkpoints.
_HF_REPO_ID: str = "deruyter92/fmpose_temp"

# Default camera-to-world rotation quaternion (from the demo script).
_DEFAULT_CAM_ROTATION = np.array(
    [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
    dtype="float32",
)


# ---------------------------------------------------------------------------
# 2D pose estimator
# ---------------------------------------------------------------------------


class HRNetEstimator:
    """Default 2D pose estimator: HRNet + YOLO, with COCO→H36M conversion.

    Thin wrapper around :class:`~fmpose3d.lib.hrnet.api.HRNetPose2d` that
    adds the COCO → H36M keypoint conversion expected by the 3D lifter.

    Parameters
    ----------
    cfg : HRNetConfig
        Estimator settings (``det_dim``, ``num_persons``, …).
    """

    def __init__(self, cfg: HRNetConfig | None = None) -> None:
        self.cfg = cfg or HRNetConfig()
        self._model = None

    def setup_runtime(self) -> None:
        """Load YOLO + HRNet models (safe to call more than once)."""
        if self._model is not None:
            return

        from fmpose3d.lib.hrnet.hrnet import HRNetPose2d

        self._model = HRNetPose2d(
            det_dim=self.cfg.det_dim,
            num_persons=self.cfg.num_persons,
            thred_score=self.cfg.thred_score,
            hrnet_cfg_file=self.cfg.hrnet_cfg_file,
            hrnet_weights_path=self.cfg.hrnet_weights_path,
        )
        self._model.setup()

    def predict(
        self, frames: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate 2D keypoints from image frames and return in H36M format.

        Parameters
        ----------
        frames : ndarray
            BGR image frames, shape ``(N, H, W, C)``.

        Returns
        -------
        keypoints : ndarray
            H36M-format 2D keypoints, shape ``(num_persons, N, 17, 2)``.
        scores : ndarray
            Per-joint confidence scores, shape ``(num_persons, N, 17)``.
        valid_frames_mask : ndarray
            Boolean mask indicating which frames contain at least one
            valid detection, shape ``(N,)``.
        """
        from fmpose3d.lib.preprocess import h36m_coco_format, revise_kpts

        self.setup_runtime()

        keypoints, scores = self._model.predict(frames)

        keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
        keypoints, scores = self._validate_predictions(
            keypoints, scores, num_frames=frames.shape[0],
        )
        valid_frames_mask = self._compute_valid_frames_mask(keypoints, scores)

        # NOTE: revise_kpts is computed for consistency but is NOT applied
        # to the returned keypoints, matching the demo script behaviour.
        _revised = revise_kpts(keypoints, scores, valid_frames)  # noqa: F841
        return keypoints, scores, valid_frames_mask

    def _validate_predictions(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        *,
        num_frames: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and normalise HRNet/H36M predictions."""
        num_joints = 17

        keypoints = np.asarray(keypoints, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)

        if keypoints.shape[0] == 0:
            # h36m_coco_format can drop all persons when all frames are empty.
            return (
                np.zeros((1, num_frames, num_joints, 2), dtype=np.float32),
                np.zeros((1, num_frames, num_joints), dtype=np.float32),
            )

        if keypoints.ndim != 4 or keypoints.shape[-2:] != (num_joints, 2):
            raise ValueError(
                f"Invalid HRNet keypoints shape {keypoints.shape}; "
                f"expected (num_persons, num_frames, {num_joints}, 2)."
            )
        if scores.ndim != 3 or scores.shape[-1] != num_joints:
            raise ValueError(
                f"Invalid HRNet scores shape {scores.shape}; "
                f"expected (num_persons, num_frames, {num_joints})."
            )
        if keypoints.shape[:2] != scores.shape[:2]:
            raise ValueError(
                "HRNet keypoints/scores leading dimensions do not match: "
                f"{keypoints.shape[:2]} vs {scores.shape[:2]}."
            )
        if keypoints.shape[1] != num_frames:
            raise ValueError(
                f"HRNet frame count mismatch: got {keypoints.shape[1]}, "
                f"expected {num_frames}."
            )
        return keypoints, scores

    @staticmethod
    def _compute_valid_frames_mask(
        keypoints: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        """Return frame-level validity mask from estimator outputs."""
        safe_scores = np.nan_to_num(scores, nan=0.0)
        has_score = np.any(safe_scores > 0, axis=-1)  # (num_persons, num_frames)

        safe_kpts = np.nan_to_num(np.abs(keypoints), nan=0.0)
        has_kpt = np.any(safe_kpts > 0, axis=(-1, -2))  # (num_persons, num_frames)
        return np.any(has_score | has_kpt, axis=0)


# Quadruped80K → Animal3D (26 keypoints) mapping table.
# -1 entries are filled by linear interpolation (see _INTERPOLATION_RULES).
_QUADRUPED80K_TO_ANIMAL3D: list[int] = [
    10, 5, -1, 26, 29, 30, 35, 22, 24, 27, 31, 32,
    -1, -1, 25, 28, 33, 34, 15, 23, 11, 6, 4, 3, 0, -1,
]

# For each -1 target index, the two source indices to average.
_INTERPOLATION_RULES: dict[int, tuple[int, int]] = {
    2: (3, 4),
    12: (24, 19),
    13: (27, 19),
    25: (22, 23),
}


class SuperAnimalEstimator:
    """2D pose estimator for animals: DeepLabCut SuperAnimal.

    Uses the ``superanimal_analyze_images`` API from DeepLabCut to
    predict quadruped keypoints, then maps them to the 26-joint
    Animal3D layout expected by the ``fmpose3d_animals`` 3D lifter.

    Parameters
    ----------
    cfg : SuperAnimalConfig
        Estimator settings (``superanimal_name``, ``max_individuals``, ...).
    """

    def __init__(self, cfg: SuperAnimalConfig | None = None) -> None:
        self.cfg = cfg or SuperAnimalConfig()

    def setup_runtime(self) -> None:
        """No-op -- DeepLabCut loads models on first call."""

    def predict(
        self, frames: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate 2D keypoints from image frames in Animal3D format.

        The method writes *frames* to a temporary directory, runs
        ``superanimal_analyze_images``, and maps the resulting
        quadruped80K keypoints to Animal3D's 26-keypoint layout.

        Parameters
        ----------
        frames : ndarray
            BGR image frames, shape ``(N, H, W, C)``.

        Returns
        -------
        keypoints : ndarray
            Animal3D-format 2D keypoints, shape ``(1, N, 26, 2)``.
            The first axis is always 1 (single individual).
        scores : ndarray
            Mapped per-joint confidence scores,
            shape ``(1, N, 26)``.
        valid_frames_mask : ndarray
            Boolean mask indicating which frames contain at least one
            valid detection, shape ``(N,)``.
        """
        import cv2
        import tempfile
        from deeplabcut.pose_estimation_pytorch.apis import (
            superanimal_analyze_images,
        )

        cfg = self.cfg
        num_frames = frames.shape[0]
        all_mapped: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write each frame as an image so DLC can read it.
            paths: list[str] = []
            for idx in range(num_frames):
                p = str(Path(tmpdir) / f"frame_{idx:06d}.png")
                cv2.imwrite(p, frames[idx])
                paths.append(p)

            # Run DeepLabCut on each frame individually.
            for img_path in paths:
                predictions = superanimal_analyze_images(
                    superanimal_name=cfg.superanimal_name,
                    model_name=cfg.sa_model_name,
                    detector_name=cfg.detector_name,
                    images=img_path,
                    max_individuals=cfg.max_individuals,
                    out_folder=tmpdir,
                )
                # predictions: {image_path: {"bodyparts": (N_ind, K, 3), ...}}
                payload = predictions.get(img_path) if isinstance(predictions, dict) else None
                if payload is None and isinstance(predictions, dict) and len(predictions) == 1:
                    payload = next(iter(predictions.values()))

                bodyparts = None if payload is None else payload.get("bodyparts")
                bodyparts = None if bodyparts is None else np.asarray(bodyparts)
                if bodyparts is None or bodyparts.shape[0] == 0:
                    # No detection -- fill with zeros and zero confidence.
                    all_mapped.append(np.zeros((1, 26, 2), dtype=np.float32))
                    all_scores.append(np.zeros((1, 26), dtype=np.float32))
                    continue

                xy = bodyparts[..., :2]   # (N_ind, K, 2)
                conf = bodyparts[..., 2]  # (N_ind, K)
                mapped = self._map_keypoints(xy)
                mapped_scores = self._map_scores(conf)

                # Take only the first individual.
                all_mapped.append(mapped[:1])
                all_scores.append(mapped_scores[:1])

        # Stack along frame axis → (1, N, 26, 2)
        kpts = np.stack(all_mapped, axis=1)  # (1, N, 26, 2)
        scores = np.stack(all_scores, axis=1)  # (1, N, 26)
        kpts, scores = self._validate_predictions(kpts, scores, num_frames=num_frames)
        valid_frames_mask = self._compute_valid_frames_mask(kpts, scores)
        return kpts, scores, valid_frames_mask

    # ------------------------------------------------------------------ #

    @staticmethod
    def _map_keypoints(xy: np.ndarray) -> np.ndarray:
        """Map keypoints from the quadruped80K dataset format (see: DeepLabCut model zoo:
        https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped)
        to the FMPose3D 3d animal 26-joint layout.

        Parameters
        ----------
        xy : ndarray
            Source keypoints, shape ``(num_individuals, K_src, 2)``.

        Returns
        -------
        mapped : ndarray
            Mapped keypoints, shape ``(num_individuals, 26, 2)``.
        """
        num_ind, num_src, _ = xy.shape
        num_tgt = len(_QUADRUPED80K_TO_ANIMAL3D)
        mapped = np.full((num_ind, num_tgt, 2), np.nan, dtype="float32")

        for tgt_idx, src_idx in enumerate(_QUADRUPED80K_TO_ANIMAL3D):
            if src_idx != -1 and src_idx < num_src:
                mapped[:, tgt_idx, :] = xy[:, src_idx, :]
            elif src_idx == -1 and tgt_idx in _INTERPOLATION_RULES:
                s1, s2 = _INTERPOLATION_RULES[tgt_idx]
                if s1 < num_src and s2 < num_src:
                    mapped[:, tgt_idx, :] = (xy[:, s1, :] + xy[:, s2, :]) / 2.0

        return mapped

    @staticmethod
    def _map_scores(conf: np.ndarray) -> np.ndarray:
        """Map confidence scores from quadruped80K to Animal3D layout."""
        num_ind, num_src = conf.shape
        num_tgt = len(_QUADRUPED80K_TO_ANIMAL3D)
        mapped = np.full((num_ind, num_tgt), np.nan, dtype=np.float32)

        for tgt_idx, src_idx in enumerate(_QUADRUPED80K_TO_ANIMAL3D):
            if src_idx != -1 and src_idx < num_src:
                mapped[:, tgt_idx] = conf[:, src_idx]
            elif src_idx == -1 and tgt_idx in _INTERPOLATION_RULES:
                s1, s2 = _INTERPOLATION_RULES[tgt_idx]
                if s1 < num_src and s2 < num_src:
                    mapped[:, tgt_idx] = (conf[:, s1] + conf[:, s2]) / 2.0

        return mapped

    def _validate_predictions(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        *,
        num_frames: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and normalise SuperAnimal predictions."""
        num_joints = 26
        keypoints = np.asarray(keypoints, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)

        if keypoints.shape[0] == 0:
            return (
                np.zeros((1, num_frames, num_joints, 2), dtype=np.float32),
                np.zeros((1, num_frames, num_joints), dtype=np.float32),
            )

        if keypoints.ndim != 4 or keypoints.shape[-2:] != (num_joints, 2):
            raise ValueError(
                f"Invalid SuperAnimal keypoints shape {keypoints.shape}; "
                f"expected (num_individuals, num_frames, {num_joints}, 2)."
            )
        if scores.ndim != 3 or scores.shape[-1] != num_joints:
            raise ValueError(
                f"Invalid SuperAnimal scores shape {scores.shape}; "
                f"expected (num_individuals, num_frames, {num_joints})."
            )
        if keypoints.shape[:2] != scores.shape[:2]:
            raise ValueError(
                "SuperAnimal keypoints/scores leading dimensions do not match: "
                f"{keypoints.shape[:2]} vs {scores.shape[:2]}."
            )
        if keypoints.shape[1] != num_frames:
            raise ValueError(
                f"SuperAnimal frame count mismatch: got {keypoints.shape[1]}, "
                f"expected {num_frames}."
            )

        # Normalise unknown values to zeros so downstream code can treat these
        # joints as invalid via score==0 while retaining shape stability.
        keypoints = np.nan_to_num(keypoints, nan=0.0)
        scores = np.nan_to_num(scores, nan=0.0)
        return keypoints, scores

    @staticmethod
    def _compute_valid_frames_mask(
        keypoints: np.ndarray, scores: np.ndarray
    ) -> np.ndarray:
        """Return frame-level validity mask from estimator outputs."""
        safe_scores = np.nan_to_num(scores, nan=0.0)
        has_score = np.any(safe_scores > 0, axis=-1)  # (num_persons, num_frames)

        safe_kpts = np.nan_to_num(np.abs(keypoints), nan=0.0)
        has_kpt = np.any(safe_kpts > 0, axis=(-1, -2))  # (num_persons, num_frames)
        return np.any(has_score | has_kpt, axis=0)


# ---------------------------------------------------------------------------
# Limb regularisation (animal post-processing)
# ---------------------------------------------------------------------------


# Limb connections used for vertical alignment (thigh → knee).
_ANIMAL_LIMB_CONNECTIONS: list[tuple[int, int]] = [
    (8, 14),   # left_front_thigh → left_front_knee
    (9, 15),   # right_front_thigh → right_front_knee
    (10, 16),  # left_back_thigh → left_back_knee
    (11, 17),  # right_back_thigh → right_back_knee
]


def compute_limb_regularization_matrix(
    pose_3d: np.ndarray,
    limb_connections: list[tuple[int, int]] = _ANIMAL_LIMB_CONNECTIONS,
) -> np.ndarray:
    """Compute a rotation that aligns average limb direction to vertical.

    The limb vectors are taken as *proximal - distal* (pointing upward)
    and averaged.  A Rodrigues rotation is computed to map the result
    onto ``(0, 0, 1)``.

    This is primarily intended for visualization and canonicalization of
    upright poses.

    .. note:: **Limitations**

       * The function assumes a stable "up" limb direction and may produce
         poor results for poses where this assumption does not hold (e.g.
         lying down, jumping, or other non-upright orientations).
       * The rotation is computed independently per frame with no temporal
         smoothing or prior, so it can be unstable across frames and may
         cause flickering in video sequences.

       If these assumptions do not match your data, consider using the raw
       predicted pose and implementing custom regularization logic suited
       to your use-case.

    Parameters
    ----------
    pose_3d : ndarray
        3D pose, shape ``(J, 3)``.
    limb_connections : list of (int, int)
        Pairs ``(start, end)`` defining each limb.

    Returns
    -------
    R : ndarray
        ``(3, 3)`` rotation matrix.
    """
    limb_vectors: list[np.ndarray] = []
    for start_idx, end_idx in limb_connections:
        vec = pose_3d[start_idx] - pose_3d[end_idx]
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            limb_vectors.append(vec / norm)

    if len(limb_vectors) == 0:
        return np.eye(3)

    avg = np.mean(limb_vectors, axis=0)
    avg = avg / (np.linalg.norm(avg) + 1e-8)
    target = np.array([0.0, 0.0, 1.0])

    v = np.cross(avg, target)
    c = np.dot(avg, target)

    if np.linalg.norm(v) < 1e-6:
        if c > 0:
            return np.eye(3)
        # Opposite -- 180-degree rotation around a perpendicular axis.
        axis = np.array([1.0, 0.0, 0.0]) if abs(avg[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = axis - avg * np.dot(axis, avg)
        axis = axis / np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)

    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))


def apply_limb_regularization(pose_3d: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply a rotation matrix to a 3D pose.

    Parameters
    ----------
    pose_3d : ndarray, shape ``(J, 3)``
    R : ndarray, shape ``(3, 3)``

    Returns
    -------
    ndarray, shape ``(J, 3)``
    """
    return (R @ pose_3d.T).T


# ---------------------------------------------------------------------------
# Post-processors
# ---------------------------------------------------------------------------


class HumanPostProcessor:
    """Post-process a raw 3D pose for the human pipeline.

    Zeros the root joint to make the pose root-relative, then
    optionally applies a ``camera_to_world`` rotation.
    """

    def __call__(
        self,
        raw_output: torch.Tensor,
        *,
        camera_rotation: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(pose_3d, pose_world)`` each of shape ``(J, 3)``.

        Parameters
        ----------
        raw_output : torch.Tensor
            Model output for one frame, shape ``(1, 1, J, 3)``.
        camera_rotation : ndarray or None
            Length-4 quaternion for ``camera_to_world``.
        """
        raw_output[:, :, 0, :] = 0  # root-relative
        pose_3d = raw_output[0, 0].cpu().detach().numpy()
        if camera_rotation is not None:
            pose_world = camera_to_world(pose_3d, R=camera_rotation, t=0)
            pose_world[:, 2] -= np.min(pose_world[:, 2])
        else:
            pose_world = pose_3d.copy()
        return pose_3d, pose_world


class AnimalPostProcessor:
    """Post-process a raw 3D pose for the animal pipeline.

    Applies limb regularisation (rotates the pose so that average limb
    direction is vertical).  No root zeroing, no ``camera_to_world``.
    """

    def __call__(
        self,
        raw_output: torch.Tensor,
        *,
        camera_rotation: np.ndarray | None,
        limb_regularization: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(pose_3d, pose_world)`` each of shape ``(J, 3)``.

        Parameters
        ----------
        raw_output : torch.Tensor
            Model output for one frame, shape ``(1, 1, J, 3)``.
        camera_rotation : ndarray or None
            Ignored (accepted for interface compatibility).
        """
        pose_3d = raw_output[0, 0].cpu().detach().numpy()
        R_reg = (
            compute_limb_regularization_matrix(pose_3d) if limb_regularization
            else np.eye(3)
        )
        pose_world = apply_limb_regularization(pose_3d, R_reg)
        return pose_3d, pose_world


# ---------------------------------------------------------------------------
# Default component resolver
# ---------------------------------------------------------------------------


def _default_components(
    model_cfg: FMPose3DConfig,
) -> tuple[
    HRNetEstimator | SuperAnimalEstimator,
    HumanPostProcessor | AnimalPostProcessor,
]:
    """Return the default ``(estimator_2d, postprocessor)`` for *model_cfg*.

    This is the **only** place in the module where ``model_type`` is
    inspected to choose pipeline components.  Adding a third pipeline
    means adding one branch here (or turning this into a registry).
    """
    if model_cfg.model_type == SupportedModel.FMPOSE3D_ANIMALS:
        return SuperAnimalEstimator(), AnimalPostProcessor()
    return HRNetEstimator(), HumanPostProcessor()


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


class Pose2DStatus(str, Enum):
    """High-level status for 2D pose estimation."""

    SUCCESS = "success"
    PARTIAL = "partial"
    EMPTY = "empty"
    INVALID = "invalid"
    UNKNOWN = "unknown"


@dataclass
class Pose2DResult:
    """Container returned by :meth:`FMPose3DInference.prepare_2d`.

    ``J`` is 17 for the human (H36M) pipeline and 26 for the animal
    (Animal3D) pipeline.
    """

    keypoints: np.ndarray
    """2D keypoints, shape ``(num_persons, num_frames, J, 2)``."""
    scores: np.ndarray
    """Per-joint confidence scores, shape ``(num_persons, num_frames, J)``."""
    image_size: tuple[int, int] = (0, 0)
    """``(height, width)`` of the source frames."""
    valid_frames_mask: np.ndarray | None = None
    """Boolean mask of frames with at least one valid detection, shape ``(N,)``."""

    @property
    def status(self) -> Pose2DStatus:
        """Prediction status derived from ``valid_frames_mask``."""
        return self.get_status_info()[0]

    @property
    def status_message(self) -> str:
        """Human-readable explanation for :attr:`status`."""
        return self.get_status_info()[1]

    def get_status_info(self) -> tuple[Pose2DStatus, str]:
        """Prediction status derived from ``valid_frames_mask``."""
        if self.valid_frames_mask is None:
            return Pose2DStatus.UNKNOWN, "No frame-validity mask provided by the estimator."
        elif not isinstance(self.valid_frames_mask, np.ndarray) or self.valid_frames_mask.ndim != 1:
            return Pose2DStatus.UNKNOWN, "invalid valid_frames_mask: must be a 1D numpy array."

        # Derive expected frame count from canonical shapes.
        if self.keypoints.ndim == 4:
            num_frames = int(self.keypoints.shape[1])
        elif self.scores.ndim == 3:
            num_frames = int(self.scores.shape[1])
        else:
            return Pose2DStatus.INVALID, "Incorrect keypoints/scores dimensions."

        if self.valid_frames_mask.shape[0] != num_frames:
            return Pose2DStatus.INVALID, "valid_frames_mask mismatches the number of frames."

        valid_count = int(np.sum(self.valid_frames_mask))
        if valid_count == 0:
            return Pose2DStatus.EMPTY, "No valid predictions in any frame."
        if valid_count < num_frames:
            return Pose2DStatus.PARTIAL, "Missing predictions in a subset of frames."
        return Pose2DStatus.SUCCESS, "Valid predictions for all frames."


@dataclass
class Pose3DResult:
    """Container returned by :meth:`FMPose3DInference.pose_3d`.

    ``J`` is 17 for the human (H36M) pipeline and 26 for the animal
    (Animal3D) pipeline.
    """

    poses_3d: np.ndarray
    """Root-relative 3D poses, shape ``(num_frames, J, 3)``."""
    poses_3d_world: np.ndarray
    """Post-processed 3D poses, shape ``(num_frames, J, 3)``.

    For human poses this contains world-coordinate poses (after
    ``camera_to_world``).  For animal poses this contains the
    limb-regularised output.
    """


#: Accepted source types for :meth:`FMPose3DInference.predict`.
#:
#: * ``str`` or ``Path`` – path to an image file or directory of images.
#: * ``np.ndarray`` – a single frame ``(H, W, C)`` or batch ``(N, H, W, C)``.
#: * ``list`` – a list of file paths or a list of ``(H, W, C)`` arrays.
Source = Union[str, Path, np.ndarray, Sequence[Union[str, Path, np.ndarray]]]


@dataclass
class _IngestedInput:
    """Normalised result of :meth:`FMPose3DInference._ingest_input`.

    Always contains a batch of BGR frames as a numpy array, regardless
    of the original source type.
    """

    frames: np.ndarray
    """BGR image frames, shape ``(N, H, W, C)``."""
    image_size: tuple[int, int]
    """``(height, width)`` of the source frames."""


# ---------------------------------------------------------------------------
# Main inference class
# ---------------------------------------------------------------------------


# FIXME @deruyter92: THIS IS TEMPORARY UNTIL WE DOWNLOAD THE WEIGHTS FROM HUGGINGFACE
SKIP_WEIGHTS_VALIDATION = object() # sentinel value to indicate that the weights should not be validated

class FMPose3DInference:
    """High-level, two-step inference API for FMPose3D.

    Supports both **human** (``model_type="fmpose3d_humans"``, 17 H36M joints)
    and **animal** (``model_type="fmpose3d_animals"``, 26 Animal3D joints)
    pipelines.  The skeleton layout, 2D estimator, and post-processing
    are chosen automatically from the model configuration.

    Typical workflow (human)::

        api = FMPose3DInference(model_weights_path="weights.pth")
        result_2d = api.prepare_2d("photo.jpg")
        result_3d = api.pose_3d(result_2d.keypoints, image_size=(H, W))

    Typical workflow (animal)::

        api = FMPose3DInference.for_animals(model_weights_path="animal_weights.pth")
        result_2d = api.prepare_2d("dog.jpg")
        result_3d = api.pose_3d(result_2d.keypoints, image_size=(H, W))

    Parameters
    ----------
    model_cfg : FMPose3DConfig, optional
        Model architecture settings (layers, channels, joints, …).
        Defaults to ``FMPose3DConfig()`` (human, 17 H36M joints).
    inference_cfg : InferenceConfig, optional
        Inference settings (sample_steps, test_augmentation, …).
        Defaults to :class:`~fmpose3d.common.config.InferenceConfig` defaults.
    model_weights_path : str
        Path to a ``.pth`` checkpoint for the 3D lifting model.
        If empty the model is created but **not** loaded with weights.
    device : str or torch.device, optional
        Compute device.  ``None`` (default) picks CUDA when available.
    estimator_2d : HRNetEstimator or SuperAnimalEstimator, optional
        2D pose estimator.  When ``None`` (default), one is created
        automatically based on ``model_cfg.model_type``.
    postprocessor : HumanPostProcessor or AnimalPostProcessor, optional
        Post-processor applied to each raw 3D pose.  When ``None``
        (default), one is created automatically based on
        ``model_cfg.model_type``.
    """

    _IMAGE_EXTENSIONS: set[str] = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    }

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_cfg: FMPose3DConfig | None = None,
        inference_cfg: InferenceConfig | None = None,
        model_weights_path: str | Path | None = None,
        device: str | torch.device | None = None,
        *,
        estimator_2d: HRNetEstimator | SuperAnimalEstimator | None = None,
        postprocessor: HumanPostProcessor | AnimalPostProcessor | None = None,
    ) -> None:
        self.model_cfg = model_cfg or FMPose3DConfig()
        self.inference_cfg = inference_cfg or InferenceConfig()
        self.model_weights_path = model_weights_path

        # Validate model weights path (download if needed)
        self._resolve_model_weights_path()

        # Skeleton configuration from the model config.
        self._joints_left: list[int] = list(self.model_cfg.joints_left)
        self._joints_right: list[int] = list(self.model_cfg.joints_right)
        self._root_joint: int = self.model_cfg.root_joint

        # Pipeline components -- resolved from config or overridden explicitly.
        default_est, default_pp = _default_components(self.model_cfg)
        self._estimator_2d: HRNetEstimator | SuperAnimalEstimator = (
            estimator_2d or default_est
        )
        self._postprocessor: HumanPostProcessor | AnimalPostProcessor = (
            postprocessor or default_pp
        )

        # Resolve device and padding configuration
        self._device: torch.device | None = self._resolve_device(device)
        self._pad: int = self._resolve_pad()

        # Lazy-loaded 3D lifting model (populated by setup_runtime)
        self._model_3d: torch.nn.Module | None = None

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def for_animals(
        cls,
        model_weights_path: str | None = None,
        *,
        device: str | torch.device | None = None,
        inference_cfg: InferenceConfig | None = None,
    ) -> "FMPose3DInference":
        """Create an instance configured for **animal** pose estimation.

        Sets ``model_type="fmpose3d_animals"`` (26-joint Animal3D
        skeleton) and disables flip test-time augmentation by default,
        matching the behaviour of ``animals/demo/vis_animals.py``.

        Parameters
        ----------
        model_weights_path : str
            Path to the animal model checkpoint.
        device : str or torch.device, optional
            Compute device.
        inference_cfg : InferenceConfig, optional
            Override inference settings.  When ``None`` (default),
            ``test_augmentation`` is set to ``False``.
        """
        if inference_cfg is None:
            inference_cfg = InferenceConfig(test_augmentation=False)
        return cls(
            model_cfg=FMPose3DConfig(model_type=SupportedModel.FMPOSE3D_ANIMALS),
            inference_cfg=inference_cfg,
            model_weights_path=model_weights_path,
            device=device,
            estimator_2d=SuperAnimalEstimator(),
            postprocessor=AnimalPostProcessor(),
        )

    def setup_runtime(self) -> None:
        """Initialise all runtime components on first use.

        Called automatically when the API is used for the first time.
        Loads the 2D estimator, the 3D lifting model, and the model
        weights in sequence.
        """
        self._setup_estimator_2d()
        self._setup_model()
        self._load_weights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        source: Source,
        *,
        camera_rotation: np.ndarray | None = _DEFAULT_CAM_ROTATION,
        seed: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> Pose3DResult:
        """End-to-end prediction: 2D pose estimation → 3D lifting.

        Convenience wrapper that calls :meth:`prepare_2d` then
        :meth:`pose_3d`.

        Parameters
        ----------
        source : Source
            Input to process.  Accepts a file path (``str`` / ``Path``),
            a directory of images, a numpy array ``(H, W, C)`` for a
            single frame, ``(N, H, W, C)`` for a batch, or a list of
            paths / arrays.  See :data:`Source` for the full type.
            Video files are **not** supported and will raise
            :class:`NotImplementedError`.
        camera_rotation : ndarray or None
            Length-4 quaternion for the camera-to-world rotation.
            See :meth:`pose_3d` for details.
        seed : int or None
            Deterministic seed for the 3D sampling step.
            See :meth:`pose_3d` for details.
        progress : ProgressCallback or None
            Optional ``(current_step, total_steps)`` callback.  Forwarded
            to the :meth:`pose_3d` step (per-frame reporting).

        Returns
        -------
        Pose3DResult
            Root-relative and world-coordinate 3D poses.
        """
        result_2d = self.prepare_2d(source)
        status, status_msg = result_2d.get_status_info()
        if status in {Pose2DStatus.EMPTY, Pose2DStatus.INVALID}:
            raise ValueError(f"2D pose estimation is not usable for 3D lifting: {status.value}. {status_msg}")
        return self.pose_3d(
            result_2d.keypoints,
            result_2d.image_size,
            camera_rotation=camera_rotation,
            seed=seed,
            progress=progress,
        )

    @torch.no_grad()
    def prepare_2d(
        self,
        source: Source,
        progress: ProgressCallback | None = None,
    ) -> Pose2DResult:
        """Estimate 2D poses from images.

        For human models this uses HRNet + YOLO (17 H36M joints); for
        animal models this uses DeepLabCut SuperAnimal (26 Animal3D
        joints).  The estimator is set up lazily by
        :meth:`setup_runtime` on first call.

        Parameters
        ----------
        source : Source
            Input to process.  Accepts a file path (``str`` / ``Path``),
            a directory of images, a numpy array ``(H, W, C)`` for a
            single frame, ``(N, H, W, C)`` for a batch, or a list of
            paths / arrays.  See :data:`Source` for the full type.
        progress : ProgressCallback or None
            Optional ``(current_step, total_steps)`` callback invoked
            before and after the 2D estimation step.

        Returns
        -------
        Pose2DResult
            2D keypoints and per-joint scores.  The result also carries
            ``image_size`` so it can be forwarded directly to
            :meth:`pose_3d`.
        """
        ingested = self._ingest_input(source)
        self.setup_runtime()
        if progress:
            progress(0, 1)
        keypoints, scores, valid_frames_mask = self._estimator_2d.predict(
            ingested.frames
        )
        if progress:
            progress(1, 1)
        return Pose2DResult(
            keypoints=keypoints,
            scores=scores,
            image_size=ingested.image_size,
            valid_frames_mask=valid_frames_mask,
        )

    @torch.no_grad()
    def pose_3d(
        self,
        keypoints_2d: np.ndarray,
        image_size: tuple[int, int],
        *,
        camera_rotation: np.ndarray | None = _DEFAULT_CAM_ROTATION,
        seed: int | None = None,
        progress: ProgressCallback | None = None,
    ) -> Pose3DResult:
        """Lift 2D keypoints to 3D using the flow-matching model.

        **Human pipeline** (``model_type="fmpose3d_humans"``):
        Mirrors ``demo/vis_in_the_wild.py`` -- normalise screen
        coordinates, flip-augmented TTA, Euler ODE sampling, zero the
        root joint, ``camera_to_world``.

        **Animal pipeline** (``model_type="fmpose3d_animals"``):
        Mirrors ``animals/demo/vis_animals.py`` -- normalise screen
        coordinates, single-pass (no TTA), Euler ODE sampling, limb
        regularisation (no root zeroing, no ``camera_to_world``).

        Parameters
        ----------
        keypoints_2d : ndarray
            2D keypoints returned by :meth:`prepare_2d`.  Accepted shapes:

            * ``(num_persons, num_frames, J, 2)`` -- first person is used.
            * ``(num_frames, J, 2)`` -- treated as a single person.
        image_size : tuple of (int, int)
            ``(height, width)`` of the source image / video frames.
        camera_rotation : ndarray or None
            Length-4 quaternion for the camera-to-world rotation applied
            to produce ``poses_3d_world``.  Defaults to the rotation used
            in the official demo.  Pass ``None`` to skip the transform
            (``poses_3d_world`` will equal ``poses_3d``).  **Ignored**
            for the animal pipeline (limb regularisation is applied
            instead).
        seed : int or None
            If given, ``torch.manual_seed(seed)`` is called before
            sampling so that results are fully reproducible.
        progress : ProgressCallback or None
            Optional ``(current_step, total_steps)`` callback invoked
            after each frame is lifted to 3D.

        Returns
        -------
        Pose3DResult
            Root-relative and post-processed 3D poses.
        """
        self.setup_runtime()
        model = self._model_3d
        h, w = image_size
        steps = self.inference_cfg.sample_steps

        # Optional deterministic seeding
        if seed is not None:
            torch.manual_seed(seed)

        # Normalise input shape to (num_frames, J, 2)
        if keypoints_2d.ndim == 4:
            kpts = keypoints_2d[0]  # first person
        elif keypoints_2d.ndim == 3:
            kpts = keypoints_2d
        else:
            raise ValueError(
                f"Expected keypoints_2d with 3 or 4 dims, got {keypoints_2d.ndim}"
            )

        num_frames = kpts.shape[0]
        all_poses_3d: list[np.ndarray] = []
        all_poses_world: list[np.ndarray] = []

        if progress:
            progress(0, num_frames)

        for i in range(num_frames):
            normed = normalize_screen_coordinates(kpts[i : i + 1], w=w, h=h)
            raw_output = self._run_euler_sample(normed, model, steps)
            pose_3d_np, pose_world = self._postprocessor(
                raw_output, camera_rotation=camera_rotation,
            )
            all_poses_3d.append(pose_3d_np)
            all_poses_world.append(pose_world)

            if progress:
                progress(i + 1, num_frames)

        return Pose3DResult(
            poses_3d=np.stack(all_poses_3d, axis=0),
            poses_3d_world=np.stack(all_poses_world, axis=0),
        )

    # ------------------------------------------------------------------
    # Private helpers – sampling & post-processing
    # ------------------------------------------------------------------

    def _run_euler_sample(
        self,
        normed: np.ndarray,
        model: torch.nn.Module,
        steps: int,
    ) -> torch.Tensor:
        """Run one Euler ODE sample, optionally with flip-TTA.

        Parameters
        ----------
        normed : ndarray
            Normalised 2D keypoints for a single frame, shape
            ``(1, J, 2)``.
        model : torch.nn.Module
            The 3D lifting model.
        steps : int
            Number of Euler integration steps.

        Returns
        -------
        torch.Tensor
            Raw model output, shape ``(1, 1, J, 3)`` (after extracting
            the centre frame and adding a singleton dim).
        """
        jl = self._joints_left
        jr = self._joints_right

        if self.inference_cfg.test_augmentation:
            # Build flip-augmented conditioning pair.
            normed_flip = copy.deepcopy(normed)
            normed_flip[:, :, 0] *= -1
            normed_flip[:, jl + jr] = normed_flip[:, jr + jl]
            input_2d = np.concatenate(
                (np.expand_dims(normed, axis=0),
                 np.expand_dims(normed_flip, axis=0)),
                0,
            )  # (2, F, J, 2)
            input_2d = input_2d[np.newaxis, :, :, :, :]  # (1, 2, F, J, 2)
            input_t = torch.from_numpy(input_2d.astype("float32")).to(self.device)

            # Two independent Euler ODE runs.
            y = torch.randn(
                input_t.size(0), input_t.size(2), input_t.size(3), 3,
                device=self.device,
            )
            output_non_flip = euler_sample(input_t[:, 0], y, steps, model)

            y_flip = torch.randn(
                input_t.size(0), input_t.size(2), input_t.size(3), 3,
                device=self.device,
            )
            output_flip = euler_sample(input_t[:, 1], y_flip, steps, model)

            # Un-flip & average.
            output_flip[:, :, :, 0] *= -1
            output_flip[:, :, jl + jr, :] = output_flip[:, :, jr + jl, :]

            output = (output_non_flip + output_flip) / 2
        else:
            input_2d = normed[np.newaxis]  # (1, F, J, 2)
            input_t = torch.from_numpy(input_2d.astype("float32")).to(self.device)
            y = torch.randn(
                input_t.size(0), input_t.size(1), input_t.size(2), 3,
                device=self.device,
            )
            output = euler_sample(input_t, y, steps, model)

        # Extract the centre frame → (1, 1, J, 3).
        return output[0:, self._pad].unsqueeze(1)

    # ------------------------------------------------------------------
    # Private helpers – device & padding
    # ------------------------------------------------------------------

    def _resolve_device(self, device) -> None:
        """Set ``self.device`` from the constructor argument."""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def _resolve_pad(self) -> int:
        """Derived from frames setting (single-frame models ⇒ pad=0)."""
        return (self.model_cfg.frames - 1) // 2

    # ------------------------------------------------------------------
    # Private helpers – model loading
    # ------------------------------------------------------------------

    def _setup_estimator_2d(self) -> None:
        """Load the 2D estimator's runtime resources (safe to call repeatedly)."""
        self._estimator_2d.setup_runtime()

    def _setup_model(self) -> torch.nn.Module:
        """Initialise the 3D lifting model on first use."""
        if self._model_3d is None:
            ModelClass = get_model(self.model_cfg.model_type)
            self._model_3d = ModelClass(self.model_cfg).to(self.device)
            self._model_3d.eval()
        return self._model_3d

    def _load_weights(self) -> None:
        """Load checkpoint weights into ``self._model_3d``.

        Mirrors the demo's loading strategy: iterate over the model's own
        state-dict keys and pull matching entries from the checkpoint so that
        extra keys in the checkpoint are silently ignored.
        """
        if self._model_3d is None:
            raise ValueError("Model not initialised. Call setup_runtime() first.")
        weights = self._resolve_model_weights_path()
        state_dict = torch.load(
            weights,
            weights_only=True,
            map_location=self.device,
        )
        self._model_3d.load_state_dict(state_dict)

    # ------------------------------------------------------------------
    # Private helpers – input resolution
    # ------------------------------------------------------------------

    def _resolve_model_weights_path(self) -> None:      
        if self.model_weights_path is None:
            self._download_model_weights()
        self.model_weights_path = Path(self.model_weights_path).resolve()
        if not self.model_weights_path.is_file():
            raise ValueError(
                f"Model weights file not found: {self.model_weights_path}. "
                "Please provide a valid path to a .pth checkpoint file in the "
                "FMPose3DInference constructor. Or leave it empty to download "
                "the weights from huggingface."
            )
        return self.model_weights_path

    def _download_model_weights(self) -> None:
        """Download model weights from HuggingFace Hub.

        The weight file is determined by the current ``model_cfg.model_type``
        (e.g. ``"fmpose3d_humans"`` -> ``fmpose3d_humans.pth``).  Files are
        cached locally by :func:`huggingface_hub.hf_hub_download` so
        subsequent calls are instant.

        Sets ``self.model_weights_path`` to the local cached file path.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required to download model weights. "
                "Install it with:  pip install huggingface_hub. Or download "
                "the weights manually and set model_weights_path to the weights file."
            ) from None

        filename = f"{self.model_cfg.model_type.value}.pth"
        self.model_weights_path = hf_hub_download(
            repo_id=_HF_REPO_ID,
            filename=filename,
        )

    def _ingest_input(self, source: Source) -> _IngestedInput:
        """Normalise *source* into a ``(N, H, W, C)`` frames array.

        Accepted *source* values:

        * **str / Path** – path to a single image or a directory of images.
        * **ndarray (H, W, C)** – a single BGR frame.
        * **ndarray (N, H, W, C)** – a batch of BGR frames.
        * **list of str/Path** – multiple image file paths.
        * **list of ndarray** – multiple ``(H, W, C)`` BGR frames.

        Video files are not yet supported and will raise
        :class:`NotImplementedError`.

        Parameters
        ----------
        source : Source
            The input to resolve.

        Returns
        -------
        _IngestedInput
            Contains ``frames`` as ``(N, H, W, C)`` and ``image_size``
            as ``(height, width)``.
        """
        import cv2

        # -- numpy array (single frame or batch) ----------------------------
        if isinstance(source, np.ndarray):
            if source.ndim == 3:
                frames = source[np.newaxis]  # (1, H, W, C)
            elif source.ndim == 4:
                frames = source
            else:
                raise ValueError(
                    f"Expected ndarray with 3 (H,W,C) or 4 (N,H,W,C) dims, "
                    f"got {source.ndim}"
                )
            h, w = frames.shape[1], frames.shape[2]
            return _IngestedInput(frames=frames, image_size=(h, w))

        # -- list / sequence ------------------------------------------------
        if isinstance(source, (list, tuple)):
            if len(source) == 0:
                raise ValueError("Empty source list.")

            first = source[0]

            # List of arrays
            if isinstance(first, np.ndarray):
                frames = np.stack(list(source), axis=0)
                h, w = frames.shape[1], frames.shape[2]
                return _IngestedInput(frames=frames, image_size=(h, w))

            # List of paths
            if isinstance(first, (str, Path)):
                loaded = []
                for p in source:
                    p = Path(p)
                    self._check_not_video(p)
                    img = cv2.imread(str(p))
                    if img is None:
                        raise FileNotFoundError(
                            f"Could not read image: {p}"
                        )
                    loaded.append(img)
                frames = np.stack(loaded, axis=0)
                h, w = frames.shape[1], frames.shape[2]
                return _IngestedInput(frames=frames, image_size=(h, w))

            raise TypeError(
                f"Unsupported element type in source list: {type(first)}"
            )

        # -- str / Path (file or directory) ---------------------------------
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Source path does not exist: {p}")

        self._check_not_video(p)

        if p.is_dir():
            images = sorted(
                f for f in p.iterdir()
                if f.suffix.lower() in self._IMAGE_EXTENSIONS
            )
            if not images:
                raise FileNotFoundError(
                    f"No image files found in directory: {p}"
                )
            loaded = []
            for img_path in images:
                img = cv2.imread(str(img_path))
                if img is None:
                    raise FileNotFoundError(
                        f"Could not read image: {img_path}"
                    )
                loaded.append(img)
            frames = np.stack(loaded, axis=0)
            h, w = frames.shape[1], frames.shape[2]
            return _IngestedInput(frames=frames, image_size=(h, w))

        # Single image file
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {p}")
        frames = img[np.newaxis]  # (1, H, W, C)
        h, w = frames.shape[1], frames.shape[2]
        return _IngestedInput(frames=frames, image_size=(h, w))

    def _check_not_video(self, p: Path) -> None:
        """Raise :class:`NotImplementedError` if *p* looks like a video."""
        _VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTS:
            raise NotImplementedError(
                f"Video input is not yet supported (got {p}). "
                "Please extract frames and pass them as image paths or arrays."
            )
