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
    ModelConfig,
)
from fmpose3d.models import get_model

#: Progress callback signature: ``(current_step, total_steps) -> None``.
ProgressCallback = Callable[[int, int], None]


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
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        """
        from fmpose3d.lib.preprocess import h36m_coco_format, revise_kpts

        self.setup_runtime()

        keypoints, scores = self._model.predict(frames)

        keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
        # NOTE: revise_kpts is computed for consistency but is NOT applied
        # to the returned keypoints, matching the demo script behaviour.
        _revised = revise_kpts(keypoints, scores, valid_frames)  # noqa: F841

        return keypoints, scores


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class Pose2DResult:
    """Container returned by :meth:`FMPose3DInference.prepare_2d`."""

    keypoints: np.ndarray
    """H36M-format 2D keypoints, shape ``(num_persons, num_frames, 17, 2)``."""
    scores: np.ndarray
    """Per-joint confidence scores, shape ``(num_persons, num_frames, 17)``."""
    image_size: tuple[int, int] = (0, 0)
    """``(height, width)`` of the source frames."""


@dataclass
class Pose3DResult:
    """Container returned by :meth:`FMPose3DInference.pose_3d`."""

    poses_3d: np.ndarray
    """Root-relative 3D poses, shape ``(num_frames, 17, 3)``."""
    poses_3d_world: np.ndarray
    """World-coordinate 3D poses, shape ``(num_frames, 17, 3)``."""


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


class FMPose3DInference:
    """High-level, two-step inference API for FMPose3D.

    Typical workflow::

        api = FMPose3DInference(model_weights_path="weights.pth")
        result_2d = api.prepare_2d("photo.jpg")
        result_3d = api.pose_3d(result_2d.keypoints, image_size=(H, W))

    Parameters
    ----------
    model_cfg : ModelConfig, optional
        Model architecture settings (layers, channels, …).
        Defaults to :class:`~fmpose3d.common.config.FMPose3DConfig` defaults.
    inference_cfg : InferenceConfig, optional
        Inference settings (sample_steps, test_augmentation, …).
        Defaults to :class:`~fmpose3d.common.config.InferenceConfig` defaults.
    model_weights_path : str
        Path to a ``.pth`` checkpoint for the 3D lifting model.
        If empty the model is created but **not** loaded with weights.
    device : str or torch.device, optional
        Compute device.  ``None`` (default) picks CUDA when available.
    """

    # H36M joint indices for left / right flip augmentation
    _JOINTS_LEFT: list[int] = [4, 5, 6, 11, 12, 13]
    _JOINTS_RIGHT: list[int] = [1, 2, 3, 14, 15, 16]

    _IMAGE_EXTENSIONS: set[str] = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    }

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_cfg: ModelConfig | None = None,
        inference_cfg: InferenceConfig | None = None,
        model_weights_path: str = "",
        device: str | torch.device | None = None,
    ) -> None:
        self.model_cfg = model_cfg or FMPose3DConfig()
        self.inference_cfg = inference_cfg or InferenceConfig()
        self.model_weights_path = model_weights_path

        # Resolve device and padding configuration
        self._device: torch.device | None = self._resolve_device(device)
        self._pad: int = self._resolve_pad()

        # Lazy-loaded models  (populated by setup_runtime)
        self._model_3d: torch.nn.Module | None = None
        self._estimator_2d: HRNetEstimator | None = None

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
        """Estimate 2D poses using HRNet + YOLO.

        The estimator is set up lazily by :meth:`setup_runtime` on first
        call.

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
            H36M-format 2D keypoints and per-joint scores.  The result
            also carries ``image_size`` so it can be forwarded directly
            to :meth:`pose_3d`.
        """
        ingested = self._ingest_input(source)
        self.setup_runtime()
        if progress:
            progress(0, 1)
        keypoints, scores = self._estimator_2d.predict(ingested.frames)
        if progress:
            progress(1, 1)
        return Pose2DResult(
            keypoints=keypoints,
            scores=scores,
            image_size=ingested.image_size,
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

        The pipeline exactly mirrors ``demo/vis_in_the_wild.py``'s
        ``get_3D_pose_from_image``: normalise screen coordinates, build a
        flip-augmented conditioning pair, run two independent Euler ODE
        integrations (each with its own noise sample), un-flip and average,
        zero the root joint, then convert to world coordinates.

        Parameters
        ----------
        keypoints_2d : ndarray
            2D keypoints returned by :meth:`prepare_2d`.  Accepted shapes:

            * ``(num_persons, num_frames, 17, 2)`` – first person is used.
            * ``(num_frames, 17, 2)`` – treated as a single person.
        image_size : tuple of (int, int)
            ``(height, width)`` of the source image / video frames.
        camera_rotation : ndarray or None
            Length-4 quaternion for the camera-to-world rotation applied
            to produce ``poses_3d_world``.  Defaults to the rotation used
            in the official demo.  Pass ``None`` to skip the transform
            (``poses_3d_world`` will equal ``poses_3d``).
        seed : int or None
            If given, ``torch.manual_seed(seed)`` is called before
            sampling so that results are fully reproducible.  Use the
            same seed in the demo script (by inserting
            ``torch.manual_seed(seed)`` before the ``torch.randn`` calls)
            to obtain bit-identical results.
        progress : ProgressCallback or None
            Optional ``(current_step, total_steps)`` callback invoked
            after each frame is lifted to 3D.

        Returns
        -------
        Pose3DResult
            Root-relative and world-coordinate 3D poses.
        """
        self.setup_runtime()
        model = self._model_3d
        h, w = image_size
        steps = self.inference_cfg.sample_steps
        use_flip = self.inference_cfg.test_augmentation
        jl = self._JOINTS_LEFT
        jr = self._JOINTS_RIGHT

        # Optional deterministic seeding
        if seed is not None:
            torch.manual_seed(seed)

        # Normalise input shape to (num_frames, 17, 2)
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
            frame_kpts = kpts[i : i + 1]  # (1, 17, 2)

            # Normalise to [-1, 1] range  (same as demo)
            normed = normalize_screen_coordinates(frame_kpts, w=w, h=h)

            if use_flip:
                # -- build flip-augmented conditioning (matches demo exactly) --
                normed_flip = copy.deepcopy(normed)
                normed_flip[:, :, 0] *= -1
                normed_flip[:, jl + jr] = normed_flip[:, jr + jl]
                input_2d = np.concatenate(
                    (np.expand_dims(normed, axis=0), np.expand_dims(normed_flip, axis=0)),
                    0,
                )  # (2, F, J, 2)
                input_2d = input_2d[np.newaxis, :, :, :, :]  # (1, 2, F, J, 2)
                input_t = torch.from_numpy(input_2d.astype("float32")).to(self.device)

                # -- two independent Euler ODE runs (matches demo exactly) --
                y = torch.randn(
                    input_t.size(0), input_t.size(2), input_t.size(3), 3,
                    device=self.device,
                )
                output_3d_non_flip = euler_sample(
                    input_t[:, 0], y, steps, model,
                )

                y_flip = torch.randn(
                    input_t.size(0), input_t.size(2), input_t.size(3), 3,
                    device=self.device,
                )
                output_3d_flip = euler_sample(
                    input_t[:, 1], y_flip, steps, model,
                )

                # -- un-flip & average (matches demo exactly) --
                output_3d_flip[:, :, :, 0] *= -1
                output_3d_flip[:, :, jl + jr, :] = output_3d_flip[
                    :, :, jr + jl, :
                ]

                output = (output_3d_non_flip + output_3d_flip) / 2
            else:
                input_2d = normed[np.newaxis]  # (1, F, J, 2)
                input_t = torch.from_numpy(input_2d.astype("float32")).to(self.device)
                y = torch.randn(
                    input_t.size(0), input_t.size(1), input_t.size(2), 3,
                    device=self.device,
                )
                output = euler_sample(input_t, y, steps, model)

            # Extract single-frame result → (17, 3)  (matches demo exactly)
            output = output[0:, self._pad].unsqueeze(1)
            output[:, :, 0, :] = 0  # root-relative
            pose_np = output[0, 0].cpu().detach().numpy()
            all_poses_3d.append(pose_np)

            # Camera-to-world transform  (matches demo exactly)
            if camera_rotation is not None:
                pose_world = camera_to_world(pose_np, R=camera_rotation, t=0)
                pose_world[:, 2] -= np.min(pose_world[:, 2])
            else:
                pose_world = pose_np.copy()
            all_poses_world.append(pose_world)

            if progress:
                progress(i + 1, num_frames)

        poses_3d = np.stack(all_poses_3d, axis=0)  # (num_frames, 17, 3)
        poses_world = np.stack(all_poses_world, axis=0)  # (num_frames, 17, 3)

        return Pose3DResult(poses_3d=poses_3d, poses_3d_world=poses_world)

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

    def _setup_estimator_2d(self) -> HRNetEstimator:
        """Initialise the HRNet 2D pose estimator on first use."""
        if self._estimator_2d is None:
            self._estimator_2d = HRNetEstimator()
        return self._estimator_2d

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
        if not self.model_weights_path:
            raise ValueError(
                "No model weights path provided. Pass 'model_weights_path' "
                "to the FMPose3DInference constructor."
            )
        weights = Path(self.model_weights_path)
        if not weights.exists():
            raise ValueError(
                f"Model weights file not found: {weights}. "
                "Please provide a valid path to a .pth checkpoint file in the "
                "FMPose3DInference constructor."
            )
        if self._model_3d is None:
            raise ValueError("Model not initialised. Call setup_runtime() first.")
        pre_dict = torch.load(
            self.model_weights_path,
            weights_only=True,
            map_location=self.device,
        )
        model_dict = self._model_3d.state_dict()
        for name in model_dict:
            if name in pre_dict:
                model_dict[name] = pre_dict[name]
        self._model_3d.load_state_dict(model_dict)

    # ------------------------------------------------------------------
    # Private helpers – input resolution
    # ------------------------------------------------------------------

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
