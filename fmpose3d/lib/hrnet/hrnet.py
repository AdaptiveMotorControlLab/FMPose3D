"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0

FMPose3D – clean HRNet 2D pose estimation API.

Provides :class:`HRNetPose2d`, a self-contained wrapper around the
HRNet + YOLO detection pipeline that accepts numpy arrays directly
(no file I/O, no argparse, no global yacs config leaking out).

Usage::

    api = HRNetPose2d(det_dim=416, num_persons=1)
    api.setup()                       # loads YOLO + HRNet weights
    keypoints, scores = api.predict(frames)   # (M, N, 17, 2), (M, N, 17)
"""

from __future__ import annotations

import copy
import os.path as osp
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from fmpose3d.lib.checkpoint.download_checkpoints import (
    ensure_checkpoints,
    get_checkpoint_path,
)


class HRNetPose2d:
    """Self-contained 2D pose estimator (YOLO detector + HRNet).

    A self-contained HRNet 2D pose estimator that accepts numpy arrays directly.
    It serves as alternative to the gen_video_kpts function in fmpose3d/lib/hrnet/gen_kpts.py, 
    which generates 2D keypoints from a video file.

    Parameters
    ----------
    det_dim : int
        YOLO input resolution (default 416).
    num_persons : int
        Maximum number of persons to track per frame (default 1).
    thred_score : float
        YOLO object-confidence threshold (default 0.30).
    hrnet_cfg_file : str
        Path to the HRNet YAML experiment config.  Empty string (default)
        uses the bundled ``w48_384x288_adam_lr1e-3.yaml``.
    hrnet_weights_path : str
        Path to the HRNet ``.pth`` checkpoint.  Empty string (default)
        uses the auto-downloaded ``pose_hrnet_w48_384x288.pth``.
    """

    def __init__(
        self,
        det_dim: int = 416,
        num_persons: int = 1,
        thred_score: float = 0.30,
        hrnet_cfg_file: str = "",
        hrnet_weights_path: str = "",
    ) -> None:
        self.det_dim = det_dim
        self.num_persons = num_persons
        self.thred_score = thred_score
        self.hrnet_cfg_file = hrnet_cfg_file
        self.hrnet_weights_path = hrnet_weights_path

        # Populated by setup()
        self._human_model = None
        self._pose_model = None
        self._people_sort = None
        self._hrnet_cfg = None  # frozen yacs CfgNode used by PreProcess / get_final_preds

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """``True`` once :meth:`setup` has been called."""
        return self._human_model is not None

    def setup(self) -> "HRNetPose2d":
        """Load YOLO detector and HRNet pose model.

        Can safely be called more than once (subsequent calls are no-ops).

        Returns ``self`` so you can write ``api = HRNetPose2d().setup()``.
        """
        if self.is_ready:
            return self

        ensure_checkpoints()

        # --- resolve paths ---------------------------------------------------
        hrnet_cfg_file = self.hrnet_cfg_file
        if not hrnet_cfg_file:
            hrnet_cfg_file = osp.join(
                osp.dirname(osp.abspath(__file__)),
                "experiments",
                "w48_384x288_adam_lr1e-3.yaml",
            )

        hrnet_weights = self.hrnet_weights_path
        if not hrnet_weights:
            hrnet_weights = get_checkpoint_path("pose_hrnet_w48_384x288.pth")

        # --- build internal yacs config (kept private) -----------------------
        from fmpose3d.lib.hrnet.lib.config import cfg as _global_cfg
        from fmpose3d.lib.hrnet.lib.config import update_config as _update_cfg
        from types import SimpleNamespace

        _global_cfg.defrost()
        _update_cfg(
            _global_cfg,
            SimpleNamespace(cfg=hrnet_cfg_file, opts=[], modelDir=hrnet_weights),
        )
        # Snapshot the frozen cfg so we can pass it to PreProcess / get_final_preds.
        self._hrnet_cfg = _global_cfg

        # cudnn tuning
        cudnn.benchmark = self._hrnet_cfg.CUDNN.BENCHMARK
        cudnn.deterministic = self._hrnet_cfg.CUDNN.DETERMINISTIC
        cudnn.enabled = self._hrnet_cfg.CUDNN.ENABLED

        # --- load models -----------------------------------------------------
        from fmpose3d.lib.yolov3.human_detector import load_model as _yolo_load
        from fmpose3d.lib.sort.sort import Sort

        self._human_model = _yolo_load(inp_dim=self.det_dim)
        self._pose_model = self._load_hrnet(self._hrnet_cfg)
        self._people_sort = Sort(min_hits=0)

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self, frames: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate 2D keypoints for a batch of BGR frames.

        Parameters
        ----------
        frames : ndarray, shape ``(N, H, W, C)``
            BGR images.  A single frame ``(H, W, C)`` is also accepted
            and will be treated as a batch of one.

        Returns
        -------
        keypoints : ndarray, shape ``(num_persons, N, 17, 2)``
            COCO-format 2D keypoints in pixel coordinates.
        scores : ndarray, shape ``(num_persons, N, 17)``
            Per-joint confidence scores.
        """
        if not self.is_ready:
            self.setup()

        if frames.ndim == 3:
            frames = frames[np.newaxis]

        kpts_result = []
        scores_result = []

        for i in range(frames.shape[0]):
            kpts, sc = self._estimate_frame(frames[i])
            kpts_result.append(kpts)
            scores_result.append(sc)

        keypoints = np.array(kpts_result)   # (N, M, 17, 2)
        scores = np.array(scores_result)    # (N, M, 17)

        # (N, M, 17, 2) → (M, N, 17, 2)
        keypoints = keypoints.transpose(1, 0, 2, 3)
        # (N, M, 17) → (M, N, 17)
        scores = scores.transpose(1, 0, 2)

        return keypoints, scores

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_hrnet(config):
        """Instantiate HRNet and load checkpoint weights."""
        from fmpose3d.lib.hrnet.lib.models import pose_hrnet

        model = pose_hrnet.get_pose_net(config, is_train=False)
        if torch.cuda.is_available():
            model = model.cuda()

        state_dict = torch.load(config.OUTPUT_DIR, weights_only=True)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        model.eval()
        return model

    def _estimate_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run detection + pose estimation on a single BGR frame.

        Returns
        -------
        kpts : ndarray, shape ``(num_persons, 17, 2)``
        scores : ndarray, shape ``(num_persons, 17)``
        """
        from fmpose3d.lib.yolov3.human_detector import yolo_human_det
        from fmpose3d.lib.hrnet.lib.utils.utilitys import PreProcess
        from fmpose3d.lib.hrnet.lib.utils.inference import get_final_preds

        num_persons = self.num_persons

        bboxs, det_scores = yolo_human_det(
            frame, self._human_model, reso=self.det_dim, confidence=self.thred_score,
        )

        if bboxs is None or not bboxs.any():
            # No detection – return zeros
            kpts = np.zeros((num_persons, 17, 2), dtype=np.float32)
            scores = np.zeros((num_persons, 17), dtype=np.float32)
            return kpts, scores

        # Track
        people_track = self._people_sort.update(bboxs)

        if people_track.shape[0] == 1:
            people_track_ = people_track[-1, :-1].reshape(1, 4)
        elif people_track.shape[0] >= 2:
            people_track_ = people_track[-num_persons:, :-1].reshape(num_persons, 4)
            people_track_ = people_track_[::-1]
        else:
            kpts = np.zeros((num_persons, 17, 2), dtype=np.float32)
            scores = np.zeros((num_persons, 17), dtype=np.float32)
            return kpts, scores

        track_bboxs = []
        for bbox in people_track_:
            bbox = [round(i, 2) for i in list(bbox)]
            track_bboxs.append(bbox)

        with torch.no_grad():
            inputs, origin_img, center, scale = PreProcess(
                frame, track_bboxs, self._hrnet_cfg, num_persons,
            )
            inputs = inputs[:, [2, 1, 0]]

            if torch.cuda.is_available():
                inputs = inputs.cuda()
            output = self._pose_model(inputs)

            preds, maxvals = get_final_preds(
                self._hrnet_cfg,
                output.clone().cpu().numpy(),
                np.asarray(center),
                np.asarray(scale),
            )

        kpts = np.zeros((num_persons, 17, 2), dtype=np.float32)
        scores = np.zeros((num_persons, 17), dtype=np.float32)
        for i, kpt in enumerate(preds):
            kpts[i] = kpt
        for i, score in enumerate(maxvals):
            scores[i] = score.squeeze()

        return kpts, scores

