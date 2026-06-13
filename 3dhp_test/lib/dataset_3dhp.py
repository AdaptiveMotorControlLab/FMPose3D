"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

import numpy as np
from torch.utils.data import Dataset

from .camera import normalize_screen_coordinates, resolution_for_subject


TEST_NPZ_TO_FMPOSE_17 = np.array(
    [14, 8, 9, 10, 11, 12, 13, 15, 1, 16, 0, 5, 6, 7, 2, 3, 4],
    dtype=np.int64,
)


class ThreeDHPTestDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        subjects,
        test_augmentation=True,
        kps_left=None,
        kps_right=None,
        joints_left=None,
        joints_right=None,
    ):
        self.dataset_path = dataset_path
        self.subjects = subjects
        self.test_augmentation = bool(test_augmentation)
        self.kps_left = kps_left or [4, 5, 6, 11, 12, 13]
        self.kps_right = kps_right or [1, 2, 3, 14, 15, 16]
        self.joints_left = joints_left or [4, 5, 6, 11, 12, 13]
        self.joints_right = joints_right or [1, 2, 3, 14, 15, 16]
        self._data = self._load()
        self.pairs = self._build_pairs()

    def _load(self):
        raw = np.load(self.dataset_path, allow_pickle=True)["data"].item()
        out = {}

        for subject in self.subjects:
            if subject not in raw:
                raise KeyError(f"Subject {subject} is missing from {self.dataset_path}")

            anim = raw[subject]
            valid = anim["valid"].astype(bool)
            data_2d = anim["data_2d"][valid][:, TEST_NPZ_TO_FMPOSE_17, :]
            data_3d = anim["data_3d"][valid][:, TEST_NPZ_TO_FMPOSE_17, :] / 1000.0

            data_3d[:, 1:] -= data_3d[:, :1]

            width, height = resolution_for_subject(subject)
            data_2d = normalize_screen_coordinates(data_2d, w=width, h=height)

            out[subject] = {
                "positions_2d": data_2d,
                "positions_3d": data_3d,
                "valid_count": int(valid.sum()),
                "original_count": int(valid.shape[0]),
            }

        return out

    def _build_pairs(self):
        pairs = []
        for subject in self.subjects:
            n_frames = self._data[subject]["positions_2d"].shape[0]
            for frame_idx in range(n_frames):
                pairs.append((subject, frame_idx))
        return pairs

    def summary(self):
        return {
            subject: {
                "valid_count": self._data[subject]["valid_count"],
                "original_count": self._data[subject]["original_count"],
            }
            for subject in self.subjects
        }

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        subject, frame_idx = self.pairs[index]
        pose_2d = self._data[subject]["positions_2d"][frame_idx : frame_idx + 1].copy()
        pose_3d = self._data[subject]["positions_3d"][frame_idx : frame_idx + 1].copy()

        input_2d = pose_2d[None, ...]

        if self.test_augmentation:
            flip_2d = pose_2d.copy()
            flip_2d[:, :, 0] *= -1
            flip_2d[:, self.kps_left + self.kps_right] = flip_2d[:, self.kps_right + self.kps_left]
            input_2d = np.concatenate((input_2d, flip_2d[None, ...]), axis=0)

        return (
            np.zeros(9, dtype=np.float32),
            pose_3d,
            input_2d.astype(np.float32),
            "Seq1",
            subject,
            0,
        )
