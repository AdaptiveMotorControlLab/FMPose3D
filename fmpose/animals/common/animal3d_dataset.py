import json
from typing import List, Optional, Tuple

import numpy as np
import torch
from common.camera import normalize_screen_coordinates
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# from yacs.config import CfgNode





class TrainDataset(Dataset):
    def __init__(self, is_train: bool, json_file: str):
        super().__init__()
        self.focal_length = 1000

        json_file = json_file
        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.is_train = is_train

    def __len__(self):
        return len(self.data["data"])

    def __getitem__(self, item):
        data = self.data["data"][item]
        # safely check for reproj_kp_2d
        reproj = data.get("reproj_kp_2d", None)
        if reproj is not None:
            keypoint_2d = np.array(reproj, dtype=np.float32)
        else:
            keypoint_2d = np.array(data.get("keypoint_2d", []), dtype=np.float32)
        # normalize 2D keypoints
        hight = np.array(data["height"])
        width = np.array(data["width"])
        keypoint_2d = normalize_screen_coordinates(keypoint_2d[..., :2], width, hight)

        # build 3D keypoints; append ones; fallback to zeros if missing
        if "keypoint_3d" in data and data["keypoint_3d"] is not None:
            kp3d = np.array(data["keypoint_3d"], dtype=np.float32)
            keypoint_3d = np.concatenate((kp3d, np.ones((len(kp3d), 1), dtype=np.float32)), axis=-1)
        else:
            keypoint_3d = np.zeros((len(keypoint_2d), 4), dtype=np.float32)
        bbox = data["bbox"]  # [x, y, w, h]
        ori_keypoint_2d = keypoint_2d.copy()
        # print("keypoint_3d:",keypoint_3d.shape)
        # root-relative to joint 0
        if keypoint_3d.shape[0] > 1:
            keypoint_3d[1:, :] -= keypoint_3d[:1, :]

        item = {
            "keypoints_2d": keypoint_2d,  #
            "keypoints_3d": keypoint_3d,
        }
        return item


class EvaluationDataset(Dataset):
    def __init__(self, json_file: str):
        super().__init__()
        self.focal_length = 1000

        with open(json_file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data["data"])

    def __getitem__(self, item):
        data = self.data["data"][item]
        # read 2D keypoints (switch to reproj if desired)
        keypoint_2d = np.array(data.get("keypoint_2d", []), dtype=np.float32)

        # build 3D keypoints and root-relative to joint 0
        if "keypoint_3d" in data and data["keypoint_3d"] is not None:
            keypoint_3d = np.concatenate(
                (data["keypoint_3d"], np.ones((len(data["keypoint_3d"]), 1))), axis=-1
            ).astype(np.float32)
            keypoint_3d[:, 1:, :] -= keypoint_3d[:, :1, :]
        else:
            keypoint_3d = np.zeros((len(keypoint_2d), 4), dtype=np.float32)
        bbox = data["bbox"]  # [x, y, w, h]
        # normalize 2D keypoints
        hight = np.array(data["height"], dtype=np.float32)
        width = np.array(data["width"], dtype=np.float32)
        keypoint_2d = normalize_screen_coordinates(keypoint_2d[..., :2], width, hight)

        ori_keypoint_2d = keypoint_2d.copy()

        item = {
            "keypoints_2d": keypoint_2d,
            "keypoints_3d": keypoint_3d,
        }
        return item
