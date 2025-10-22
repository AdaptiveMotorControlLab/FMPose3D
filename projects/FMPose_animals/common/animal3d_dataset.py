
import numpy as np
import torch
# from yacs.config import CfgNode

from torch.utils.data import ConcatDataset
from typing import List

import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
from .utils import get_example, expand_to_aspect_ratio


class TrainDataset(Dataset):
    def __init__(self, is_train: bool, json_file: str):
        super().__init__()
        self.focal_length = 1000

        json_file = json_file
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.is_train = is_train

    def __len__(self):
        return len(self.data['data'])

    def __getitem__(self, item):
        data = self.data['data'][item]
        keypoint_2d = np.array(data['keypoint_2d'], dtype=np.float32)
        if 'keypoint_3d' in data:
            keypoint_3d = np.concatenate(
                (data['keypoint_3d'], np.ones((len(data['keypoint_3d']), 1))), axis=-1).astype(np.float32)
        else:
            keypoint_3d = np.zeros((len(keypoint_2d), 4), dtype=np.float32)
        bbox = data['bbox']  # [x, y, w, h]
        ori_keypoint_2d = keypoint_2d.copy()

        item = {
                'keypoints_2d': keypoint_2d, #
                'keypoints_3d': keypoint_3d,
                }
        return item
    

class EvaluationDataset(Dataset):
    def __init__(self, json_file: str):
        super().__init__()
        self.focal_length = 1000

        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.is_train = False

    def __len__(self):
        return len(self.data['data'])

    def __getitem__(self, item):
        data = self.data['data'][item]
        keypoint_2d = np.array(data['keypoint_2d'], dtype=np.float32)
        keypoint_3d = np.concatenate(
            (data['keypoint_3d'], np.ones((len(data['keypoint_3d']), 1))), axis=-1).astype(np.float32)
        bbox = data['bbox']  # [x, y, w, h]

        ori_keypoint_2d = keypoint_2d.copy()
  
        item = {
                'keypoints_2d': keypoint_2d,
                'keypoints_3d': keypoint_3d,
              }
        return item


    