import numpy as np
import torch
# from yacs.config import CfgNode

from torch.utils.data import ConcatDataset
from typing import List

import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple

from common.camera import normalize_screen_coordinates


class TrainDataset(Dataset):
    def __init__(self, is_train: bool, json_file: str, root_joint: int = 12):
        super().__init__()
        self.focal_length = 1000
        self.root_joint = root_joint  # Root joint index for making coordinates relative

        json_file = json_file
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.is_train = is_train

    def __len__(self):
        return len(self.data['data'])

    def __getitem__(self, item):
        data = self.data['data'][item]
        # 安全地检查是否存在 reproj_kp_2d
        reproj = data.get('reproj_kp_2d', None)
        if reproj is not None:
            keypoint_2d = np.array(reproj, dtype=np.float32)
        else:
            keypoint_2d = np.array(data.get('keypoint_2d', []), dtype=np.float32)
         # normalize 2D keypoints
        hight = np.array(data['height'])
        width = np.array(data['width'])
        keypoint_2d = normalize_screen_coordinates(keypoint_2d[...,:2], width, hight)
         
         
        # 3D keypoints：先转 numpy，再与 1 拼接；若不存在则置零
        if 'keypoint_3d' in data and data['keypoint_3d'] is not None:
            kp3d = np.array(data['keypoint_3d'], dtype=np.float32)
            keypoint_3d = np.concatenate(
                (kp3d, np.ones((len(kp3d), 1), dtype=np.float32)), axis=-1)
        else:
            keypoint_3d = np.zeros((len(keypoint_2d), 4), dtype=np.float32)
        
        # Make 3D keypoints root-relative
        if keypoint_3d.shape[0] > self.root_joint:
            root_pos = keypoint_3d[self.root_joint:self.root_joint+1, :].copy()  # (1, 4)
            keypoint_3d = keypoint_3d - root_pos  # All joints relative to root
            # Now root joint should be exactly [0,0,0,0]
        
        bbox = data['bbox']  # [x, y, w, h]
        ori_keypoint_2d = keypoint_2d.copy()
 
        item = {
                 'keypoints_2d': keypoint_2d, #
                 'keypoints_3d': keypoint_3d,
                 }
        return item
    

class EvaluationDataset(Dataset):
    def __init__(self, json_file: str, root_joint: int = 12):
        super().__init__()
        self.focal_length = 1000
        self.root_joint = root_joint  # Root joint index for making coordinates relative

        with open(json_file, 'r') as f:
            self.data = json.load(f)


    def __len__(self):
        return len(self.data['data'])

    def __getitem__(self, item):
        data = self.data['data'][item]
        # 读取 2D keypoints（若有 reproj 可按需改为 reproj）
        keypoint_2d = np.array(data.get('keypoint_2d', []), dtype=np.float32)

        # 构造 3D keypoints
        if 'keypoint_3d' in data and data['keypoint_3d'] is not None:
            kp3d = np.array(data['keypoint_3d'], dtype=np.float32)
            keypoint_3d = np.concatenate(
                (kp3d, np.ones((len(kp3d), 1), dtype=np.float32)), axis=-1)
        else:
            keypoint_3d = np.zeros((len(keypoint_2d), 4), dtype=np.float32)
        
        # Make 3D keypoints root-relative
        if keypoint_3d.shape[0] > self.root_joint:
            root_pos = keypoint_3d[self.root_joint:self.root_joint+1, :].copy()  # (1, 4)
            keypoint_3d = keypoint_3d - root_pos  # All joints relative to root
            # Now root joint should be exactly [0,0,0,0]
        
        bbox = data['bbox']  # [x, y, w, h]
        # normalize 2D keypoints
        hight = np.array(data['height'], dtype=np.float32)
        width = np.array(data['width'], dtype=np.float32)
        keypoint_2d = normalize_screen_coordinates(keypoint_2d[...,:2], width, hight)
        
        

        ori_keypoint_2d = keypoint_2d.copy()
  
        item = {
                'keypoints_2d': keypoint_2d,
                'keypoints_3d': keypoint_3d,
              }
        return item


