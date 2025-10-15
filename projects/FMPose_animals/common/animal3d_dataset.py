import copy
import os
import numpy as np
import torch
from yacs.config import CfgNode
import cv2
import pyrootutils
from torch.utils.data import ConcatDataset
from typing import List
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import json
import hydra
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple
from .utils import get_example, expand_to_aspect_ratio


class TrainDataset(Dataset):
    def __init__(self, cfg: CfgNode, is_train: bool, root_image: str, json_file: str):
        super().__init__()
        self.root_image = root_image
        self.focal_length = cfg.SMAL.get("FOCAL_LENGTH", 1000)

        json_file = json_file
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.is_train = is_train
        self.IMG_SIZE = cfg.MODEL.IMAGE_SIZE
        self.MEAN = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.STD = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.use_skimage_antialias = cfg.DATASETS.get('USE_SKIMAGE_ANTIALIAS', False)
        self.border_mode = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
        }[cfg.DATASETS.get('BORDER_MODE', 'constant')]

        self.augm_config = cfg.DATASETS.CONFIG

    def __len__(self):
        return len(self.data['data'])

    def __getitem__(self, item):
        data = self.data['data'][item]
        key = data['img_path']
        image = np.array(Image.open(os.path.join(self.root_image, key)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.root_image, data['mask_path'])).convert('L'))
        category_idx = data['supercategory']
        keypoint_2d = np.array(data['keypoint_2d'], dtype=np.float32)
        if 'keypoint_3d' in data:
            keypoint_3d = np.concatenate(
                (data['keypoint_3d'], np.ones((len(data['keypoint_3d']), 1))), axis=-1).astype(np.float32)
        else:
            keypoint_3d = np.zeros((len(keypoint_2d), 4), dtype=np.float32)
        bbox = data['bbox']  # [x, y, w, h]
        center = np.array([(bbox[0] * 2 + bbox[2]) // 2, (bbox[1] * 2 + bbox[3]) // 2])
        pose = np.array(data['pose'], dtype=np.float32) if 'pose' in data else np.zeros(105, dtype=np.float32)  # [105, ]
        betas = np.array(data['shape'] + data['shape_extra'], dtype=np.float32) if 'shape' in data else np.zeros(41, dtype=np.float32)  # [41, ]
        translation = np.array(data['trans'], dtype=np.float32) if 'trans' in data else np.zeros(3, dtype=np.float32)  # [3, ]
        has_pose = np.array(1., dtype=np.float32) if not (pose.all() == 0) else np.array(0., dtype=np.float32)
        has_betas = np.array(1., dtype=np.float32) if not (betas.all() == 0) else np.array(0., dtype=np.float32)
        has_translation = np.array(1., dtype=np.float32) if not (translation.all() == 0) else np.array(0., dtype=np.float32)
        ori_keypoint_2d = keypoint_2d.copy()
        center_x, center_y = center[0], center[1]
        bbox_size = max([bbox[2], bbox[3]])

        smal_params = {'global_orient': pose[:3],
                       'pose': pose[3:],
                       'betas': betas,
                       'transl': translation,
                       }
        has_smal_params = {'global_orient': has_pose,
                           'pose': has_pose,
                           'betas': has_betas,
                           'transl': has_translation,
                           }
        smal_params_is_axis_angle = {'global_orient': True,
                                     'pose': True,
                                     'betas': False,
                                     'transl': False,
                                     }

        augm_config = copy.deepcopy(self.augm_config)
        img_rgba = np.concatenate([image, mask[:, :, None]], axis=2)
        img_patch_rgba, keypoints_2d, keypoints_3d, smal_params, has_smal_params, img_size, trans, img_border_mask = get_example(
            img_rgba,
            center_x, center_y,
            bbox_size, bbox_size,
            keypoint_2d, keypoint_3d,
            smal_params, has_smal_params,
            self.IMG_SIZE, self.IMG_SIZE,
            self.MEAN, self.STD, self.is_train, augm_config,
            is_bgr=False, return_trans=True,
            use_skimage_antialias=self.use_skimage_antialias,
            border_mode=self.border_mode
        )
        img_patch = (img_patch_rgba[:3, :, :])
        mask_patch = (img_patch_rgba[3, :, :] / 255.0).clip(0, 1)
        if (mask_patch < 0.5).all():
            mask_patch = np.ones_like(mask_patch)

        item = {'img': img_patch,
                'mask': mask_patch,
                'keypoints_2d': keypoints_2d,
                'keypoints_3d': keypoints_3d,
                'orig_keypoints_2d': ori_keypoint_2d,
                'box_center': np.array(center.copy(), dtype=np.float32),
                'box_size': float(bbox_size),
                'img_size': np.array(1.0 * img_size[::-1].copy(), dtype=np.float32),
                'smal_params': smal_params,
                'has_smal_params': has_smal_params,
                'smal_params_is_axis_angle': smal_params_is_axis_angle,
                '_trans': trans,
                'focal_length': np.array([self.focal_length, self.focal_length], dtype=np.float32),
                'category': np.array(category_idx, dtype=np.int32),
                'supercategory': np.array(category_idx, dtype=np.int32),
                "img_border_mask": img_border_mask.astype(np.float32),
                "has_mask": np.array(1, dtype=np.float32)}
        return item
    

class EvaluationDataset(Dataset):
    def __init__(self, root_image: str,  json_file: str, augm_config,
                 focal_length: int=1000, image_size: int=256, 
                 mean: List[float]=[0.485, 0.456, 0.406], std: List[float]=[0.229, 0.224, 0.225]):
        super().__init__()
        self.root_image = root_image
        self.focal_length = focal_length

        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.is_train = False
        self.IMG_SIZE = image_size
        self.MEAN = 255. * np.array(mean)
        self.STD = 255. * np.array(std)
        self.use_skimage_antialias = False
        self.border_mode = cv2.BORDER_CONSTANT
        self.augm_config = augm_config

    def __len__(self):
        return len(self.data['data'])

    def __getitem__(self, item):
        data = self.data['data'][item]
        key = data['img_path']
        image = np.array(Image.open(os.path.join(self.root_image, key)).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.root_image, data['mask_path'])).convert('L'))
        category_idx = data['supercategory']
        keypoint_2d = np.array(data['keypoint_2d'], dtype=np.float32)
        keypoint_3d = np.concatenate(
            (data['keypoint_3d'], np.ones((len(data['keypoint_3d']), 1))), axis=-1).astype(np.float32)
        bbox = data['bbox']  # [x, y, w, h]
        center = np.array([(bbox[0] * 2 + bbox[2]) // 2, (bbox[1] * 2 + bbox[3]) // 2])
        pose = np.array(data['pose'], dtype=np.float32)  # [105, ]
        try:
            betas = np.array(data['shape'] + data['shape_extra'], dtype=np.float32)  # [41, ]
        except:
            betas = np.array(data['shape'], dtype=np.float32)
        translation = np.array(data['trans'], dtype=np.float32)  # [3, ]
        has_pose = np.array(1., dtype=np.float32) if pose is not None else np.array(0., dtype=np.float32)
        has_betas = np.array(1., dtype=np.float32) if betas is not None else np.array(0., dtype=np.float32)
        has_translation = np.array(1., dtype=np.float32) if translation is not None else np.array(0., dtype=np.float32)
        ori_keypoint_2d = keypoint_2d.copy()
        center_x, center_y = center[0], center[1]
        
        scale = np.array([bbox[2], bbox[3]], dtype=np.float32) / 200.
        bbox_size = expand_to_aspect_ratio(scale*200, None).max()
        bbox_expand_factor = bbox_size / ((scale*200).max())

        smal_params = {'global_orient': pose[:3],
                       'pose': pose[3:],
                       'betas': betas,
                       'transl': translation,
                       'bone': np.zeros(24, dtype=np.float32) if 'bone' not in data else np.array(data['bone'])
                       }
        has_smal_params = {'global_orient': has_pose,
                           'pose': has_pose,
                           'betas': has_betas,
                           'transl': has_translation,
                           'bone': np.array(1., dtype=np.float32) if 'bone' in data else np.array(0., dtype=np.float32),
                           }
        smal_params_is_axis_angle = {'global_orient': True,
                                     'pose': True,
                                     'betas': False,
                                     'transl': False,
                                     'bone': False
                                     }

        augm_config = copy.deepcopy(self.augm_config)
        img_rgba = np.concatenate([image, mask[:, :, None]], axis=2)
        img_patch_rgba, keypoints_2d, keypoints_3d, smal_params, has_smal_params, img_size, trans, img_border_mask = get_example(
            img_rgba,
            center_x, center_y,
            bbox_size, bbox_size,
            keypoint_2d, keypoint_3d,
            smal_params, has_smal_params,
            self.IMG_SIZE, self.IMG_SIZE,
            self.MEAN, self.STD, self.is_train, augm_config,
            is_bgr=False, return_trans=True,
            use_skimage_antialias=self.use_skimage_antialias,
            border_mode=self.border_mode
        )
        img_patch = (img_patch_rgba[:3, :, :])
        mask_patch = (img_patch_rgba[3, :, :] / 255.0).clip(0, 1)
        if (mask_patch < 0.5).all():
            mask_patch = np.ones_like(mask_patch)

        item = {'img': img_patch,
                'mask': mask_patch,
                'keypoints_2d': keypoints_2d,
                'keypoints_3d': keypoints_3d,
                'orig_keypoints_2d': ori_keypoint_2d,
                'box_center': np.array(center.copy(), dtype=np.float32),
                'box_size': float(bbox_size),
                'img_size': np.array(1.0 * img_size[::-1].copy(), dtype=np.float32),
                'smal_params': smal_params,
                'has_smal_params': has_smal_params,
                'smal_params_is_axis_angle': smal_params_is_axis_angle,
                '_trans': trans,
                'focal_length': np.array([self.focal_length, self.focal_length], dtype=np.float32),
                'category': np.array(category_idx, dtype=np.int32),
                'bbox_expand_factor': bbox_expand_factor,
                'supercategory': np.array(category_idx, dtype=np.int32),
                "img_border_mask": img_border_mask.astype(np.float32),
                'has_mask': np.array(1., dtype=np.float32)}
        return item


class OptionAnimalDataset(Dataset):
    def __init__(self, cfg: CfgNode):
        datasets = []
        weights = []

        dataset_configs = cfg.DATASETS
        for dataset_name in dataset_configs:
            if dataset_name != "CONFIG":
                datasets.append(TrainDataset(cfg, 
                                            is_train=True,
                                            root_image=dataset_configs[dataset_name].ROOT_IMAGE,
                                            json_file=dataset_configs[dataset_name].JSON_FILE.TRAIN))
                weights.extend([dataset_configs[dataset_name].WEIGHT] * len(datasets[-1]))

        # Concatenate all enabled datasets
        if datasets:
            self.dataset = ConcatDataset(datasets)
            self.weights = torch.tensor(weights, dtype=torch.float32)
        else:
            raise ValueError("No datasets enabled in the configuration.")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    