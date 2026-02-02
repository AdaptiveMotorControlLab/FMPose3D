"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

import copy
import gc
import glob
import os
import random
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import from_numpy as FN
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.append(os.path.dirname(sys.path[0]))

from common.camera import normalize_screen_coordinates
from common.lifter3d import load_camera_params, load_h5_keypoints


class ArberDataset(Dataset):
    def __init__(
        self,
        cfg,
        path,
        split,
        cam_names,
        root_index=12,
        joint_num=23,
        sampling_gap=100,
        frame_per_video=9000,
        norm_rate=50.0,
        img_W=2048,
        img_H=1536,
        arg_views=1,
        resize_2D_scale=0.5,
        visualize=False,
    ):

        self.cfg = cfg
        self.cam_names = cam_names
        self.joint_num = joint_num
        self.root_index = root_index
        self.img_W = img_W * resize_2D_scale
        self.img_H = img_H * resize_2D_scale
        self.arg_views = arg_views
        self.split = split
        self.visualize = visualize

        # subject_index: category names
        subject_index = os.listdir(path)
        subject_index.sort()

        # use split to define start and end frame
        if split == "Train":
            self.subject_list = subject_index
            self.start_frame = 0
            self.end_frame = 10000
        elif split == "Valid":
            self.subject_list = subject_index
            self.start_frame = 3
            self.end_frame = 8000
        elif split == "Test":
            self.subject_list = subject_index
            self.start_frame = 6
            self.end_frame = 10000
        elif split == "Infer":
            self.subject_list = subject_index[:1]
            self.start_frame = 0
            self.end_frame = 2000000

        # prepare pose data
        print("prepare the pose data...")
        self.pose_3D_list = []
        self.pose_2D_list = []
        self.sample_info_list = []
        self.cam_para_list = []

        for sub_idx, subject_name in enumerate(self.subject_list):  # iterate on subject
            print(subject_name)
            subject_folder = os.path.join(path, subject_name)

            # load asked cameras
            yaml_files = []
            for cam in cam_names:
                yaml_files.extend(
                    sorted(glob.glob(os.path.join(subject_folder, f"calibration/*{cam}*.yaml")))
                )

            # yaml_files = sorted(glob.glob(os.path.join(subject_folder,'calibration/*.yaml')))
            cameras = [load_camera_params(yaml) for yaml in yaml_files]
            self.cam_para_list = cameras

            # load triangulated 3d points
            # points_3d_np = np.load(os.path.join(subject_folder,'triangulated_3d.npy'))   # shape (num_frames, 23, 3)

            # apply norm_rate on translation vector
            for i in range(len(cam_names)):
                cameras[i]["T"] = cameras[i]["T"] / norm_rate

            # load all 2D keypoints from asked cameras
            h5_files = []
            for cam in cam_names:
                # print("cam:", cam)
                # h5_files.extend(sorted(glob.glob(os.path.join(subject_folder,f'pose2d_dlc/Camera_{cam}*.h5')))) # for cspnext model
                h5_files.extend(
                    sorted(glob.glob(os.path.join(subject_folder, f"pose2d_dlc/*{cam}*.h5")))
                )  # for rtmpose model

            keypoints_2d = [
                load_h5_keypoints(h5) for h5 in h5_files
            ]  # （num_cameras,num_frames,23,3) # for rtmpose model
            # keypoints_2d = [load_h5_keypoints_cspnext(h5) for h5 in h5_files]   # （num_cameras,num_frames,23,3) # for cspnext model

            # get total frame - > real end frame
            total_frame_num = keypoints_2d[0].shape[0]
            real_end_frame = min(self.end_frame, total_frame_num)

            for idx in tqdm(
                range(self.start_frame, real_end_frame, sampling_gap)
            ):  # get temporal video fragment of 2D and 3D keypoints
                idx = max(idx, self.t_pad)
                idx = min(idx, real_end_frame - self.t_pad - 1)
                left_frame_id = idx - self.t_pad
                right_frame_id = idx + self.t_pad + 1

                # record sample info
                tmp_info = np.zeros(2)
                tmp_info[0] = sub_idx
                tmp_info[1] = idx

                # extract 3d fragment, get 3D points from npy file

                points_3d_fragment = (
                    np.load(os.path.join(subject_folder, "triangulated_3d.npy"))[
                        left_frame_id:right_frame_id, :, :3
                    ]
                    / norm_rate
                )  # load from prepared .npy and apply norm_rate

                # print("points_3d_fragment shape:", points_3d_fragment.shape) # (num_frames, 23, 3)
                keypoints_2d = np.array(keypoints_2d)  # Ensure it's a NumPy array
                # get 2D keypoint vis
                points_2d_vis_np = keypoints_2d[:, left_frame_id:right_frame_id, :, 2:]  # N,T,K,1

                # clip vis
                points_2d_vis_np = np.clip(points_2d_vis_np, 0, 1)

                # get 2D keypoint
                points_2d_np = keypoints_2d[:, left_frame_id:right_frame_id, :, :2]  # N,T,K,2
                # # get 3D keypoint from 3D lifting
                # points_2d_fragment_np = np.array(points_2d_fragment) # (num_cams, num_frames, num_joints, 2) N,T,K,2
                # points_3d_fragment = triangulate_3d_batch(points_2d_fragment_np,cameras)  #(num_frames, 23, 3) T,K,3

                # get 3D pose from world to camera, with respect to different camera
                points_3d = np.zeros(
                    (self.t_length, self.joint_num, 3, len(self.cam_names))
                )  # initialize 3D keypints, from T,K,3 to T,K,3,N
                points_3d_world = np.reshape(points_3d_fragment, (-1, 3))  # T,K,3
                # print("before and after reshape",points_3d_fragment.shape,points_3d_world.shape)
                for cam_idx, cam in enumerate(cam_names):
                    # todo: check transformation
                    points_3d_cam = (
                        np.dot(points_3d_world, cameras[cam_idx]["R"].T) + cameras[cam_idx]["T"].T
                    )

                    points_3d[:, :, :, cam_idx] = np.reshape(
                        points_3d_cam, (self.t_length, self.joint_num, 3)
                    )  # T,K,3,N

                # get relative 3D pose
                points_3d_root = copy.deepcopy(
                    points_3d[:, self.root_index : self.root_index + 1, :, :]
                )
                rela_points_3d = points_3d - points_3d_root

                del points_3d, points_3d_root
                gc.collect()

                # normalize 2D pose
                points_2d_np = normalize_screen_coordinates(
                    copy.deepcopy(points_2d_np), self.img_W, self.img_H
                )  # N,T,K,2

                # get fake vis3d
                points_vis3D = np.ones((self.t_length, self.joint_num, 1))

                self.pose_3D_list.append(rela_points_3d)
                self.pose_2D_list.append(
                    np.nan_to_num(points_2d_np.transpose(1, 2, 3, 0))
                )  # transpose to T,K,2,N
                self.vid2D_list.append(
                    points_2d_vis_np.transpose(1, 2, 3, 0)
                )  # Transpose to T,K,1,N, move N to the end
                self.vid3D_list.append(points_vis3D)
                self.sample_info_list.append(tmp_info)

                del points_2d_np, points_vis3D, points_2d_vis_np
                gc.collect()
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.pose_3D_list)

    def __getitem__(self, index):
        return self.getitem(index)

    def getitem(self, index):

        pose_3D = self.pose_3D_list[index].copy()
        pose_2D = self.pose_2D_list[index].copy()
        vid_3D = self.vid3D_list[index].copy()
        vid_2D = self.vid2D_list[index].copy()
        sample_info = self.sample_info_list[index]

        if "TRAIN" in self.split.upper() and self.arg_views > 0:
            pose_3D, pose_2D = self.view_aug(pose_3D, pose_2D)
            tmp_vid = np.repeat(
                np.expand_dims(copy.deepcopy(vid_3D), axis=-1), self.arg_views, axis=-1
            )
            vid_2D = np.concatenate((vid_2D, tmp_vid), axis=-1)
            # clip vid into 0,1
            # vid_2D = np.clip(vid_2D,0,1)

        pose_root = copy.deepcopy(pose_3D[:, self.root_index : self.root_index + 1, :, :])
        pose_3D[:, self.root_index : self.root_index + 1, :, :] = 0.0
        pose_3D = np.nan_to_num(pose_3D, nan=0)
        pose_2D = np.concatenate((pose_2D, vid_2D), axis=2)

        return (
            FN(pose_3D).float(),
            FN(pose_root).float(),
            FN(pose_2D).float(),
            FN(vid_3D).float(),
            FN(sample_info).float(),
        )


if __name__ == "__main__":
    from common.arguments import parse_args
    from common.config import config as cfg
    from common.config import reset_config, update_config

    from scripts.reset_config_arber import reset_config_arber
    from scripts.reset_config_rat7m import reset_config_rat7m

    cam_names = ["Camera0", "Camera1", "Camera2", "Camera3", "Camera4", "Camera5"]
    data_dir = "/workspace/MTFpose/data/Arber_tiny"
    args = parse_args()
    update_config(args.cfg)
    reset_config(cfg, args)
    reset_config_arber(cfg)

    args = parse_args()
    update_config(args.cfg)  ###config file->cfg
    reset_config(cfg, args)  ###arg -> cfg
    reset_config_rat7m(cfg)

    print(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cfg.GPU)

    root_index = cfg.TINY_DATA.ROOT_INDEX
    sampling_gap = cfg.TINY_DATA.SAMPLING_GAP
    joint_num = cfg.TINY_DATA.NUM_JOINTS
    img_W, img_H = cfg.TINY_DATA.IMG_SIZE
    use_2d_gt = cfg.DATA.USE_GT_2D
    receptive_field = cfg.NETWORK.TEMPORAL_LENGTH
    pad = receptive_field // 2
    causal_shift = 0
    train_dataset = ArberDataset(
        cfg,
        cfg.ARBER_DATA.ROOT_DIR,
        "Train",
        cam_names,
        pad,
        root_index=root_index,
        use_2D_gt=use_2d_gt,
        joint_num=23,
        sampling_gap=60,
        img_W=img_W,
        img_H=img_H,
        arg_views=0,
        resize_2D_scale=cfg.ARBER_DATA.RESIZE_SCALE,
    )

    pose_3D, pose_root, pose_2D, vid_3D, rotation, sample_info = train_dataset.getitem(2)
    print(
        "output at item 250, pose_3D",
        pose_3D.shape,
        "pose_root",
        pose_root.shape,
        "pose_2D",
        pose_2D.shape,
        "vid_3D",
        vid_3D.shape,
        "rotation",
        rotation.shape,
        "sample_info",
        sample_info,
    )
    # output at item 250, pose_3D torch.Size([7, 23, 3, 6]) pose_root torch.Size([7, 1, 3, 6]) pose_2D torch.Size([7, 23, 3, 6]) vid_3D torch.Size([7, 23, 1]) rotation torch.Size([3, 3, 1, 6, 6]) sample_info tensor([  0., 120.])
    print("in camera 0", pose_2D[0, 0, :, 0], "in camera 1", pose_2D[0, 0, :, 1])
    print(f"pose_2D maxime: {pose_2D.max().item():.4f}")
    print(f"pose_2D mini: {pose_2D.min().item():.4f}")
