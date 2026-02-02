"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import os.path as osp
import argparse
import time
import numpy as np
from tqdm import tqdm
import json
import torch
import torch.backends.cudnn as cudnn
import cv2
import copy

from fmpose3d.lib.hrnet.lib.utils.utilitys import plot_keypoint, PreProcess, write, load_json
from fmpose3d.lib.hrnet.lib.config import cfg, update_config
from fmpose3d.lib.hrnet.lib.utils.transforms import *
from fmpose3d.lib.hrnet.lib.utils.inference import get_final_preds
from fmpose3d.lib.hrnet.lib.models import pose_hrnet

cfg_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'experiments') + '/'

# Auto-download checkpoints if missing and get checkpoint paths
from fmpose3d.lib.checkpoint.download_checkpoints import ensure_checkpoints, get_checkpoint_path
ensure_checkpoints()

# Loading human detector model
from fmpose3d.lib.yolov3.human_detector import load_model as yolo_model
from fmpose3d.lib.yolov3.human_detector import yolo_human_det as yolo_det
from fmpose3d.lib.sort.sort import Sort

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default=cfg_dir + 'w48_384x288_adam_lr1e-3.yaml',
                        help='experiment configure file name')
    parser.add_argument('opts', nargs=argparse.REMAINDER, default=None,
                        help="Modify config options using the command-line")
    parser.add_argument('--modelDir', type=str, default=get_checkpoint_path('pose_hrnet_w48_384x288.pth'),
                        help='The model directory')
    parser.add_argument('--det-dim', type=int, default=416,
                        help='The input dimension of the detected image')
    parser.add_argument('--thred-score', type=float, default=0.30,
                        help='The threshold of object Confidence')
    parser.add_argument('-a', '--animation', action='store_true',
                        help='output animation')
    parser.add_argument('-np', '--num-person', type=int, default=1,
                        help='The maximum number of estimated poses')
    parser.add_argument("-v", "--video", type=str, default='camera',
                        help="input video file name")
    
    parser.add_argument('--type', type=str, default='image', help='input type, onlys support image or video')
    parser.add_argument('--path', type=str, default='images/running.png', help='the path of your file')

    parser.add_argument('--gpu', type=str, default='0', help='input video')
    args, unknown = parser.parse_known_args()
    
    # Set opts to empty list to avoid passing unknown arguments to yacs config
    args.opts = []

    return args

def reset_config(args):
    update_config(cfg, args)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


# load model
def model_load(config):
    model = pose_hrnet.get_pose_net(config, is_train=False)
    if torch.cuda.is_available():
        model = model.cuda()

    state_dict = torch.load(config.OUTPUT_DIR, weights_only=True)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k  # remove module.
        #  print(name,'\t')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    # print('HRNet network successfully loaded')
    
    return model

def gen_from_image(args, frame, people_sort, human_model, pose_model, det_dim=416, num_peroson=1, gen_output=False):
    
    bboxs, scores = yolo_det(frame, human_model, reso=det_dim, confidence=args.thred_score)
    if bboxs is None or not bboxs.any():
        print('No person detected!')
        bboxs = bboxs_pre
        scores = scores_pre
    else:
        bboxs_pre = copy.deepcopy(bboxs) 
        scores_pre = copy.deepcopy(scores)

    # Using Sort to track people
    people_track = people_sort.update(bboxs)

    # Track the first two people in the video and remove the ID
    if people_track.shape[0] == 1:
        people_track_ = people_track[-1, :-1].reshape(1, 4)
    elif people_track.shape[0] >= 2:
        people_track_ = people_track[-num_peroson:, :-1].reshape(num_peroson, 4)
        people_track_ = people_track_[::-1]
    else:
        return [], []

    track_bboxs = []
    for bbox in people_track_:
        bbox = [round(i, 2) for i in list(bbox)]
        track_bboxs.append(bbox)

    with torch.no_grad():
        # bbox is coordinate location
        inputs, origin_img, center, scale = PreProcess(frame, track_bboxs, cfg, num_peroson)

        inputs = inputs[:, [2, 1, 0]]

        if torch.cuda.is_available():
            inputs = inputs.cuda()
        output = pose_model(inputs)

        # compute coordinate
        preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), np.asarray(center), np.asarray(scale))

    kpts = np.zeros((num_peroson, 17, 2), dtype=np.float32)
    scores = np.zeros((num_peroson, 17), dtype=np.float32)
    for i, kpt in enumerate(preds):
        kpts[i] = kpt

    for i, score in enumerate(maxvals):
        scores[i] = score.squeeze()
    
    return kpts, scores


def gen_video_kpts(path, det_dim=416, num_peroson=1, gen_output=False, type='image'):
    # Updating configuration
    args1 = parse_args()
    reset_config(args1)

    # Loading detector and pose model, initialize sort for track
    human_model = yolo_model(inp_dim=det_dim)
    pose_model = model_load(cfg)
    people_sort = Sort(min_hits=0)

    kpts_result = []
    scores_result = []
    if type == "image":
        frame = cv2.imread(path)
        kpts, scores = gen_from_image(args1, frame, people_sort, human_model, pose_model, det_dim=det_dim, num_peroson=num_peroson, gen_output=gen_output)
        kpts_result.append(kpts)
        scores_result.append(scores)

    if type == 'video':
        cap = cv2.VideoCapture(path)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for ii in tqdm(range(video_length)):
            ret, frame = cap.read()
            if not ret:
                continue

            kpts, scores = gen_from_image(args1, frame, people_sort, human_model, pose_model, det_dim=det_dim, num_peroson=num_peroson, gen_output=gen_output)
            kpts_result.append(kpts)
            scores_result.append(scores)

    keypoints = np.array(kpts_result)
    scores = np.array(scores_result)

    keypoints = keypoints.transpose(1, 0, 2, 3)  # (T, M, N, 2) --> (M, T, N, 2)
    scores = scores.transpose(1, 0, 2)  # (T, M, N) --> (M, T, N)

    return keypoints, scores

if __name__=="__main__":
    
    image_path = "./demo/images/xinxiaomeng_pose.png"
