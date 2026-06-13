"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

import hashlib

import torch
from torch.autograd import Variable


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def get_variable(split, target):
    out = []
    for item in target:
        if split == "train":
            out.append(Variable(item, requires_grad=False).contiguous().type(torch.cuda.FloatTensor))
        else:
            out.append(Variable(item).contiguous().cuda().type(torch.cuda.FloatTensor))
    return out


def define_error_list(actions):
    return {
        action: {
            "p1": AccumLoss(),
            "p2": AccumLoss(),
            "pck": AccumLoss(),
            "auc": AccumLoss(),
        }
        for action in actions
    }


def define_actions_3dhp(action="*", train=False):
    if train:
        return ["Seq1", "Seq2"]
    return ["Seq1"]


def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value

