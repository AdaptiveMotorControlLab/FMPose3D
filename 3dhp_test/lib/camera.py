"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

import numpy as np
import torch


def normalize_screen_coordinates(x, w, h):
    assert x.shape[-1] == 2
    return x / w * 2 - [1, h / w]


def project_to_2d(x, camera_params):
    """
    Project 3D points to normalized 2D using the same camera model as the
    original FMPose 3DHP inference script.
    """
    assert x.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert x.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(x.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]

    xx = torch.clamp(x[..., :2] / x[..., 2:], min=-1, max=1)
    r2 = torch.sum(xx[..., :2] ** 2, dim=len(xx.shape) - 1, keepdim=True)

    radial = 1 + torch.sum(
        k * torch.cat((r2, r2 ** 2, r2 ** 3), dim=len(r2.shape) - 1),
        dim=len(r2.shape) - 1,
        keepdim=True,
    )
    tan = torch.sum(p * xx, dim=len(xx.shape) - 1, keepdim=True)
    xxx = xx * (radial + tan) + p * r2

    return f * xxx + c


def resolution_for_subject(subject):
    if subject in {"TS5", "TS6"}:
        return 1920, 1080
    return 2048, 2048


def camera_params_for_subject(subject):
    # Official MPI-INF-3DHP test intrinsics, normalized with
    # normalize_screen_coordinates(): x / width * 2 - [1, height / width].
    if subject in {"TS5", "TS6"}:
        return [
            1.7541495005289713,
            1.7541495005289713,
            -0.021502558390299464,
            0.020459111531575536,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    return [
        1.4650119543075562,
        1.4650119543075562,
        -0.006945967674255371,
        0.018097639083862305,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
