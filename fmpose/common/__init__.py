"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

"""
Common utilities for FMPose.
"""

from .arguments import opts
from .h36m_dataset import Human36mDataset
from .load_data_hm36 import Fusion
from .utils import (
    mpjpe_cal,
    p_mpjpe,
    AccumLoss,
    save_model,
    save_top_N_models,
    test_calculation,
    print_error,
    get_varialbe,
)

__all__ = [
    "opts",
    "Human36mDataset",
    "Fusion",
    "mpjpe_cal",
    "p_mpjpe",
    "AccumLoss",
    "save_model",
    "save_top_N_models",
    "test_calculation",
    "print_error",
    "get_varialbe",
]

