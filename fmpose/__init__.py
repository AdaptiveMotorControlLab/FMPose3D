"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose: 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

__version__ = "0.0.5"
__author__ = "Ti Wang, Xiaohang Yu, Mackenzie Weygandt Mathis"
__license__ = "MIT"

# Import key components for easy access
from .aggregation_methods import (
    average_aggregation,
    aggregation_select_single_best_hypothesis_by_2D_error,
    aggregation_RPEA_weighted_by_2D_error,
)

# Import 2D pose detection utilities
from .lib.hrnet.gen_kpts import gen_video_kpts
from .lib.preprocess import h36m_coco_format, revise_kpts

# Make commonly used classes/functions available at package level
__all__ = [
    # Aggregation methods
    "average_aggregation",
    "aggregation_select_single_best_hypothesis_by_2D_error",
    "aggregation_RPEA_weighted_by_2D_error",
    # 2D pose detection
    "gen_video_kpts",
    "h36m_coco_format",
    "revise_kpts",
    # Version
    "__version__",
]

