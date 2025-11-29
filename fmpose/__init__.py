"""
FMPose: 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose: 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Accepted by IEEE Transactions on Multimedia (TMM), 2025.
"""

__version__ = "0.0.1"
__author__ = "Ti Wang, Xiaohang Yu, Mackenzie Weygandt Mathis"
__license__ = "MIT"

# Import key components for easy access
from .aggregation_methods import (
    average_aggregation,
    aggregation_select_single_best_hypothesis_by_2D_error,
    aggregation_RPEA_weighted_by_2D_error,
)

# Make commonly used classes/functions available at package level
__all__ = [
    "average_aggregation",
    "aggregation_select_single_best_hypothesis_by_2D_error",
    "aggregation_RPEA_weighted_by_2D_error",
    "__version__",
]

