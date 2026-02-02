"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

"""
FMPose models.
"""

from .graph_frames import Graph
from .model_animal3d import Model
__all__ = [
    "Graph",
    "Model",
]

