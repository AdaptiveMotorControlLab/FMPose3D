"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

"""
FMPose3D models.
"""

from .base_model import BaseModel, register_model, get_model, list_models

# Import model subpackages so their @register_model decorators execute.
from .fmpose3d import Graph, Model
# Import animal models so their @register_model decorators execute.
from fmpose3d.animals import models as _animal_models  # noqa: F401

__all__ = [
    "BaseModel",
    "register_model",
    "get_model",
    "list_models",
    "Graph",
    "Model",
]

