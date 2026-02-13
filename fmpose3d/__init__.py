"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

__version__ = "0.0.7"
__author__ = "Ti Wang, Xiaohang Yu, Mackenzie Weygandt Mathis"
__license__ = "Apache 2.0"

# Import key components for easy access
from .aggregation_methods import (
    average_aggregation,
    aggregation_select_single_best_hypothesis_by_2D_error,
    aggregation_RPEA_joint_level,
)

# Configuration dataclasses
from .common.config import (
    FMPose3DConfig,
    HRNetConfig,
    InferenceConfig,
    ModelConfig,
    SupportedModel,
    PipelineConfig,
)

# High-level inference API
from .inference_api.fmpose3d import (
    FMPose3DInference,
    HRNetEstimator,
    Pose2DResult,
    Pose3DResult,
    Source,
)

# Model registry
from .models import BaseModel, register_model, get_model, list_models

# Import 2D pose detection utilities
from .lib.hrnet.gen_kpts import gen_video_kpts
from .lib.hrnet.hrnet import HRNetPose2d
from .lib.preprocess import h36m_coco_format, revise_kpts

# Make commonly used classes/functions available at package level
__all__ = [
    # Inference API
    "FMPose3DInference",
    "HRNetEstimator",
    "Pose2DResult",
    "Pose3DResult",
    "Source",
    # Configuration
    "FMPose3DConfig",
    "HRNetConfig",
    "InferenceConfig",
    "ModelConfig",
    "SupportedModel",
    "PipelineConfig",
    # Aggregation methods
    "average_aggregation",
    "aggregation_select_single_best_hypothesis_by_2D_error",
    "aggregation_RPEA_joint_level",
    # Model registry
    "BaseModel",
    "register_model",
    "get_model",
    "list_models",
    # 2D pose detection
    "HRNetPose2d",
    "gen_video_kpts",
    "h36m_coco_format",
    "revise_kpts",
    # Version
    "__version__",
]

