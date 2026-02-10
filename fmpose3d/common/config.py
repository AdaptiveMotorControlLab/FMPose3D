"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

import math
from dataclasses import dataclass, field, fields, asdict
from typing import List


# ---------------------------------------------------------------------------
# Dataclass configuration groups
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    model: str = ""
    model_type: str = "fmpose3d"
    layers: int = 3
    channel: int = 512
    d_hid: int = 1024
    token_dim: int = 256
    n_joints: int = 17
    out_joints: int = 17
    in_channels: int = 2
    out_channels: int = 3
    frames: int = 1
    """Optional: load model class from a specific file path."""


@dataclass
class DatasetConfig:
    """Dataset and data loading configuration."""

    dataset: str = "h36m"
    keypoints: str = "cpn_ft_h36m_dbb"
    root_path: str = "dataset/"
    actions: str = "*"
    downsample: int = 1
    subset: float = 1.0
    stride: int = 1
    crop_uv: int = 0
    out_all: int = 1
    train_views: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    test_views: List[int] = field(default_factory=lambda: [0, 1, 2, 3])

    # Derived / set during parse based on dataset choice
    subjects_train: str = "S1,S5,S6,S7,S8"
    subjects_test: str = "S9,S11"
    root_joint: int = 0
    joints_left: List[int] = field(default_factory=list)
    joints_right: List[int] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""

    train: bool = False
    nepoch: int = 41
    batch_size: int = 128
    lr: float = 1e-3
    lr_decay: float = 0.95
    lr_decay_large: float = 0.5
    large_decay_epoch: int = 5
    workers: int = 8
    data_augmentation: bool = True
    reverse_augmentation: bool = False
    norm: float = 0.01


@dataclass
class InferenceConfig:
    """Evaluation and testing configuration."""

    test: int = 1
    test_augmentation: bool = True
    test_augmentation_flip_hypothesis: bool = False
    test_augmentation_FlowAug: bool = False
    sample_steps: int = 3
    eval_multi_steps: bool = False
    eval_sample_steps: str = "1,3,5,7,9"
    num_hypothesis_list: str = "1"
    hypothesis_num: int = 1
    guidance_scale: float = 1.0


@dataclass
class AggregationConfig:
    """Hypothesis aggregation configuration."""

    topk: int = 3
    exp_temp: float = 0.002
    mode: str = "exp"
    opt_steps: int = 2


@dataclass
class CheckpointConfig:
    """Checkpoint loading and saving configuration."""

    reload: bool = False
    model_dir: str = ""
    model_weights_path: str = ""
    checkpoint: str = ""
    previous_dir: str = "./pre_trained_model/pretrained"
    num_saved_models: int = 3
    previous_best_threshold: float = math.inf
    previous_name: str = ""


@dataclass
class RefinementConfig:
    """Post-refinement model configuration."""

    post_refine: bool = False
    post_refine_reload: bool = False
    previous_post_refine_name: str = ""
    lr_refine: float = 1e-5
    refine: bool = False
    reload_refine: bool = False
    previous_refine_name: str = ""


@dataclass
class OutputConfig:
    """Output, logging, and file management configuration."""

    create_time: str = ""
    filename: str = ""
    create_file: int = 1
    debug: bool = False
    folder_name: str = ""
    sh_file: str = ""


@dataclass
class DemoConfig:
    """Demo / inference configuration."""

    type: str = "image"
    """Input type: ``'image'`` or ``'video'``."""
    path: str = "demo/images/running.png"
    """Path to input file or directory."""


@dataclass
class RuntimeConfig:
    """Runtime environment configuration."""

    gpu: str = "0"
    pad: int = 0  # derived: (frames - 1) // 2
    single: bool = False
    reload_3d: bool = False


# ---------------------------------------------------------------------------
# Composite configuration
# ---------------------------------------------------------------------------

_SUB_CONFIG_CLASSES = {
    "model_cfg": ModelConfig,
    "dataset_cfg": DatasetConfig,
    "training_cfg": TrainingConfig,
    "inference_cfg": InferenceConfig,
    "aggregation_cfg": AggregationConfig,
    "checkpoint_cfg": CheckpointConfig,
    "refinement_cfg": RefinementConfig,
    "output_cfg": OutputConfig,
    "demo_cfg": DemoConfig,
    "runtime_cfg": RuntimeConfig,
}


@dataclass
class FMPoseConfig:
    """Top-level configuration for FMPose3D.

    Groups related settings into sub-configs::

        config.model_cfg.layers
        config.training_cfg.lr
    """

    model_cfg: ModelConfig = field(default_factory=ModelConfig)
    dataset_cfg: DatasetConfig = field(default_factory=DatasetConfig)
    training_cfg: TrainingConfig = field(default_factory=TrainingConfig)
    inference_cfg: InferenceConfig = field(default_factory=InferenceConfig)
    aggregation_cfg: AggregationConfig = field(default_factory=AggregationConfig)
    checkpoint_cfg: CheckpointConfig = field(default_factory=CheckpointConfig)
    refinement_cfg: RefinementConfig = field(default_factory=RefinementConfig)
    output_cfg: OutputConfig = field(default_factory=OutputConfig)
    demo_cfg: DemoConfig = field(default_factory=DemoConfig)
    runtime_cfg: RuntimeConfig = field(default_factory=RuntimeConfig)

    # -- construction from argparse namespace ---------------------------------

    @classmethod
    def from_namespace(cls, ns) -> "FMPoseConfig":
        """Build a :class:`FMPoseConfig` from an ``argparse.Namespace``

        Example::

            args = opts().parse()
            cfg = FMPoseConfig.from_namespace(args)
        """
        raw = vars(ns) if hasattr(ns, "__dict__") else dict(ns)

        def _pick(dc_class, src: dict):
            names = {f.name for f in fields(dc_class)}
            return dc_class(**{k: v for k, v in src.items() if k in names})

        return cls(**{
            group_name: _pick(dc_class, raw)
            for group_name, dc_class in _SUB_CONFIG_CLASSES.items()
        })

    # -- utilities ------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a flat dictionary of all configuration values."""
        result = {}
        for group_name in _SUB_CONFIG_CLASSES:
            result.update(asdict(getattr(self, group_name)))
        return result

