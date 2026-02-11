"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

import math
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, List

# ---------------------------------------------------------------------------
# Dataclass configuration groups
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_type: str = "fmpose3d_humans"


# Per-model-type defaults for fields marked with INFER_FROM_MODEL_TYPE.
# Also consumed by PipelineConfig.for_model_type to set cross-config
# values (dataset, sample_steps, etc.).
_FMPOSE3D_DEFAULTS: Dict[str, Dict] = {
    "fmpose3d_humans": {
        "n_joints": 17,
        "out_joints": 17,
        "dataset": "h36m",
        "sample_steps": 3,
        "joints_left": [4, 5, 6, 11, 12, 13],
        "joints_right": [1, 2, 3, 14, 15, 16],
        "root_joint": 0,
    },
    "fmpose3d_animals": {
        "n_joints": 26,
        "out_joints": 26,
        "dataset": "animal3d",
        "sample_steps": 5,
        "joints_left": [0, 3, 5, 8, 10, 12, 14, 16, 20, 22],
        "joints_right": [1, 4, 6, 9, 11, 13, 15, 17, 21, 23],
        "root_joint": 7,
    },
}

# Sentinel object for defaults that are inferred from the model type.
INFER_FROM_MODEL_TYPE = object()

@dataclass
class FMPose3DConfig(ModelConfig):
    model_type: str = "fmpose3d_humans"
    model: str = ""
    layers: int = 5
    channel: int = 512
    d_hid: int = 1024
    token_dim: int = 256
    n_joints: int = INFER_FROM_MODEL_TYPE  # type: ignore[assignment]
    out_joints: int = INFER_FROM_MODEL_TYPE  # type: ignore[assignment]
    joints_left: List[int] = INFER_FROM_MODEL_TYPE  # type: ignore[assignment]
    joints_right: List[int] = INFER_FROM_MODEL_TYPE  # type: ignore[assignment]
    root_joint: int = INFER_FROM_MODEL_TYPE  # type: ignore[assignment]
    in_channels: int = 2
    out_channels: int = 3
    frames: int = 1

    def __post_init__(self):
        defaults = _FMPOSE3D_DEFAULTS.get(self.model_type)
        if defaults is None:
            supported = ", ".join(sorted(_FMPOSE3D_DEFAULTS))
            raise ValueError(
                f"Unknown model_type {self.model_type!r}; supported: {supported}"
            )
        for f in fields(self):
            if getattr(self, f.name) is INFER_FROM_MODEL_TYPE:
                setattr(self, f.name, defaults[f.name])

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
class Pose2DConfig:
    """2D pose estimator configuration."""
    pose2d_model: str = "hrnet"


@dataclass
class HRNetConfig(Pose2DConfig):
    """HRNet 2D pose detector configuration.

    Attributes
    ----------
    det_dim : int
        YOLO input resolution for human detection (default 416).
    num_persons : int
        Maximum number of persons to estimate per frame (default 1).
    thred_score : float
        YOLO object-confidence threshold (default 0.30).
    hrnet_cfg_file : str
        Path to the HRNet YAML experiment config.  When left empty the
        bundled ``w48_384x288_adam_lr1e-3.yaml`` is used.
    hrnet_weights_path : str
        Path to the HRNet ``.pth`` checkpoint.  When left empty the
        auto-downloaded ``pose_hrnet_w48_384x288.pth`` is used.
    """
    pose2d_model: str = "hrnet"
    det_dim: int = 416
    num_persons: int = 1
    thred_score: float = 0.30
    hrnet_cfg_file: str = ""
    hrnet_weights_path: str = ""


@dataclass
class SuperAnimalConfig(Pose2DConfig):
    """DeepLabCut SuperAnimal 2D pose detector configuration.

    Uses the DeepLabCut ``superanimal_analyze_images`` API to detect
    animal keypoints in the quadruped80K format, then maps them to the
    Animal3D 26-keypoint layout expected by the ``fmpose3d_animals``
    3D lifter.

    Attributes
    ----------
    superanimal_name : str
        Name of the SuperAnimal model (default ``"superanimal_quadruped"``).
    sa_model_name : str
        Backbone architecture (default ``"hrnet_w32"``).
    detector_name : str
        Object detector used for animal bounding boxes.
    max_individuals : int
        Maximum number of individuals to detect per image (default 1).
    """
    pose2d_model: str = "superanimal"
    superanimal_name: str = "superanimal_quadruped"
    sa_model_name: str = "hrnet_w32"
    detector_name: str = "fasterrcnn_resnet50_fpn_v2"
    max_individuals: int = 1


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
    "pose2d_cfg": Pose2DConfig,
    "demo_cfg": DemoConfig,
    "runtime_cfg": RuntimeConfig,
}


@dataclass
class PipelineConfig:
    """Top-level configuration for FMPose3D pipeline.

    Groups related settings into sub-configs::

        config.model_cfg.layers
        config.training_cfg.lr
    """

    model_cfg: ModelConfig = field(default_factory=FMPose3DConfig)
    dataset_cfg: DatasetConfig = field(default_factory=DatasetConfig)
    training_cfg: TrainingConfig = field(default_factory=TrainingConfig)
    inference_cfg: InferenceConfig = field(default_factory=InferenceConfig)
    aggregation_cfg: AggregationConfig = field(default_factory=AggregationConfig)
    checkpoint_cfg: CheckpointConfig = field(default_factory=CheckpointConfig)
    refinement_cfg: RefinementConfig = field(default_factory=RefinementConfig)
    output_cfg: OutputConfig = field(default_factory=OutputConfig)
    pose2d_cfg: Pose2DConfig = field(default_factory=HRNetConfig)
    demo_cfg: DemoConfig = field(default_factory=DemoConfig)
    runtime_cfg: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_namespace(cls, ns) -> "PipelineConfig":
        """Build a :class:`PipelineConfig` from an ``argparse.Namespace``

        Example::

            args = opts().parse()
            cfg = PipelineConfig.from_namespace(args)
        """
        raw = vars(ns) if hasattr(ns, "__dict__") else dict(ns)

        def _pick(dc_class, src: dict):
            names = {f.name for f in fields(dc_class)}
            return dc_class(**{k: v for k, v in src.items() if k in names})

        kwargs = {}
        for group_name, dc_class in _SUB_CONFIG_CLASSES.items():
            if group_name == "model_cfg" and raw.get("model_type", 'fmpose3d_humans') in _FMPOSE3D_DEFAULTS:
                dc_class = FMPose3DConfig
            elif group_name == "pose2d_cfg":
                p2d = raw.get("pose2d_model", "hrnet")
                if p2d == "superanimal":
                    dc_class = SuperAnimalConfig
                elif p2d == "hrnet":
                    dc_class = HRNetConfig
            kwargs[group_name] = _pick(dc_class, raw)
        return cls(**kwargs)

    # -- utilities ------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a flat dictionary of all configuration values."""
        result = {}
        for group_name in _SUB_CONFIG_CLASSES:
            result.update(asdict(getattr(self, group_name)))
        return result

