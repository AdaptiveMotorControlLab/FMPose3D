"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

import argparse
import math

import pytest

from fmpose3d.common.config import (
    PipelineConfig,
    FMPose3DConfig,
    DatasetConfig,
    TrainingConfig,
    InferenceConfig,
    AggregationConfig,
    CheckpointConfig,
    RefinementConfig,
    OutputConfig,
    DemoConfig,
    RuntimeConfig,
    _SUB_CONFIG_CLASSES,
)


# ---------------------------------------------------------------------------
# Sub-config defaults
# ---------------------------------------------------------------------------


class TestFMPose3DConfig:
    def test_defaults(self):
        cfg = FMPose3DConfig()
        assert cfg.layers == 5
        assert cfg.channel == 512
        assert cfg.d_hid == 1024
        assert cfg.n_joints == 17
        assert cfg.out_joints == 17
        assert cfg.frames == 1

    def test_custom_values(self):
        cfg = FMPose3DConfig(layers=5, channel=256, n_joints=26)
        assert cfg.layers == 5
        assert cfg.channel == 256
        assert cfg.n_joints == 26


class TestDatasetConfig:
    def test_defaults(self):
        cfg = DatasetConfig()
        assert cfg.dataset == "h36m"
        assert cfg.keypoints == "cpn_ft_h36m_dbb"
        assert cfg.root_path == "dataset/"
        assert cfg.train_views == [0, 1, 2, 3]
        assert cfg.joints_left == []
        assert cfg.joints_right == []

    def test_list_defaults_are_independent(self):
        """Each instance should get its own list, not a shared reference."""
        a = DatasetConfig()
        b = DatasetConfig()
        a.joints_left.append(99)
        assert 99 not in b.joints_left

    def test_custom_values(self):
        cfg = DatasetConfig(
            dataset="rat7m",
            root_path="Rat7M_data/",
            joints_left=[8, 10, 11],
            joints_right=[9, 14, 15],
        )
        assert cfg.dataset == "rat7m"
        assert cfg.root_path == "Rat7M_data/"
        assert cfg.joints_left == [8, 10, 11]


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.train is False
        assert cfg.nepoch == 41
        assert cfg.batch_size == 128
        assert cfg.lr == pytest.approx(1e-3)
        assert cfg.lr_decay == pytest.approx(0.95)
        assert cfg.data_augmentation is True

    def test_custom_values(self):
        cfg = TrainingConfig(lr=5e-4, nepoch=100)
        assert cfg.lr == pytest.approx(5e-4)
        assert cfg.nepoch == 100


class TestInferenceConfig:
    def test_defaults(self):
        cfg = InferenceConfig()
        assert cfg.test == 1
        assert cfg.test_augmentation is True
        assert cfg.sample_steps == 3
        assert cfg.eval_sample_steps == "1,3,5,7,9"
        assert cfg.hypothesis_num == 1
        assert cfg.guidance_scale == pytest.approx(1.0)


class TestAggregationConfig:
    def test_defaults(self):
        cfg = AggregationConfig()
        assert cfg.topk == 3
        assert cfg.exp_temp == pytest.approx(0.002)
        assert cfg.mode == "exp"
        assert cfg.opt_steps == 2


class TestCheckpointConfig:
    def test_defaults(self):
        cfg = CheckpointConfig()
        assert cfg.reload is False
        assert cfg.model_weights_path == ""
        assert cfg.previous_dir == "./pre_trained_model/pretrained"
        assert cfg.num_saved_models == 3
        assert cfg.previous_best_threshold == math.inf

    def test_mutability(self):
        cfg = CheckpointConfig()
        cfg.previous_best_threshold = 42.5
        cfg.previous_name = "best_model.pth"
        assert cfg.previous_best_threshold == pytest.approx(42.5)
        assert cfg.previous_name == "best_model.pth"


class TestRefinementConfig:
    def test_defaults(self):
        cfg = RefinementConfig()
        assert cfg.post_refine is False
        assert cfg.lr_refine == pytest.approx(1e-5)
        assert cfg.refine is False


class TestOutputConfig:
    def test_defaults(self):
        cfg = OutputConfig()
        assert cfg.create_time == ""
        assert cfg.create_file == 1
        assert cfg.debug is False
        assert cfg.folder_name == ""


class TestDemoConfig:
    def test_defaults(self):
        cfg = DemoConfig()
        assert cfg.type == "image"
        assert cfg.path == "demo/images/running.png"


class TestRuntimeConfig:
    def test_defaults(self):
        cfg = RuntimeConfig()
        assert cfg.gpu == "0"
        assert cfg.pad == 0
        assert cfg.single is False
        assert cfg.reload_3d is False


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_default_construction(self):
        """All sub-configs are initialised with their defaults."""
        cfg = PipelineConfig()
        assert isinstance(cfg.model_cfg, FMPose3DConfig)
        assert isinstance(cfg.dataset_cfg, DatasetConfig)
        assert isinstance(cfg.training_cfg, TrainingConfig)
        assert isinstance(cfg.inference_cfg, InferenceConfig)
        assert isinstance(cfg.aggregation_cfg, AggregationConfig)
        assert isinstance(cfg.checkpoint_cfg, CheckpointConfig)
        assert isinstance(cfg.refinement_cfg, RefinementConfig)
        assert isinstance(cfg.output_cfg, OutputConfig)
        assert isinstance(cfg.demo_cfg, DemoConfig)
        assert isinstance(cfg.runtime_cfg, RuntimeConfig)

    def test_partial_construction(self):
        """Supplying only some sub-configs leaves the rest at defaults."""
        cfg = PipelineConfig(
            model_cfg=FMPose3DConfig(layers=5),
            training_cfg=TrainingConfig(lr=2e-4),
        )
        assert cfg.model_cfg.layers == 5
        assert cfg.training_cfg.lr == pytest.approx(2e-4)
        # Others keep defaults
        assert cfg.dataset_cfg.dataset == "h36m"
        assert cfg.runtime_cfg.gpu == "0"

    def test_sub_config_mutation(self):
        """Mutating a sub-config field is reflected on the config."""
        cfg = PipelineConfig()
        cfg.training_cfg.lr = 0.01
        assert cfg.training_cfg.lr == pytest.approx(0.01)

    def test_sub_config_replacement(self):
        """Replacing an entire sub-config works."""
        cfg = PipelineConfig()
        cfg.model_cfg = FMPose3DConfig(layers=10, channel=1024)
        assert cfg.model_cfg.layers == 10
        assert cfg.model_cfg.channel == 1024

    # -- to_dict --------------------------------------------------------------

    def test_to_dict_returns_flat_dict(self):
        cfg = PipelineConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        # Spot-check keys from different groups
        assert "layers" in d
        assert "dataset" in d
        assert "lr" in d
        assert "topk" in d
        assert "gpu" in d

    def test_to_dict_reflects_custom_values(self):
        cfg = PipelineConfig(
            model_cfg=FMPose3DConfig(layers=7),
            aggregation_cfg=AggregationConfig(topk=5),
        )
        d = cfg.to_dict()
        assert d["layers"] == 7
        assert d["topk"] == 5

    def test_to_dict_no_duplicate_keys(self):
        """Every field name should be unique across all sub-configs."""
        cfg = PipelineConfig()
        d = cfg.to_dict()
        all_field_names = []
        for dc_class in _SUB_CONFIG_CLASSES.values():
            from dataclasses import fields as dc_fields
            all_field_names.extend(f.name for f in dc_fields(dc_class))
        assert len(all_field_names) == len(set(all_field_names)), (
            "Duplicate field names across sub-configs"
        )

    # -- from_namespace -------------------------------------------------------

    def test_from_namespace_basic(self):
        ns = argparse.Namespace(
            # FMPose3DConfig
            model="test_model",
            model_type="fmpose3d_humans",
            layers=5,
            channel=256,
            d_hid=512,
            token_dim=128,
            n_joints=20,
            out_joints=20,
            in_channels=2,
            out_channels=3,
            frames=3,
            # DatasetConfig
            dataset="rat7m",
            keypoints="cpn",
            root_path="Rat7M_data/",
            actions="*",
            downsample=1,
            subset=1.0,
            stride=1,
            crop_uv=0,
            out_all=1,
            train_views=[0, 1],
            test_views=[2, 3],
            subjects_train="S1",
            subjects_test="S2",
            root_joint=4,
            joints_left=[8, 10],
            joints_right=[9, 14],
            # TrainingConfig
            train=True,
            nepoch=100,
            batch_size=64,
            lr=5e-4,
            lr_decay=0.99,
            lr_decay_large=0.5,
            large_decay_epoch=10,
            workers=4,
            data_augmentation=False,
            reverse_augmentation=False,
            norm=0.01,
            # InferenceConfig
            test=1,
            test_augmentation=False,
            test_augmentation_flip_hypothesis=False,
            test_augmentation_FlowAug=False,
            sample_steps=5,
            eval_multi_steps=True,
            eval_sample_steps="1,3,5",
            num_hypothesis_list="1,3",
            hypothesis_num=3,
            guidance_scale=1.5,
            # AggregationConfig
            topk=5,
            exp_temp=0.001,
            mode="softmax",
            opt_steps=3,
            # CheckpointConfig
            reload=True,
            model_dir="/tmp",
            model_weights_path="/tmp/weights.pth",
            checkpoint="/tmp/ckpt",
            previous_dir="./pre_trained",
            num_saved_models=5,
            previous_best_threshold=50.0,
            previous_name="best.pth",
            # RefinementConfig
            post_refine=False,
            post_refine_reload=False,
            previous_post_refine_name="",
            lr_refine=1e-5,
            refine=False,
            reload_refine=False,
            previous_refine_name="",
            # OutputConfig
            create_time="250101",
            filename="run1",
            create_file=1,
            debug=True,
            folder_name="exp1",
            sh_file="train.sh",
            # DemoConfig
            type="video",
            path="/tmp/video.mp4",
            # RuntimeConfig
            gpu="1",
            pad=1,
            single=True,
            reload_3d=False,
        )
        cfg = PipelineConfig.from_namespace(ns)

        # Verify a sample from each group
        assert cfg.model_cfg.layers == 5
        assert cfg.model_cfg.channel == 256
        assert cfg.dataset_cfg.dataset == "rat7m"
        assert cfg.dataset_cfg.joints_left == [8, 10]
        assert cfg.training_cfg.train is True
        assert cfg.training_cfg.nepoch == 100
        assert cfg.inference_cfg.sample_steps == 5
        assert cfg.inference_cfg.guidance_scale == pytest.approx(1.5)
        assert cfg.aggregation_cfg.topk == 5
        assert cfg.checkpoint_cfg.reload is True
        assert cfg.checkpoint_cfg.previous_best_threshold == pytest.approx(50.0)
        assert cfg.refinement_cfg.lr_refine == pytest.approx(1e-5)
        assert cfg.output_cfg.debug is True
        assert cfg.demo_cfg.type == "video"
        assert cfg.runtime_cfg.gpu == "1"

    def test_from_namespace_ignores_unknown_fields(self):
        """Extra attributes in the namespace that don't match any field are ignored."""
        ns = argparse.Namespace(
            layers=3, channel=512, unknown_field="should_be_ignored",
        )
        cfg = PipelineConfig.from_namespace(ns)
        assert cfg.model_cfg.layers == 3
        assert cfg.model_cfg.channel == 512
        assert not hasattr(cfg, "unknown_field")

    def test_from_namespace_partial_namespace(self):
        """A namespace missing some fields uses dataclass defaults for those."""
        ns = argparse.Namespace(layers=10, gpu="2")
        cfg = PipelineConfig.from_namespace(ns)
        assert cfg.model_cfg.layers == 10
        assert cfg.runtime_cfg.gpu == "2"
        # Unset fields keep defaults
        assert cfg.model_cfg.channel == 512
        assert cfg.training_cfg.lr == pytest.approx(1e-3)

    # -- round-trip: from_namespace ↔ to_dict ---------------------------------

    def test_roundtrip_from_namespace_to_dict(self):
        """Values fed via from_namespace appear identically in to_dict."""
        ns = argparse.Namespace(
            layers=8, channel=1024, dataset="animal3d", lr=2e-4, topk=7, gpu="3",
        )
        cfg = PipelineConfig.from_namespace(ns)
        d = cfg.to_dict()
        assert d["layers"] == 8
        assert d["channel"] == 1024
        assert d["dataset"] == "animal3d"
        assert d["lr"] == pytest.approx(2e-4)
        assert d["topk"] == 7
        assert d["gpu"] == "3"

    def test_to_dict_after_mutation(self):
        """to_dict reflects in-place mutations on sub-configs."""
        cfg = PipelineConfig()
        cfg.training_cfg.lr = 0.123
        cfg.model_cfg.layers = 99
        d = cfg.to_dict()
        assert d["lr"] == pytest.approx(0.123)
        assert d["layers"] == 99
