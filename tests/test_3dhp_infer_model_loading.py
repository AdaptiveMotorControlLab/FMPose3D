"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

import importlib.util
from pathlib import Path

import torch

from fmpose3d.models import get_model
from fmpose3d.utils.weights import resolve_weights_path


def load_infer_3dhp_module():
    module_path = Path(__file__).resolve().parents[1] / "3dhp_test" / "infer_3dhp.py"
    spec = importlib.util.spec_from_file_location("infer_3dhp", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_3dhp_model_loads_from_registry():
    infer_3dhp = load_infer_3dhp_module()

    model_cls = infer_3dhp.load_model_class(None, "fmpose3d_humans")

    assert model_cls is get_model("fmpose3d_humans")


def test_camera_tensor_uses_each_sample_subject():
    infer_3dhp = load_infer_3dhp_module()
    subjects = ["TS4", "TS5"]

    cam_tensor = infer_3dhp.camera_tensor_for_subjects(subjects, torch.device("cpu"))

    assert cam_tensor.shape == (2, 9)
    assert not torch.equal(cam_tensor[0], cam_tensor[1])


def test_3dhp_imports_common_weight_resolver():
    infer_3dhp = load_infer_3dhp_module()

    assert infer_3dhp.resolve_weights_path is resolve_weights_path


def test_p_mpjpe_mixed_actions_uses_per_sample_errors():
    infer_3dhp = load_infer_3dhp_module()
    eval_cal = infer_3dhp.eval_cal
    error_sum = infer_3dhp.define_error_list(["Walk", "Run"])

    target = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ],
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ],
        ],
        dtype=torch.float32,
    )
    predicted = target.clone()
    predicted[1, 0, 1] = torch.tensor([1.4, 0.2, 0.3])
    predicted[1, 0, 2] = torch.tensor([-0.1, 0.7, 0.5])

    expected = eval_cal.p_mpjpe(
        predicted.numpy().reshape(-1, 3, 3),
        target.numpy().reshape(-1, 3, 3),
    )

    eval_cal.mpjpe_by_action_p2(
        predicted,
        target,
        ["Walk", "Run"],
        error_sum,
    )

    assert error_sum["Walk"]["p2"].avg == float(expected[0])
    assert error_sum["Run"]["p2"].avg == float(expected[1])
    assert error_sum["Walk"]["p2"].avg != error_sum["Run"]["p2"].avg
