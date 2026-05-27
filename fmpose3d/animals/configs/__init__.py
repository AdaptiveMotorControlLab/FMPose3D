"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0


Bundled DLC ``pytorch_config.yaml`` files for the animal 2D detector.

These yamls describe FMPose3D's fine-tuned SuperAnimal-Quadruped variants
and are loaded by :class:`fmpose3d.inference_api.SuperAnimalEstimator` when
the user does not supply an explicit ``pytorch_config_path``. They are
shipped as package data (see ``pyproject.toml`` ``[tool.setuptools.package-data]``).
"""

from pathlib import Path

CONFIGS_DIR = Path(__file__).parent

SA_FINETUNE_HRNET_W32_YAML: str = str(CONFIGS_DIR / "sa_finetune_hrnet_w32.yaml")
"""DLC config for SA-Quadruped HRNet-w32 fine-tuned on Animal3D +
Control-Animal3D with the 26-joint Animal3D output layout."""

__all__ = ["CONFIGS_DIR", "SA_FINETUNE_HRNET_W32_YAML"]
