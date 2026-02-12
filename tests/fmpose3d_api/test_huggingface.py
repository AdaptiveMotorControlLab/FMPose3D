"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0

Tests for HuggingFace Hub integration (model weight downloading).

Verifies that ``FMPose3DInference._download_model_weights`` resolves the
correct repo/filename for each model type, raises helpful errors when
``huggingface_hub`` is not installed, and that the auto-download path is
triggered when no explicit weights path is provided.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fmpose3d.fmpose3d import FMPose3DInference


class TestDownloadModelWeights:
    """Tests for ``FMPose3DInference._download_model_weights``."""

    def test_calls_hf_hub_download_humans(self):
        """Human model type resolves to the correct repo/filename."""
        api = FMPose3DInference(device="cpu")

        with patch("fmpose3d.fmpose3d.hf_hub_download", create=True) as mock_dl:
            mock_dl.return_value = "/fake/cache/fmpose3d_humans.pth"
            # Patch the import inside the method
            with patch.dict("sys.modules", {"huggingface_hub": MagicMock(hf_hub_download=mock_dl)}):
                api._download_model_weights()

        mock_dl.assert_called_once_with(
            repo_id="deruyter92/fmpose_temp",
            filename="fmpose3d_humans.pth",
        )
        assert api.model_weights_path == "/fake/cache/fmpose3d_humans.pth"

    def test_calls_hf_hub_download_animals(self):
        """Animal model type resolves to the correct repo/filename."""
        api = FMPose3DInference.for_animals(device="cpu")

        with patch("fmpose3d.fmpose3d.hf_hub_download", create=True) as mock_dl:
            mock_dl.return_value = "/fake/cache/fmpose3d_animals.pth"
            with patch.dict("sys.modules", {"huggingface_hub": MagicMock(hf_hub_download=mock_dl)}):
                api._download_model_weights()

        mock_dl.assert_called_once_with(
            repo_id="deruyter92/fmpose_temp",
            filename="fmpose3d_animals.pth",
        )
        assert api.model_weights_path == "/fake/cache/fmpose3d_animals.pth"

    def test_missing_huggingface_hub_raises(self):
        """ImportError with a helpful message when huggingface_hub is absent."""
        api = FMPose3DInference(device="cpu")

        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="huggingface_hub is required"):
                api._download_model_weights()

    def test_resolve_triggers_download_when_no_path(self):
        """_resolve_model_weights_path calls _download when path is None."""
        # Suppress the auto-resolve that __init__ triggers so we can
        # test _resolve_model_weights_path in isolation.
        with patch.object(FMPose3DInference, "_resolve_model_weights_path"):
            api = FMPose3DInference(device="cpu")

        # Reset to None — simulating "no explicit path provided".
        api.model_weights_path = None

        with patch.object(api, "_download_model_weights") as mock_dl:
            # After download, set a fake path that won't pass is_file()
            def _set_fake_path():
                api.model_weights_path = "/fake/weights.pth"
            mock_dl.side_effect = _set_fake_path

            with pytest.raises(ValueError, match="Model weights file not found"):
                api._resolve_model_weights_path()

            mock_dl.assert_called_once()
