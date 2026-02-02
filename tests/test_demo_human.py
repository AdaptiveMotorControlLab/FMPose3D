"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

import pytest
import os
import sys
import shutil
import numpy as np

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def test_image_path():
    """Path to the test image."""
    return os.path.join(PROJECT_ROOT, 'demo/images/running.png')


@pytest.fixture
def test_output_dir(tmp_path):
    """Temporary output directory for test results."""
    output_dir = tmp_path / "test_output/"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir) + "/"


def test_2d_pose_estimation(test_image_path, test_output_dir):
    """Test that 2D pose estimation runs and produces output."""
    # Import here to avoid import issues at collection time
    from fmpose3d.lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
    from fmpose3d.lib.preprocess import h36m_coco_format, revise_kpts
    
    # Run 2D pose estimation
    keypoints, scores = hrnet_pose(test_image_path, det_dim=416, num_peroson=1, gen_output=True, type='image')
    
    # Check output shapes
    assert keypoints is not None, "Keypoints should not be None"
    assert scores is not None, "Scores should not be None"
    assert keypoints.shape[2] == 17, f"Expected 17 joints, got {keypoints.shape[2]}"
    assert keypoints.shape[3] == 2, f"Expected 2D coordinates, got shape {keypoints.shape[3]}"
    
    # Convert to H36M format
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    assert keypoints is not None, "H36M format conversion failed"
    
    # Save keypoints
    output_2d_dir = test_output_dir + 'input_2D/'
    os.makedirs(output_2d_dir, exist_ok=True)
    np.savez_compressed(output_2d_dir + 'keypoints.npz', reconstruction=keypoints)
    
    assert os.path.exists(output_2d_dir + 'keypoints.npz'), "Keypoints file should be created"


def test_demo_pipeline_runs(test_image_path):
    """Test that the full demo pipeline can be imported and key components work."""
    # Test imports
    from fmpose3d.lib.hrnet.gen_kpts import gen_video_kpts
    from fmpose3d.lib.preprocess import h36m_coco_format, revise_kpts
    from fmpose3d.models import Model
    
    assert gen_video_kpts is not None
    assert h36m_coco_format is not None
    assert Model is not None
    
    # Test image exists
    assert os.path.exists(test_image_path), f"Test image not found: {test_image_path}"