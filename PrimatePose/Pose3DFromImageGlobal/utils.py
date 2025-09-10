import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os


def normalize_2d_pose(pose_2d, image_size):
    """
    Normalize 2D pose coordinates to [-1, 1] range
    Args:
        pose_2d: (N, 3) array with [x, y, visibility]
        image_size: (width, height) tuple
    Returns:
        normalized_pose: (N, 3) array with normalized coordinates
    """
    normalized_pose = pose_2d.copy()
    normalized_pose[:, 0] = 2 * (pose_2d[:, 0] / image_size[0]) - 1  # x
    normalized_pose[:, 1] = 2 * (pose_2d[:, 1] / image_size[1]) - 1  # y
    return normalized_pose


def denormalize_2d_pose(normalized_pose, image_size):
    """
    Denormalize 2D pose coordinates from [-1, 1] to image coordinates
    Args:
        normalized_pose: (N, 3) array with normalized coordinates
        image_size: (width, height) tuple
    Returns:
        pose_2d: (N, 3) array with image coordinates
    """
    pose_2d = normalized_pose.copy()
    pose_2d[:, 0] = (normalized_pose[:, 0] + 1) * image_size[0] / 2  # x
    pose_2d[:, 1] = (normalized_pose[:, 1] + 1) * image_size[1] / 2  # y
    return pose_2d


def center_3d_pose(pose_3d, root_joint_idx=16):
    """
    Center 3D pose by subtracting root joint position
    Args:
        pose_3d: (N, 3) array with 3D coordinates
        root_joint_idx: index of root joint (default: body_center for ap10k)
    Returns:
        centered_pose: (N, 3) array with centered coordinates
    """
    if torch.is_tensor(pose_3d):
        root_pos = pose_3d[root_joint_idx:root_joint_idx+1]
        centered_pose = pose_3d - root_pos
    else:
        root_pos = pose_3d[root_joint_idx:root_joint_idx+1]
        centered_pose = pose_3d - root_pos
    
    return centered_pose


def compute_bone_lengths(pose_3d, bone_connections):
    """
    Compute bone lengths from 3D pose
    Args:
        pose_3d: (N, 3) array with 3D coordinates
        bone_connections: list of (joint1, joint2) tuples
    Returns:
        bone_lengths: list of bone length values
    """
    bone_lengths = []
    
    for joint1, joint2 in bone_connections:
        if joint1 < len(pose_3d) and joint2 < len(pose_3d):
            if torch.is_tensor(pose_3d):
                bone_vec = pose_3d[joint2] - pose_3d[joint1]
                bone_length = torch.norm(bone_vec).item()
            else:
                bone_vec = pose_3d[joint2] - pose_3d[joint1]
                bone_length = np.linalg.norm(bone_vec)
            
            bone_lengths.append(bone_length)
    
    return bone_lengths


def apply_procrustes_alignment(pred_pose, gt_pose, valid_mask=None):
    """
    Apply Procrustes alignment to align predicted pose with ground truth
    Args:
        pred_pose: (N, 3) predicted 3D pose
        gt_pose: (N, 3) ground truth 3D pose
        valid_mask: (N,) boolean mask for valid joints
    Returns:
        aligned_pose: (N, 3) aligned predicted pose
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        s: scalar scaling factor
    """
    if torch.is_tensor(pred_pose):
        pred_np = pred_pose.cpu().numpy()
        gt_np = gt_pose.cpu().numpy()
    else:
        pred_np = pred_pose
        gt_np = gt_pose
    
    if valid_mask is not None:
        if torch.is_tensor(valid_mask):
            valid_mask = valid_mask.cpu().numpy()
        pred_np = pred_np[valid_mask]
        gt_np = gt_np[valid_mask]
    
    # Center the poses
    pred_mean = np.mean(pred_np, axis=0)
    gt_mean = np.mean(gt_np, axis=0)
    
    pred_centered = pred_np - pred_mean
    gt_centered = gt_np - gt_mean
    
    # Compute optimal rotation using SVD
    H = pred_centered.T @ gt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute optimal scaling
    pred_norm = np.sum(pred_centered ** 2)
    if pred_norm > 0:
        s = np.sum(S) / pred_norm
    else:
        s = 1.0
    
    # Apply transformation
    pred_aligned = s * (pred_np - pred_mean) @ R.T + gt_mean
    
    # Extend to full pose if valid_mask was used
    if valid_mask is not None:
        full_aligned = pred_pose.copy()
        if torch.is_tensor(pred_pose):
            full_aligned[valid_mask] = torch.tensor(pred_aligned, device=pred_pose.device, dtype=pred_pose.dtype)
        else:
            full_aligned[valid_mask] = pred_aligned
        pred_aligned = full_aligned
    
    # Compute translation
    t = gt_mean - s * pred_mean @ R.T
    
    return pred_aligned, R, t, s


def compute_3d_metrics(pred_poses, gt_poses, valid_masks=None, align=True):
    """
    Compute 3D pose estimation metrics
    Args:
        pred_poses: (B, N, 3) predicted poses
        gt_poses: (B, N, 3) ground truth poses  
        valid_masks: (B, N) boolean masks for valid joints
        align: whether to apply Procrustes alignment
    Returns:
        metrics: dict with various 3D metrics
    """
    if torch.is_tensor(pred_poses):
        pred_poses = pred_poses.cpu().numpy()
    if torch.is_tensor(gt_poses):
        gt_poses = gt_poses.cpu().numpy()
    if valid_masks is not None and torch.is_tensor(valid_masks):
        valid_masks = valid_masks.cpu().numpy()
    
    batch_size = pred_poses.shape[0]
    
    # Per-sample metrics
    mpjpe_scores = []  # Mean Per Joint Position Error
    pa_mpjpe_scores = []  # Procrustes Aligned MPJPE
    
    for i in range(batch_size):
        pred = pred_poses[i]
        gt = gt_poses[i]
        valid_mask = valid_masks[i] if valid_masks is not None else None
        
        # Compute MPJPE
        if valid_mask is not None:
            errors = np.linalg.norm(pred - gt, axis=1)
            errors = errors[valid_mask]
            mpjpe = np.mean(errors) if len(errors) > 0 else 0
        else:
            errors = np.linalg.norm(pred - gt, axis=1)
            mpjpe = np.mean(errors)
        
        mpjpe_scores.append(mpjpe)
        
        # Compute PA-MPJPE (Procrustes Aligned)
        if align:
            try:
                aligned_pred, _, _, _ = apply_procrustes_alignment(pred, gt, valid_mask)
                if valid_mask is not None:
                    pa_errors = np.linalg.norm(aligned_pred - gt, axis=1)
                    pa_errors = pa_errors[valid_mask]
                    pa_mpjpe = np.mean(pa_errors) if len(pa_errors) > 0 else 0
                else:
                    pa_errors = np.linalg.norm(aligned_pred - gt, axis=1)
                    pa_mpjpe = np.mean(pa_errors)
            except:
                pa_mpjpe = mpjpe  # Fallback if alignment fails
        else:
            pa_mpjpe = mpjpe
        
        pa_mpjpe_scores.append(pa_mpjpe)
    
    metrics = {
        'mpjpe': np.mean(mpjpe_scores),
        'pa_mpjpe': np.mean(pa_mpjpe_scores),
        'mpjpe_std': np.std(mpjpe_scores),
        'pa_mpjpe_std': np.std(pa_mpjpe_scores),
        'per_sample_mpjpe': mpjpe_scores,
        'per_sample_pa_mpjpe': pa_mpjpe_scores
    }
    
    return metrics


def create_skeleton_visualization(pose_3d, connections, keypoint_names=None, 
                                 colors=None, save_path=None, title="3D Pose"):
    """
    Create a detailed 3D skeleton visualization
    Args:
        pose_3d: (N, 3) 3D pose coordinates
        connections: list of (joint1, joint2) tuples defining skeleton
        keypoint_names: list of keypoint names
        colors: list of colors for different body parts
        save_path: path to save the figure
        title: figure title
    Returns:
        fig: matplotlib figure object
    """
    if torch.is_tensor(pose_3d):
        pose_3d = pose_3d.cpu().numpy()
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for different body parts if not provided
    if colors is None:
        colors = {
            'head': 'red',
            'torso': 'blue', 
            'arms': 'green',
            'legs': 'orange',
            'tail': 'purple'
        }
    
    # Categorize connections by body part
    head_connections = [(1, 2), (1, 3), (1, 4), (2, 5), (3, 6)]
    torso_connections = [(11, 12), (11, 13), (12, 14), (13, 14), (14, 15), (15, 16), (16, 17)]
    arm_connections = [(12, 18), (18, 20), (20, 22), (13, 19), (19, 21), (21, 23)]
    leg_connections = [(16, 24), (16, 25), (24, 26), (25, 26), (24, 27), (27, 29), (29, 31), (25, 28), (28, 30), (30, 32)]
    tail_connections = [(17, 33), (33, 34), (34, 35), (35, 36)]
    
    connection_groups = [
        (head_connections, colors['head'], 'Head'),
        (torso_connections, colors['torso'], 'Torso'),
        (arm_connections, colors['arms'], 'Arms'),
        (leg_connections, colors['legs'], 'Legs'),
        (tail_connections, colors['tail'], 'Tail')
    ]
    
    # Plot skeleton connections
    for group_connections, color, label in connection_groups:
        plotted_label = False
        for joint1, joint2 in group_connections:
            if joint1 < len(pose_3d) and joint2 < len(pose_3d):
                label_to_use = label if not plotted_label else None
                ax.plot([pose_3d[joint1, 0], pose_3d[joint2, 0]],
                       [pose_3d[joint1, 1], pose_3d[joint2, 1]],
                       [pose_3d[joint1, 2], pose_3d[joint2, 2]], 
                       color=color, linewidth=2, alpha=0.8, label=label_to_use)
                plotted_label = True
    
    # Plot keypoints
    ax.scatter(pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2], 
              c='black', s=50, alpha=0.9, label='Keypoints')
    
    # Add keypoint labels if provided
    if keypoint_names:
        for i, (x, y, z) in enumerate(pose_3d):
            if i < len(keypoint_names):
                ax.text(x, y, z, f'{i}:{keypoint_names[i][:3]}', fontsize=6)
    else:
        for i, (x, y, z) in enumerate(pose_3d):
            ax.text(x, y, z, f'{i}', fontsize=8)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14)
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([pose_3d[:, 0].max() - pose_3d[:, 0].min(),
                         pose_3d[:, 1].max() - pose_3d[:, 1].min(),
                         pose_3d[:, 2].max() - pose_3d[:, 2].min()]).max() / 2.0
    mid_x = (pose_3d[:, 0].max() + pose_3d[:, 0].min()) * 0.5
    mid_y = (pose_3d[:, 1].max() + pose_3d[:, 1].min()) * 0.5
    mid_z = (pose_3d[:, 2].max() + pose_3d[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def save_poses_as_json(poses_3d, poses_2d, image_paths, output_path):
    """
    Save poses in JSON format for further analysis
    Args:
        poses_3d: list or array of 3D poses
        poses_2d: list or array of 2D poses
        image_paths: list of image paths
        output_path: path to save JSON file
    """
    if torch.is_tensor(poses_3d):
        poses_3d = poses_3d.cpu().numpy()
    if torch.is_tensor(poses_2d):
        poses_2d = poses_2d.cpu().numpy()
    
    data = {
        'num_samples': len(poses_3d),
        'poses': []
    }
    
    for i in range(len(poses_3d)):
        pose_data = {
            'image_path': image_paths[i] if i < len(image_paths) else f'sample_{i}',
            'pose_3d': poses_3d[i].tolist(),
            'pose_2d': poses_2d[i].tolist() if poses_2d is not None else None
        }
        data['poses'].append(pose_data)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_poses_from_json(json_path):
    """
    Load poses from JSON file
    Args:
        json_path: path to JSON file
    Returns:
        poses_3d: array of 3D poses
        poses_2d: array of 2D poses (if available)
        image_paths: list of image paths
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    poses_3d = []
    poses_2d = []
    image_paths = []
    
    for pose_data in data['poses']:
        poses_3d.append(pose_data['pose_3d'])
        if pose_data['pose_2d'] is not None:
            poses_2d.append(pose_data['pose_2d'])
        image_paths.append(pose_data['image_path'])
    
    poses_3d = np.array(poses_3d)
    poses_2d = np.array(poses_2d) if poses_2d else None
    
    return poses_3d, poses_2d, image_paths


# AP10K specific keypoint indices and connections
AP10K_KEYPOINT_NAMES = [
    "forehead", "head", "left_eye", "right_eye", "nose", "left_ear", "right_ear",
    "mouth_front_top", "mouth_front_bottom", "mouth_back_left", "mouth_back_right",
    "neck", "left_shoulder", "right_shoulder", "upper_back", "torso_mid_back",
    "body_center", "lower_back", "left_elbow", "right_elbow", "left_wrist",
    "right_wrist", "left_hand", "right_hand", "left_hip", "right_hip", "center_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_foot", "right_foot",
    "root_tail", "mid_tail", "mid_end_tail", "end_tail"
]

AP10K_SKELETON_CONNECTIONS = [
    # Head connections
    (1, 2), (1, 3), (1, 4), (2, 5), (3, 6),
    # Torso connections  
    (11, 12), (11, 13), (12, 14), (13, 14), (14, 15), (15, 16), (16, 17),
    # Arm connections
    (12, 18), (18, 20), (20, 22), (13, 19), (19, 21), (21, 23),
    # Hip connections
    (16, 24), (16, 25), (24, 26), (25, 26),
    # Leg connections
    (24, 27), (27, 29), (29, 31), (25, 28), (28, 30), (30, 32),
    # Tail connections
    (17, 33), (33, 34), (34, 35), (35, 36)
]


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Create dummy 3D pose
    pose_3d = np.random.randn(37, 3)
    
    # Test centering
    centered_pose = center_3d_pose(pose_3d)
    print(f"Original pose center: {np.mean(pose_3d, axis=0)}")
    print(f"Centered pose center: {np.mean(centered_pose, axis=0)}")
    
    # Test bone length computation
    bone_lengths = compute_bone_lengths(pose_3d, AP10K_SKELETON_CONNECTIONS)
    print(f"Number of bones: {len(bone_lengths)}")
    print(f"Average bone length: {np.mean(bone_lengths):.3f}")
    
    # Test 3D metrics
    gt_pose = pose_3d + np.random.randn(37, 3) * 0.1  # Add noise
    pred_poses = np.expand_dims(pose_3d, 0)
    gt_poses = np.expand_dims(gt_pose, 0)
    
    metrics = compute_3d_metrics(pred_poses, gt_poses)
    print(f"MPJPE: {metrics['mpjpe']:.3f}")
    print(f"PA-MPJPE: {metrics['pa_mpjpe']:.3f}")
    
    print("Utility functions test completed!") 