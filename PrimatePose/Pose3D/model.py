import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class Pose3DEstimator(nn.Module):
    """
    3D pose estimation model from 2D pose only
    Architecture: MLP -> Multi-head Attention (3 rounds) -> MLP -> 3D pose
    """
    def __init__(self, num_keypoints=37, embed_dim=256, num_heads=8, root_joint_idx=11):
        super(Pose3DEstimator, self).__init__()
        
        self.num_keypoints = num_keypoints
        self.embed_dim = embed_dim
        self.root_joint_idx = root_joint_idx
        
        # Input embedding: 2D pose -> high-dimensional space
        self.input_embedding = nn.Sequential(
            nn.Linear(2, embed_dim // 4),  # x, y -> embed_dim/4
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 4, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Positional encoding for keypoints
        self.pos_encoding = nn.Parameter(torch.randn(1, num_keypoints, embed_dim))
        
        # Multi-head attention blocks (3 rounds)
        self.attention_blocks = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
            for _ in range(3)
        ])
        
        # Layer normalization for each attention block
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(3)
        ])
        
        # Feed-forward networks for each attention block
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, embed_dim)
            ) for _ in range(3)
        ])
        
        # Output MLP for 3D pose prediction
        # Predict: root_depth + relative_coordinates for other joints
        output_dim = 1 + (num_keypoints - 1) * 3  # root_depth + (N-1)*3 relative coords
        self.output_mlp = nn.Sequential(
            nn.Linear(embed_dim * num_keypoints, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, output_dim)
        )
        
        # Camera parameter predictor (from global feature)
        self.camera_predictor = nn.Sequential(
            nn.Linear(embed_dim * num_keypoints, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 9)  # focal_length + rotation(3) + translation(3) + scale + center
        )
        
    def forward(self, pose_2d, valid_mask=None):
        """
        Args:
            pose_2d: (batch_size, num_keypoints, 3) 2D pose with [x, y, visibility]
            valid_mask: (batch_size, num_keypoints) boolean mask for valid keypoints
        Returns:
            dict with:
                - pose_3d: (batch_size, num_keypoints, 3) predicted 3D pose
                - camera_params: dict with camera parameters
        """
        batch_size = pose_2d.shape[0]
        
        # Extract 2D coordinates (ignore visibility for now)
        pose_2d_xy = pose_2d[:, :, :2]  # (batch_size, num_keypoints, 2)
        
        # Embed 2D coordinates to high-dimensional space
        embedded = self.input_embedding(pose_2d_xy)  # (batch_size, num_keypoints, embed_dim)
        
        # Add positional encoding
        embedded = embedded + self.pos_encoding
        
        # Apply attention blocks (3 rounds)
        x = embedded
        for i in range(3):
            # Self-attention
            attn_out, _ = self.attention_blocks[i](x, x, x)
            x = self.layer_norms[i](x + attn_out)  # Residual connection
            
            # Feed-forward
            ffn_out = self.ffns[i](x)
            x = x + ffn_out  # Residual connection
        
        # Global feature for camera parameters
        global_feature = x.view(batch_size, -1)  # (batch_size, embed_dim * num_keypoints)
        
        # Predict camera parameters
        camera_params_raw = self.camera_predictor(global_feature)
        
        # Split into components
        focal_length = torch.sigmoid(camera_params_raw[:, 0:1]) * 1000 + 500  # Range [500, 1500]
        rotation = camera_params_raw[:, 1:4]  # Rotation angles [rx, ry, rz]
        translation = camera_params_raw[:, 4:7]  # Translation vector [tx, ty, tz]
        scale = torch.sigmoid(camera_params_raw[:, 7:8]) * 2 + 0.5  # Scale factor [0.5, 2.5]
        center = torch.tanh(camera_params_raw[:, 8:9])  # Center offset [-1, 1]
        
        camera_params = {
            'focal_length': focal_length,
            'rotation': rotation,
            'translation': translation,
            'scale': scale,
            'center': center
        }
        
        # Predict 3D pose
        pose_3d_prediction = self.output_mlp(global_feature)
        
        # Split prediction: root_depth + relative_coordinates
        root_depth = pose_3d_prediction[:, 0:1]  # (batch_size, 1)
        relative_coords = pose_3d_prediction[:, 1:].view(batch_size, self.num_keypoints - 1, 3)
        
        # Reconstruct absolute 3D pose
        pose_3d = self._reconstruct_absolute_pose(pose_2d_xy, root_depth, relative_coords)
        
        return {
            'pose_3d': pose_3d,
            'camera_params': camera_params
        }
    
    def _reconstruct_absolute_pose(self, pose_2d_xy, root_depth, relative_coords):
        """
        Reconstruct absolute 3D pose from root depth and relative coordinates
        """
        batch_size = pose_2d_xy.shape[0]
        device = pose_2d_xy.device
        
        # Initialize 3D pose tensor
        pose_3d = torch.zeros(batch_size, self.num_keypoints, 3, device=device)
        
        # Set root joint (neck) position
        root_2d = pose_2d_xy[:, self.root_joint_idx, :]  # (batch_size, 2)
        pose_3d[:, self.root_joint_idx, :2] = root_2d  # x, y from 2D
        pose_3d[:, self.root_joint_idx, 2:3] = root_depth  # z from prediction
        
        # Set other joints as relative to root
        rel_idx = 0
        for i in range(self.num_keypoints):
            if i != self.root_joint_idx:
                # Add relative coordinates to root position
                pose_3d[:, i, :] = pose_3d[:, self.root_joint_idx, :] + relative_coords[:, rel_idx, :]
                rel_idx += 1
        
        return pose_3d


def project_3d_to_2d(pose_3d, camera_params, image_size=(256, 256)):
    """
    Project 3D pose to 2D using simplified camera parameters
    Assumes 3D pose is already in camera coordinate system
    Args:
        pose_3d: (batch_size, num_keypoints, 3) 3D pose in camera coordinates
        camera_params: dict with camera parameters (focal_length, scale)
        image_size: tuple (width, height) of image
    Returns:
        pose_2d_proj: (batch_size, num_keypoints, 2) projected 2D pose
    """
    batch_size, num_keypoints, _ = pose_3d.shape
    device = pose_3d.device
    
    # Extract camera parameters
    focal_length = camera_params['focal_length']  # (batch_size, 1)
    scale = camera_params['scale']  # (batch_size, 1)
    
    # Extract 3D coordinates (already in camera coordinate system)
    x_3d = pose_3d[:, :, 0]  # (batch_size, num_keypoints)
    y_3d = pose_3d[:, :, 1]  # (batch_size, num_keypoints)
    z_3d = pose_3d[:, :, 2]  # (batch_size, num_keypoints)
    
    # Avoid division by zero
    z_3d = torch.clamp(z_3d, min=0.1)
    
    # Simple perspective projection
    x_2d = (focal_length * x_3d / z_3d) * scale
    y_2d = (focal_length * y_3d / z_3d) * scale
    
    # Convert to image coordinates (center at image center)
    x_2d = x_2d + image_size[0] / 2
    y_2d = y_2d + image_size[1] / 2
    
    # Stack to get 2D pose
    pose_2d_proj = torch.stack([x_2d, y_2d], dim=2)  # (batch_size, num_keypoints, 2)
    
    return pose_2d_proj


def convert_to_relative_pose(pose_3d, root_joint_idx=11):
    """
    Convert absolute 3D pose to relative coordinates with respect to root joint
    Args:
        pose_3d: (batch_size, num_keypoints, 3) absolute 3D pose
        root_joint_idx: index of root joint (default: 11 for neck)
    Returns:
        relative_pose: (batch_size, num_keypoints, 3) relative 3D pose
        root_position: (batch_size, 3) root joint position
    """
    root_position = pose_3d[:, root_joint_idx:root_joint_idx+1, :]  # (batch_size, 1, 3)
    relative_pose = pose_3d - root_position  # Broadcast subtraction
    
    return relative_pose, root_position.squeeze(1)


def convert_to_absolute_pose(relative_pose, root_position, root_joint_idx=11):
    """
    Convert relative 3D pose to absolute coordinates
    Args:
        relative_pose: (batch_size, num_keypoints, 3) relative 3D pose
        root_position: (batch_size, 3) root joint position
        root_joint_idx: index of root joint (default: 11 for neck)
    Returns:
        absolute_pose: (batch_size, num_keypoints, 3) absolute 3D pose
    """
    root_position = root_position.unsqueeze(1)  # (batch_size, 1, 3)
    absolute_pose = relative_pose + root_position  # Broadcast addition
    
    return absolute_pose


if __name__ == "__main__":
    # Test the model
    model = Pose3DEstimator(num_keypoints=37, embed_dim=256, num_heads=8)
    
    # Dummy input
    batch_size = 4
    pose_2d = torch.randn(batch_size, 37, 3)
    
    # Forward pass
    predictions = model(pose_2d)
    
    print("Model output:")
    print(f"  pose_3d: {predictions['pose_3d'].shape}")
    print(f"  camera_params: {list(predictions['camera_params'].keys())}")
    
    # Test projection
    pose_2d_proj = project_3d_to_2d(predictions['pose_3d'], predictions['camera_params'])
    print(f"  projected_2d: {pose_2d_proj.shape}")
    
    print("Model test passed!") 