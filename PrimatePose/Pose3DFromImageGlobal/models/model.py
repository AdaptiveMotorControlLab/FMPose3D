import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class VisualEncoder(nn.Module):
    """
    Visual encoder using ImageNet pretrained ResNet backbone
    """
    def __init__(self, backbone='resnet50', pretrained=True, feature_dim=2048):
        super(VisualEncoder, self).__init__()
        
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add adaptive pooling to ensure consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, 3, H, W) image tensor
        Returns:
            features: (batch_size, feature_dim) visual features
        """
        features = self.backbone(x)  # (batch_size, feature_dim, H, W)
        features = self.adaptive_pool(features)  # (batch_size, feature_dim, 1, 1)
        features = features.flatten(1)  # (batch_size, feature_dim)
        return features


class Pose2DEncoder(nn.Module):
    """
    Encoder for 2D pose keypoints with visual features
    """
    def __init__(self, num_keypoints=37, visual_dim=2048, hidden_dim=256, output_dim=512):
        super(Pose2DEncoder, self).__init__()
        
        self.num_keypoints = num_keypoints
        pose_input_dim = num_keypoints * 2  # x, y coordinates (ignore visibility for now)
        
        # Combined input: 2D pose + visual features
        input_dim = pose_input_dim + visual_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, pose_2d, visual_features, valid_mask=None):
        """
        Args:
            pose_2d: (batch_size, num_keypoints, 3) with [x, y, visibility]
            visual_features: (batch_size, visual_dim) visual features from encoder
            valid_mask: (batch_size, num_keypoints) boolean mask for valid keypoints
        Returns:
            encoded: (batch_size, output_dim) encoded 2D pose features
        """
        batch_size = pose_2d.shape[0]
        
        # Extract x, y coordinates
        xy_coords = pose_2d[:, :, :2]  # (batch_size, num_keypoints, 2)
        
        # Apply valid mask if provided
        if valid_mask is not None:
            # Zero out invalid keypoints
            xy_coords = xy_coords * valid_mask.unsqueeze(-1).float()
        
        # Flatten to (batch_size, num_keypoints * 2)
        xy_flat = xy_coords.view(batch_size, -1)
        
        # Concatenate 2D pose with visual features
        combined_input = torch.cat([xy_flat, visual_features], dim=1)
        
        # Encode
        encoded = self.encoder(combined_input)
        return encoded


class CameraPredictor(nn.Module):
    """
    Predicts camera parameters from image features
    """
    def __init__(self, input_dim=2048, hidden_dim=512):
        super(CameraPredictor, self).__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 9)  # Camera parameters: focal_length, rotation (3x3), translation (3,)
        )
        
    def forward(self, visual_features):
        """
        Args:
            visual_features: (batch_size, feature_dim) visual features from encoder
        Returns:
            camera_params: (batch_size, 9) camera parameters
                - focal_length: (batch_size, 1)
                - rotation: (batch_size, 3) rotation angles (can be converted to 3x3 matrix)
                - translation: (batch_size, 3) translation vector
        """
        camera_params = self.predictor(visual_features)
        
        # Split into components
        focal_length = torch.sigmoid(camera_params[:, 0:1]) * FOCAL_LENGTH_SCALE + FOCAL_LENGTH_OFFSET  # Range [500, 1500]
        rotation = camera_params[:, 1:4]  # Rotation angles
        translation = camera_params[:, 4:7]  # Translation
        scale = torch.sigmoid(camera_params[:, 7:8]) * self.SCALE_FACTOR_RANGE + self.SCALE_FACTOR_OFFSET  # Scale factor [0.5, 2.5]
        center = torch.tanh(camera_params[:, 8:9])  # Center offset [-1, 1]
        
        return {
            'focal_length': focal_length,
            'rotation': rotation,
            'translation': translation,
            'scale': scale,
            'center': center
        }


class Pose3DPredictor(nn.Module):
    """
    Predicts 3D pose from combined image and 2D pose features
    """
    def __init__(self, visual_dim=2048, pose_2d_dim=512, num_keypoints=37, hidden_dim=1024):
        super(Pose3DPredictor, self).__init__()
        
        self.num_keypoints = num_keypoints
        input_dim = visual_dim + pose_2d_dim
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_keypoints * 3)  # x, y, z for each keypoint
        )
        
    def forward(self, visual_features, pose_2d_features):
        """
        Args:
            visual_features: (batch_size, visual_dim) features from visual encoder
            pose_2d_features: (batch_size, pose_2d_dim) features from 2D pose encoder
        Returns:
            pose_3d: (batch_size, num_keypoints, 3) predicted 3D pose
        """
        # Concatenate features
        combined_features = torch.cat([visual_features, pose_2d_features], dim=1)
        
        # Predict 3D pose
        pose_3d_flat = self.predictor(combined_features)
        
        # Reshape to (batch_size, num_keypoints, 3)
        pose_3d = pose_3d_flat.view(-1, self.num_keypoints, 3)
        
        return pose_3d


class Pose3DEstimator(nn.Module):
    """
    Complete 3D pose estimation model
    """
    def __init__(self, num_keypoints=37, backbone='resnet50', pretrained=True):
        super(Pose3DEstimator, self).__init__()
        
        self.num_keypoints = num_keypoints
        
        # Components
        self.visual_encoder = VisualEncoder(backbone=backbone, pretrained=pretrained)
        self.pose_2d_encoder = Pose2DEncoder(
            num_keypoints=num_keypoints,
            visual_dim=self.visual_encoder.feature_dim
        )
        self.camera_predictor = CameraPredictor(input_dim=self.visual_encoder.feature_dim)
        self.pose_3d_predictor = Pose3DPredictor(
            visual_dim=self.visual_encoder.feature_dim,
            pose_2d_dim=512,
            num_keypoints=num_keypoints
        )
        
    def forward(self, image, pose_2d, valid_mask=None):
        """
        Args:
            image: (batch_size, 3, H, W) input images
            pose_2d: (batch_size, num_keypoints, 3) 2D pose with [x, y, visibility]
            valid_mask: (batch_size, num_keypoints) boolean mask for valid keypoints
        Returns:
            dict with:
                - pose_3d: (batch_size, num_keypoints, 3) predicted 3D pose
                - camera_params: dict with camera parameters
        """
        # Extract visual features
        visual_features = self.visual_encoder(image)
        
        # Extract 2D pose features (now includes visual features)
        pose_2d_features = self.pose_2d_encoder(pose_2d, visual_features, valid_mask)
        
        # Predict camera parameters
        camera_params = self.camera_predictor(visual_features)
        
        # Predict 3D pose
        pose_3d = self.pose_3d_predictor(visual_features, pose_2d_features)
        
        return {
            'pose_3d': pose_3d,
            'camera_params': camera_params
        }


def rotation_matrix_from_angles(angles):
    """
    Convert rotation angles to rotation matrix
    Args:
        angles: (batch_size, 3) rotation angles [rx, ry, rz]
    Returns:
        rotation_matrix: (batch_size, 3, 3) rotation matrices
    """
    batch_size = angles.shape[0]
    device = angles.device
    
    rx, ry, rz = angles[:, 0], angles[:, 1], angles[:, 2]
    
    # Create rotation matrices for each axis
    zeros = torch.zeros(batch_size, device=device)
    ones = torch.ones(batch_size, device=device)
    
    # Rotation around x-axis
    Rx = torch.stack([
        torch.stack([ones, zeros, zeros], dim=1),
        torch.stack([zeros, torch.cos(rx), -torch.sin(rx)], dim=1),
        torch.stack([zeros, torch.sin(rx), torch.cos(rx)], dim=1)
    ], dim=1)
    
    # Rotation around y-axis
    Ry = torch.stack([
        torch.stack([torch.cos(ry), zeros, torch.sin(ry)], dim=1),
        torch.stack([zeros, ones, zeros], dim=1),
        torch.stack([-torch.sin(ry), zeros, torch.cos(ry)], dim=1)
    ], dim=1)
    
    # Rotation around z-axis
    Rz = torch.stack([
        torch.stack([torch.cos(rz), -torch.sin(rz), zeros], dim=1),
        torch.stack([torch.sin(rz), torch.cos(rz), zeros], dim=1),
        torch.stack([zeros, zeros, ones], dim=1)
    ], dim=1)
    
    # Combined rotation: R = Rz * Ry * Rx
    R = torch.bmm(torch.bmm(Rz, Ry), Rx)
    
    return R


def project_3d_to_2d(pose_3d, camera_params, image_size=(256, 256)):
    """
    Project 3D pose to 2D using camera parameters
    Args:
        pose_3d: (batch_size, num_keypoints, 3) 3D pose
        camera_params: dict with camera parameters
        image_size: tuple (width, height) of image
    Returns:
        pose_2d_proj: (batch_size, num_keypoints, 2) projected 2D pose
    """
    batch_size, num_keypoints, _ = pose_3d.shape
    device = pose_3d.device
    
    # Extract camera parameters
    focal_length = camera_params['focal_length']  # (batch_size, 1)
    rotation_angles = camera_params['rotation']  # (batch_size, 3)
    translation = camera_params['translation']  # (batch_size, 3)
    scale = camera_params['scale']  # (batch_size, 1)
    
    # Get rotation matrix
    R = rotation_matrix_from_angles(rotation_angles)  # (batch_size, 3, 3)
    
    # Apply rotation and translation
    pose_3d_transformed = torch.bmm(pose_3d, R.transpose(1, 2))  # (batch_size, num_keypoints, 3)
    pose_3d_transformed = pose_3d_transformed + translation.unsqueeze(1)  # Add translation
    
    # Project to 2D using perspective projection
    x_3d = pose_3d_transformed[:, :, 0]  # (batch_size, num_keypoints)
    y_3d = pose_3d_transformed[:, :, 1]  # (batch_size, num_keypoints)
    z_3d = pose_3d_transformed[:, :, 2]  # (batch_size, num_keypoints)
    
    # Avoid division by zero
    z_3d = torch.clamp(z_3d, min=0.1)
    
    # Perspective projection
    x_2d = (focal_length * x_3d / z_3d) * scale
    y_2d = (focal_length * y_3d / z_3d) * scale
    
    # Convert to image coordinates (center at image center)
    x_2d = x_2d + image_size[0] / 2
    y_2d = y_2d + image_size[1] / 2
    
    # Stack to get 2D pose
    pose_2d_proj = torch.stack([x_2d, y_2d], dim=2)  # (batch_size, num_keypoints, 2)
    
    return pose_2d_proj


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Pose3DEstimator(num_keypoints=37, backbone='resnet50')
    model.to(device)
    
    # Create dummy input
    batch_size = 4
    image = torch.randn(batch_size, 3, 256, 256).to(device)
    pose_2d = torch.randn(batch_size, 37, 3).to(device)
    valid_mask = torch.ones(batch_size, 37, dtype=torch.bool).to(device)
    
    # Forward pass
    output = model(image, pose_2d, valid_mask)
    
    print(f"Input image shape: {image.shape}")
    print(f"Input pose 2D shape: {pose_2d.shape}")
    print(f"Output pose 3D shape: {output['pose_3d'].shape}")
    print(f"Camera params: {list(output['camera_params'].keys())}")
    
    # Test projection
    pose_2d_proj = project_3d_to_2d(output['pose_3d'], output['camera_params'])
    print(f"Projected 2D pose shape: {pose_2d_proj.shape}")
    
    print("Model test completed successfully!") 