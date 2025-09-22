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


class GraphConvLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0, use_residual: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual and (input_dim == output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, node_features: torch.Tensor, normalized_adjacency: torch.Tensor) -> torch.Tensor:
        aggregated = torch.matmul(normalized_adjacency, node_features)
        transformed = self.linear(aggregated)
        transformed = self.activation(transformed)
        transformed = self.dropout(transformed)
        if self.use_residual:
            transformed = transformed + node_features
        transformed = self.layer_norm(transformed)
        return transformed


class GraphPoseEncoder(nn.Module):
    def __init__(
        self,
        num_keypoints: int,
        feature_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        connections: list[tuple[int, int]] | None = None,
    ):
        super().__init__()
        if connections is None:
            connections = self._get_default_connections()
        adjacency = self._build_adjacency(num_keypoints, connections)
        normalized = self._normalize_adjacency(adjacency)
        self.register_buffer("normalized_adjacency", normalized)
        self.layers = nn.ModuleList([
            GraphConvLayer(feature_dim, feature_dim, dropout=dropout, use_residual=True)
            for _ in range(num_layers)
        ])

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        output = node_features
        for layer in self.layers:
            output = layer(output, self.normalized_adjacency)
        return output

    @staticmethod
    def _build_adjacency(num_keypoints: int, connections: list[tuple[int, int]]) -> torch.Tensor:
        adjacency = torch.zeros(num_keypoints, num_keypoints)
        for i, j in connections:
            if 0 <= i < num_keypoints and 0 <= j < num_keypoints:
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0
        return adjacency

    @staticmethod
    def _normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
        adjacency_with_self = adjacency + torch.eye(adjacency.shape[0])
        degree = adjacency_with_self.sum(dim=1)
        degree_inv_sqrt = torch.pow(torch.clamp(degree, min=1e-6), -0.5)
        degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        normalized = degree_inv_sqrt @ adjacency_with_self @ degree_inv_sqrt
        return normalized

    @staticmethod
    def _get_default_connections() -> list:
        return [
            (1, 11), (11, 12), (11, 13), (12, 22), (13, 23),
            (11, 26), (26, 27), (26, 28), (27, 31), (28, 32),
            (26, 36),
        ]

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
    Predicts camera intrinsic parameters for camera coordinate system
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
            nn.Linear(hidden_dim // 2, 4)  # Camera intrinsics: fx, fy, cx, cy
        )
        
    def forward(self, visual_features):
        """
        Args:
            visual_features: (batch_size, feature_dim) visual features from encoder
        Returns:
            camera_params: dict with camera intrinsic parameters
                - focal_length_x: (batch_size, 1) focal length in x direction
                - focal_length_y: (batch_size, 1) focal length in y direction  
                - principal_point_x: (batch_size, 1) principal point x coordinate
                - principal_point_y: (batch_size, 1) principal point y coordinate
        """
        camera_params = self.predictor(visual_features)
        
        # Split into components
        fx = torch.sigmoid(camera_params[:, 0:1]) * 1000 + 500  # Range [500, 1500]
        fy = torch.sigmoid(camera_params[:, 1:2]) * 1000 + 500  # Range [500, 1500]
        cx = torch.sigmoid(camera_params[:, 2:3]) * 256  # Range [0, 256] for 256x256 image
        cy = torch.sigmoid(camera_params[:, 3:4]) * 256  # Range [0, 256] for 256x256 image
        
        return {
            'focal_length_x': fx,
            'focal_length_y': fy,
            'principal_point_x': cx,
            'principal_point_y': cy
        }


class Pose3D(nn.Module):
    """
    Predicts 3D pose in camera coordinate system from combined image and 2D pose features
    """
    def __init__(self, visual_dim=2048, pose_2d_dim=512, num_keypoints=37, hidden_dim=1024):
        super(Pose3D, self).__init__()
        
        self.num_keypoints = num_keypoints
        input_dim = visual_dim + pose_2d_dim  # Use both visual and 2D pose features for camera space
        
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
            nn.Linear(hidden_dim // 2, num_keypoints * 3)  # x, y, z for each keypoint in camera coordinates
        )
        
    def forward(self, visual_features, pose_2d_features):
        """
        Args:
            visual_features: (batch_size, visual_dim) features from visual encoder
            pose_2d_features: (batch_size, pose_2d_dim) features from 2D pose encoder
        Returns:
            pose_3d: (batch_size, num_keypoints, 3) predicted 3D pose in camera coordinate system
        """
        # Concatenate features
        combined_features = torch.cat([visual_features, pose_2d_features], dim=1)
        
        # Predict 3D pose in camera coordinate system
        pose_3d_flat = self.predictor(combined_features)
        
        # Reshape to (batch_size, num_keypoints, 3)
        pose_3d = pose_3d_flat.view(-1, self.num_keypoints, 3)
        
        return pose_3d


class Pose3DEstimator(nn.Module):
    """
    Complete 3D pose estimation model in camera coordinate system
    Predicts 3D pose directly in camera coordinates and camera intrinsic parameters
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
        self.pose_3d_predictor = Pose3D(
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
                - pose_3d: (batch_size, num_keypoints, 3) predicted 3D pose in camera coordinate system
                - camera_params: dict with camera intrinsic parameters (fx, fy, cx, cy)
        """
        # Extract visual features
        visual_features = self.visual_encoder(image)
        
        # Extract 2D pose features (now includes visual features)
        pose_2d_features = self.pose_2d_encoder(pose_2d, visual_features, valid_mask)
        
        # Predict camera intrinsic parameters
        camera_params = self.camera_predictor(visual_features)
        
        # Predict 3D pose in camera coordinate system
        pose_3d = self.pose_3d_predictor(visual_features, pose_2d_features)
        
        return {
            'pose_3d': pose_3d,
            'camera_params': camera_params
        }


def project_3d_to_2d(pose_3d, camera_params, image_size=(256, 256)):
    """
    Project 3D pose to 2D using camera intrinsic parameters
    Args:
        pose_3d: (batch_size, num_keypoints, 3) 3D pose in camera coordinate system
        camera_params: dict with camera intrinsic parameters
        image_size: tuple (width, height) of image
    Returns:
        pose_2d_proj: (batch_size, num_keypoints, 2) projected 2D pose
    """
    batch_size, num_keypoints, _ = pose_3d.shape
    device = pose_3d.device
    width, height = image_size
    
    # Extract camera intrinsic parameters
    fx = camera_params['focal_length_x']  # (batch_size, 1)
    fy = camera_params['focal_length_y']  # (batch_size, 1)
    cx = camera_params['principal_point_x']  # (batch_size, 1)
    cy = camera_params['principal_point_y']  # (batch_size, 1)
    
    # Extract 3D coordinates (already in camera coordinate system)
    x_3d = pose_3d[:, :, 0]  # (batch_size, num_keypoints)
    y_3d = pose_3d[:, :, 1]  # (batch_size, num_keypoints)
    z_3d = pose_3d[:, :, 2]  # (batch_size, num_keypoints)
    
    # CRITICAL: Avoid division by zero, ensure positive depth
    z_3d = torch.clamp(z_3d, min=0.1)  # Minimum depth of 0.1 units
    
    # Perspective projection using camera intrinsics
    # x_2d = fx * X/Z + cx
    # y_2d = fy * Y/Z + cy
    x_2d = (fx * x_3d / z_3d) + cx  # (batch_size, num_keypoints)
    y_2d = (fy * y_3d / z_3d) + cy  # (batch_size, num_keypoints)
    
    # CRITICAL: Clamp projected coordinates to stay within image bounds
    margin = 5.0
    x_2d = torch.clamp(x_2d, margin, width - margin)
    y_2d = torch.clamp(y_2d, margin, height - margin)
    
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
    print(f"Output pose 3D shape (camera coords): {output['pose_3d'].shape}")
    print(f"Camera intrinsic params: {list(output['camera_params'].keys())}")
    
    # Test projection
    pose_2d_proj = project_3d_to_2d(output['pose_3d'], output['camera_params'])
    print(f"Projected 2D pose shape: {pose_2d_proj.shape}")
    
    print("Camera coordinate system model test completed successfully!") 