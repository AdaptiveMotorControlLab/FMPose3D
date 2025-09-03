import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
try:
    import timm
except ImportError:
    timm = None
import numpy as np

class VisualEncoder(nn.Module):
    """
    Visual encoder supporting ResNet backbones (torchvision) and ViT DINO (timm).
    - ResNet outputs are pooled to a global feature vector.
    - ViT DINO (ViT-S/16) uses timm with pretrained weights and returns a global feature vector.
    """
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        feature_dim: int | None = None,
        image_size: tuple[int, int] | None = None,
    ):
        super(VisualEncoder, self).__init__()

        self.backbone_name = backbone
        self.adaptive_pool = None

        # Torchvision ResNet family
        if backbone == 'resnet50':
            base = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
            self.encoder = nn.Sequential(*list(base.children())[:-1])
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif backbone == 'resnet34':
            base = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
            self.encoder = nn.Sequential(*list(base.children())[:-1])
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        elif backbone == 'resnet18':
            base = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
            self.encoder = nn.Sequential(*list(base.children())[:-1])
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ViT Small/16 DINO (via timm)
        elif backbone in ['vit_s16_dino', 'dino_vits16', 'vit_small_patch16_224.dino']:
            model_name = 'vit_small_patch16_224.dino'
            # Determine target image size as (H, W). Defaults to 224 if not provided
            if image_size is not None and len(image_size) == 2:
                # Upstream expects (H, W); args use [W, H]
                target_h, target_w = image_size[1], image_size[0]
                vit_img_size = (target_h, target_w)
            else:
                vit_img_size = (224, 224)
            # num_classes=0 makes forward() return the pooled features directly
            self.encoder = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                img_size=vit_img_size,
            )
            # Prefer num_features if available; fallback to embed_dim
            self.feature_dim = getattr(self.encoder, 'num_features', None) or getattr(self.encoder, 'embed_dim', 384)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 3, H, W) image tensor
        Returns:
            features: (batch_size, feature_dim) visual features
        """
        features = self.encoder(x)
        # If encoder returns a 4D CNN map, pool & flatten. If it's already (B, C), return as-is.
        if isinstance(features, torch.Tensor) and features.ndim == 4:
            if self.adaptive_pool is None:
                self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
            features = self.adaptive_pool(features)
            features = features.flatten(1)
        elif isinstance(features, torch.Tensor) and features.ndim == 3:
            # Some models might return (B, N, C); take [CLS] / mean tokens
            # Default to mean pooling across tokens
            features = features.mean(dim=1)
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


class Pose3DPredictor(nn.Module):
    """
    Graph-based 3D pose predictor in camera coordinates, following Pose3D/model.py design.
    Builds per-joint embeddings from 2D keypoints and global visual/pose features,
    applies optional graph convolution and a lightweight attention block, then predicts 3D.
    """
    def __init__(
        self,
        visual_dim: int = 2048,
        pose_2d_dim: int = 512,
        num_keypoints: int = 37,
        embed_dim: int = 256,
        num_heads: int = 8,
        use_graph_encoder: bool = True,
        graph_layers: int = 3,
        graph_dropout: float = 0.1,
    ) -> None:
        super(Pose3DPredictor, self).__init__()

        self.num_keypoints = num_keypoints
        self.embed_dim = embed_dim
        self.use_graph_encoder = use_graph_encoder

        # Project global visual features and global pose2d features to embed_dim and broadcast to all joints
        self.visual_proj = nn.Linear(visual_dim, embed_dim)
        self.pose2d_global_proj = nn.Linear(pose_2d_dim, embed_dim)

        # Positional encoding (joint index)
        self.pos_encoding = nn.Parameter(torch.randn(1, num_keypoints, embed_dim))

        # Optional graph encoder
        if self.use_graph_encoder:
            self.graph_encoder = GraphPoseEncoder(
                num_keypoints=num_keypoints,
                feature_dim=embed_dim,
                num_layers=graph_layers,
                dropout=graph_dropout,
            )

        # One round of attention + FFN as in Pose3D/model.py
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # Output MLP to 3D for each joint
        self.output_mlp = nn.Sequential(
            nn.Linear(embed_dim * num_keypoints, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embed_dim * 2, num_keypoints * 3),
        )

    def forward(self, visual_features: torch.Tensor, pose_2d_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (B, visual_dim)
            pose_2d_features: (B, pose_2d_dim) global 2D pose feature (from Pose2DEncoder)
        Returns:
            pose_3d: (B, N, 3)
        """
        B = visual_features.shape[0]
        # Broadcast global features per joint and add positional encoding
        v = self.visual_proj(visual_features).unsqueeze(1).expand(B, self.num_keypoints, -1)
        p = self.pose2d_global_proj(pose_2d_features).unsqueeze(1).expand(B, self.num_keypoints, -1)
        x = v + p + self.pos_encoding

        # Add positional encoding
        # (already included above)

        # Optional graph encoder
        if self.use_graph_encoder:
            x = self.graph_encoder(x)

        # Attention block
        attn_out, _ = self.attn(x, x, x)
        x = self.layer_norm(x + attn_out)
        x = x + self.ffn(x)

        # Output 3D
        global_feature = x.reshape(B, -1)
        pose_3d_flat = self.output_mlp(global_feature)
        pose_3d = pose_3d_flat.view(B, self.num_keypoints, 3)
        return pose_3d

class Pose3D(nn.Module):
    """
    Complete 3D pose estimation model in camera coordinate system
    Predicts 3D pose directly in camera coordinates and camera intrinsic parameters
    """
    def __init__(self, num_keypoints=37, backbone='resnet50', pretrained=True, image_size: tuple[int, int] = (256, 256)):
        super(Pose3D, self).__init__()
        
        self.num_keypoints = num_keypoints
        
        # Components
        self.visual_encoder = VisualEncoder(backbone=backbone, pretrained=pretrained, image_size=image_size)
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
    
    # Minimal smoke test for Pose3D forward
    model = Pose3D(num_keypoints=37, backbone='resnet50')
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