import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import project_3d_to_2d


class ReprojectionLoss(nn.Module):
    """
    Reprojection loss: project predicted 3D pose to 2D and compare with GT 2D pose
    """
    def __init__(self, loss_type='mse', image_size=(256, 256)):
        super(ReprojectionLoss, self).__init__()
        self.loss_type = loss_type
        self.image_size = image_size
        
        if loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        elif loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, pose_3d_pred, camera_params_pred, pose_2d_gt, valid_mask=None):
        """
        Args:
            pose_3d_pred: (batch_size, num_keypoints, 3) predicted 3D pose
            camera_params_pred: dict with predicted camera parameters
            pose_2d_gt: (batch_size, num_keypoints, 3) ground truth 2D pose [x, y, visibility]
            valid_mask: (batch_size, num_keypoints) boolean mask for valid keypoints
        Returns:
            loss: scalar reprojection loss
        """
        # Project 3D pose to 2D
        pose_2d_proj = project_3d_to_2d(pose_3d_pred, camera_params_pred, self.image_size)
        
        # Extract GT 2D coordinates
        pose_2d_gt_xy = pose_2d_gt[:, :, :2]  # (batch_size, num_keypoints, 2)
        
        # Compute loss
        loss = self.criterion(pose_2d_proj, pose_2d_gt_xy)  # (batch_size, num_keypoints, 2)
        
        # Apply valid mask if provided
        if valid_mask is not None:
            loss = loss * valid_mask.unsqueeze(-1).float()  # Zero out invalid keypoints
            
        # Average over valid keypoints and spatial dimensions
        if valid_mask is not None:
            # Only average over valid keypoints
            num_valid = valid_mask.sum(dim=1, keepdim=True).float()  # (batch_size, 1)
            num_valid = torch.clamp(num_valid, min=1.0)  # Avoid division by zero
            loss = loss.sum(dim=[1, 2]) / (num_valid * 2)  # Average per valid keypoint and x,y
            loss = loss.mean()  # Average over batch
        else:
            loss = loss.mean()
            
        return loss


class Pose3DLoss(nn.Module):
    """
    3D pose loss for supervision when 3D GT is available
    """
    def __init__(self, loss_type='mse'):
        super(Pose3DLoss, self).__init__()
        
        if loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        elif loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, pose_3d_pred, pose_3d_gt, valid_mask=None):
        """
        Args:
            pose_3d_pred: (batch_size, num_keypoints, 3) predicted 3D pose
            pose_3d_gt: (batch_size, num_keypoints, 3) ground truth 3D pose
            valid_mask: (batch_size, num_keypoints) boolean mask for valid keypoints
        Returns:
            loss: scalar 3D pose loss
        """
        loss = self.criterion(pose_3d_pred, pose_3d_gt)  # (batch_size, num_keypoints, 3)
        
        # Apply valid mask if provided
        if valid_mask is not None:
            loss = loss * valid_mask.unsqueeze(-1).float()
            
        # Average over valid keypoints and spatial dimensions
        if valid_mask is not None:
            num_valid = valid_mask.sum(dim=1, keepdim=True).float()
            num_valid = torch.clamp(num_valid, min=1.0)
            loss = loss.sum(dim=[1, 2]) / (num_valid * 3)  # Average per valid keypoint and x,y,z
            loss = loss.mean()
        else:
            loss = loss.mean()
            
        return loss


class BoneLengthLoss(nn.Module):
    """
    Bone length consistency loss to enforce realistic pose structure
    """
    def __init__(self, bone_connections=None):
        super(BoneLengthLoss, self).__init__()
        
        # Define bone connections for primates (based on ap10k keypoints)
        if bone_connections is None:
            # Define which keypoints are connected (bone pairs)
            # Based on the ap10k keypoint structure
            self.bone_connections = [
                # Head connections
                (1, 2),   # head to left_eye
                (1, 3),   # head to right_eye
                (1, 4),   # head to nose
                (2, 5),   # left_eye to left_ear
                (3, 6),   # right_eye to right_ear
                
                # Torso connections
                (11, 12), # neck to left_shoulder
                (11, 13), # neck to right_shoulder
                (12, 14), # left_shoulder to upper_back
                (13, 14), # right_shoulder to upper_back
                (14, 15), # upper_back to torso_mid_back
                (15, 16), # torso_mid_back to body_center
                (16, 17), # body_center to lower_back
                
                # Arm connections
                (12, 18), # left_shoulder to left_elbow
                (18, 20), # left_elbow to left_wrist
                (20, 22), # left_wrist to left_hand
                (13, 19), # right_shoulder to right_elbow
                (19, 21), # right_elbow to right_wrist
                (21, 23), # right_wrist to right_hand
                
                # Hip connections
                (16, 24), # body_center to left_hip
                (16, 25), # body_center to right_hip
                (24, 26), # left_hip to center_hip
                (25, 26), # right_hip to center_hip
                
                # Leg connections
                (24, 27), # left_hip to left_knee
                (27, 29), # left_knee to left_ankle
                (29, 31), # left_ankle to left_foot
                (25, 28), # right_hip to right_knee
                (28, 30), # right_knee to right_ankle
                (30, 32), # right_ankle to right_foot
                
                # Tail connections
                (17, 33), # lower_back to root_tail
                (33, 34), # root_tail to mid_tail
                (34, 35), # mid_tail to mid_end_tail
                (35, 36), # mid_end_tail to end_tail
            ]
        else:
            self.bone_connections = bone_connections
        
        self.criterion = nn.MSELoss()
    
    def forward(self, pose_3d_pred, pose_3d_ref=None, valid_mask=None):
        """
        Args:
            pose_3d_pred: (batch_size, num_keypoints, 3) predicted 3D pose
            pose_3d_ref: (batch_size, num_keypoints, 3) reference 3D pose for bone lengths
                        If None, uses average bone lengths from predicted poses
            valid_mask: (batch_size, num_keypoints) boolean mask for valid keypoints
        Returns:
            loss: scalar bone length consistency loss
        """
        batch_size = pose_3d_pred.shape[0]
        device = pose_3d_pred.device
        
        # Compute bone lengths for predicted poses
        pred_bone_lengths = []
        ref_bone_lengths = []
        
        for bone_idx, (joint1, joint2) in enumerate(self.bone_connections):
            # Check if both joints are valid
            if valid_mask is not None:
                valid_bone = valid_mask[:, joint1] & valid_mask[:, joint2]  # (batch_size,)
            else:
                valid_bone = torch.ones(batch_size, dtype=torch.bool, device=device)
            
            if valid_bone.sum() == 0:
                continue
                
            # Compute bone vectors
            pred_bone_vec = pose_3d_pred[:, joint2] - pose_3d_pred[:, joint1]  # (batch_size, 3)
            pred_bone_len = torch.norm(pred_bone_vec, dim=1)  # (batch_size,)
            
            if pose_3d_ref is not None:
                ref_bone_vec = pose_3d_ref[:, joint2] - pose_3d_ref[:, joint1]
                ref_bone_len = torch.norm(ref_bone_vec, dim=1)
            else:
                # Use mean bone length as reference
                ref_bone_len = pred_bone_len.mean().expand_as(pred_bone_len)
            
            # Only consider valid bones
            pred_bone_lengths.append(pred_bone_len[valid_bone])
            ref_bone_lengths.append(ref_bone_len[valid_bone])
        
        if len(pred_bone_lengths) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Concatenate all bone lengths
        pred_lengths = torch.cat(pred_bone_lengths)
        ref_lengths = torch.cat(ref_bone_lengths)
        
        # Compute MSE loss
        loss = self.criterion(pred_lengths, ref_lengths)
        
        return loss


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss for video sequences
    """
    def __init__(self, loss_type='mse'):
        super(TemporalConsistencyLoss, self).__init__()
        
        if loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        elif loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, pose_3d_sequence, valid_mask=None):
        """
        Args:
            pose_3d_sequence: (batch_size, seq_len, num_keypoints, 3) 3D pose sequence
            valid_mask: (batch_size, seq_len, num_keypoints) boolean mask for valid keypoints
        Returns:
            loss: scalar temporal consistency loss
        """
        if pose_3d_sequence.shape[1] < 2:
            # Need at least 2 frames for temporal consistency
            return torch.tensor(0.0, device=pose_3d_sequence.device, requires_grad=True)
        
        # Compute differences between consecutive frames
        pose_diff = pose_3d_sequence[:, 1:] - pose_3d_sequence[:, :-1]  # (batch_size, seq_len-1, num_keypoints, 3)
        
        # Compute loss on pose differences (should be small for smooth motion)
        loss = self.criterion(pose_diff, torch.zeros_like(pose_diff))
        
        # Apply valid mask if provided
        if valid_mask is not None:
            valid_diff = valid_mask[:, 1:] & valid_mask[:, :-1]  # Both frames must be valid
            loss = loss * valid_diff.unsqueeze(-1).float()
            
            # Average over valid keypoints
            num_valid = valid_diff.sum()
            if num_valid > 0:
                loss = loss.sum() / (num_valid * 3)
            else:
                loss = torch.tensor(0.0, device=pose_3d_sequence.device, requires_grad=True)
        else:
            loss = loss.mean()
            
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function for 3D pose estimation
    """
    def __init__(self, 
                 reprojection_weight=1.0,
                 pose_3d_weight=0.0,  # Only if 3D GT available
                 bone_length_weight=0.1,
                 temporal_weight=0.0,   # Only for video sequences
                 image_size=(256, 256)):
        super(CombinedLoss, self).__init__()
        
        self.reprojection_weight = reprojection_weight
        self.pose_3d_weight = pose_3d_weight
        self.bone_length_weight = bone_length_weight
        self.temporal_weight = temporal_weight
        
        # Initialize loss functions
        self.reprojection_loss = ReprojectionLoss(image_size=image_size)
        self.pose_3d_loss = Pose3DLoss()
        self.bone_length_loss = BoneLengthLoss()
        self.temporal_loss = TemporalConsistencyLoss()
    
    def forward(self, predictions, targets, valid_mask=None):
        """
        Args:
            predictions: dict with 'pose_3d' and 'camera_params'
            targets: dict with 'pose_2d' and optionally 'pose_3d'
            valid_mask: (batch_size, num_keypoints) boolean mask for valid keypoints
        Returns:
            dict with total loss and individual loss components
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Reprojection loss (main unsupervised loss)
        if self.reprojection_weight > 0:
            repr_loss = self.reprojection_loss(
                predictions['pose_3d'],
                predictions['camera_params'],
                targets['pose_2d'],
                valid_mask
            )
            loss_dict['reprojection'] = repr_loss
            total_loss += self.reprojection_weight * repr_loss
        
        # 3D pose loss (if 3D GT available)
        if self.pose_3d_weight > 0 and 'pose_3d' in targets:
            pose_3d_loss = self.pose_3d_loss(
                predictions['pose_3d'],
                targets['pose_3d'],
                valid_mask
            )
            loss_dict['pose_3d'] = pose_3d_loss
            total_loss += self.pose_3d_weight * pose_3d_loss
        
        # Bone length consistency loss
        if self.bone_length_weight > 0:
            bone_loss = self.bone_length_loss(
                predictions['pose_3d'],
                valid_mask=valid_mask
            )
            loss_dict['bone_length'] = bone_loss
            total_loss += self.bone_length_weight * bone_loss
        
        # Temporal consistency loss (for sequences)
        if self.temporal_weight > 0 and 'pose_3d_sequence' in predictions:
            temporal_loss = self.temporal_loss(
                predictions['pose_3d_sequence'],
                valid_mask
            )
            loss_dict['temporal'] = temporal_loss
            total_loss += self.temporal_weight * temporal_loss
        
        loss_dict['total'] = total_loss
        return loss_dict


if __name__ == "__main__":
    # Test loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 4
    num_keypoints = 37
    
    # Create dummy data
    pose_3d_pred = torch.randn(batch_size, num_keypoints, 3, device=device)
    pose_2d_gt = torch.randn(batch_size, num_keypoints, 3, device=device)  # [x, y, visibility]
    valid_mask = torch.ones(batch_size, num_keypoints, dtype=torch.bool, device=device)
    
    # Create dummy camera parameters
    camera_params = {
        'focal_length': torch.tensor([[1000.0]] * batch_size, device=device),
        'rotation': torch.zeros(batch_size, 3, device=device),
        'translation': torch.zeros(batch_size, 3, device=device),
        'scale': torch.ones(batch_size, 1, device=device),
        'center': torch.zeros(batch_size, 1, device=device)
    }
    
    # Test reprojection loss
    repr_loss = ReprojectionLoss()
    loss_value = repr_loss(pose_3d_pred, camera_params, pose_2d_gt, valid_mask)
    print(f"Reprojection loss: {loss_value.item():.4f}")
    
    # Test bone length loss
    bone_loss = BoneLengthLoss()
    loss_value = bone_loss(pose_3d_pred, valid_mask=valid_mask)
    print(f"Bone length loss: {loss_value.item():.4f}")
    
    # Test combined loss
    combined_loss = CombinedLoss()
    predictions = {'pose_3d': pose_3d_pred, 'camera_params': camera_params}
    targets = {'pose_2d': pose_2d_gt}
    
    loss_dict = combined_loss(predictions, targets, valid_mask)
    print(f"Combined loss: {loss_dict['total'].item():.4f}")
    print(f"Loss components: {[f'{k}: {v.item():.4f}' for k, v in loss_dict.items()]}")
    
    print("Loss function tests completed successfully!") 