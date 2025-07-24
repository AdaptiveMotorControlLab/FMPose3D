import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from tqdm import tqdm
import logging
from datetime import datetime

# Import our modules
from dataset import PrimateDataset, create_data_loaders
from model import Pose3DEstimator, project_3d_to_2d
from loss import CombinedLoss


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def load_model(checkpoint_path, num_keypoints=37, backbone='resnet50', device='cpu'):
    """Load trained model from checkpoint"""
    model = Pose3DEstimator(
        num_keypoints=num_keypoints,
        backbone=backbone,
        pretrained=False  # Not needed for inference
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def visualize_2d_pose(image, pose_2d, keypoint_names, valid_mask=None, save_path=None):
    """
    Visualize 2D pose on image
    """
    # Convert tensor to numpy if needed
    if torch.is_tensor(image):
        if image.shape[0] == 3:  # CHW format
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
    
    if torch.is_tensor(pose_2d):
        pose_2d = pose_2d.cpu().numpy()
    
    if valid_mask is not None and torch.is_tensor(valid_mask):
        valid_mask = valid_mask.cpu().numpy()
    
    # Denormalize image if needed (assuming ImageNet normalization)
    if image.dtype != np.uint8:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image * std) + mean
        image = np.clip(image, 0, 1)
    
    # Convert to BGR for OpenCV
    image_vis = (image * 255).astype(np.uint8)
    if image_vis.shape[2] == 3:
        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
    
    # Draw keypoints
    for i, (x, y, vis) in enumerate(pose_2d):
        if valid_mask is not None and not valid_mask[i]:
            continue
            
        if vis > 0:  # Valid keypoint
            cv2.circle(image_vis, (int(x), int(y)), 3, (0, 255, 0), -1)
            # Add keypoint name
            cv2.putText(image_vis, f'{i}', (int(x)+5, int(y)+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    if save_path:
        cv2.imwrite(save_path, image_vis)
    
    return image_vis


def visualize_3d_pose(pose_3d, keypoint_names, valid_mask=None, save_path=None, title="3D Pose"):
    """
    Visualize 3D pose using matplotlib
    """
    # Convert tensor to numpy if needed
    if torch.is_tensor(pose_3d):
        pose_3d = pose_3d.cpu().numpy()
    if valid_mask is not None and torch.is_tensor(valid_mask):
        valid_mask = valid_mask.cpu().numpy()
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot keypoints
    if valid_mask is not None:
        valid_pose = pose_3d[valid_mask]
        ax.scatter(valid_pose[:, 0], valid_pose[:, 1], valid_pose[:, 2], 
                  c='red', s=50, alpha=0.8)
        
        # Add labels for valid keypoints
        for i, (x, y, z) in enumerate(pose_3d):
            if valid_mask[i]:
                # Use keypoint name if available, otherwise use index
                if keypoint_names is not None and i < len(keypoint_names):
                    label = keypoint_names[i]
                else:
                    label = f'{i}'
                ax.text(x, y, z, label, fontsize=7, ha='center', va='bottom')
    else:
        ax.scatter(pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2], 
                  c='red', s=50, alpha=0.8)
        
        # Add labels
        for i, (x, y, z) in enumerate(pose_3d):
            # Use keypoint name if available, otherwise use index
            if keypoint_names is not None and i < len(keypoint_names):
                label = keypoint_names[i]
            else:
                label = f'{i}'
            ax.text(x, y, z, label, fontsize=7, ha='center', va='bottom')
    
    # Draw skeleton connections (simplified)
    # connections = [
    #     # Head connections
    #     (1, 2), (1, 3), (1, 4), (2, 5), (3, 6),
    #     # Torso connections
    #     (11, 12), (11, 13), (12, 14), (13, 14), (14, 15), (15, 16), (16, 17),
    #     # Arm connections
    #     (12, 18), (18, 20), (20, 22), (13, 19), (19, 21), (21, 23),
    #     # Hip connections
    #     (16, 24), (16, 25), (24, 26), (25, 26),
    #     # Leg connections
    #     (24, 27), (27, 29), (29, 31), (25, 28), (28, 30), (30, 32),
    #     # Tail connections
    #     (17, 33), (33, 34), (34, 35), (35, 36)
    # ]
   
    connections = [
        (1, 11), (11, 12), (11, 13), (12, 22), (13, 23), 
        (11, 26), (26, 27), (26, 28), (27, 31), (28, 32), 
        (26, 36)
    ]
    
    
    for joint1, joint2 in connections:
        if joint1 < len(pose_3d) and joint2 < len(pose_3d):
            if valid_mask is None or (valid_mask[joint1] and valid_mask[joint2]):
                ax.plot([pose_3d[joint1, 0], pose_3d[joint2, 0]],
                       [pose_3d[joint1, 1], pose_3d[joint2, 1]],
                       [pose_3d[joint1, 2], pose_3d[joint2, 2]], 'b-', alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
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


def evaluate_reprojection_error(pose_3d_pred, camera_params_pred, pose_2d_gt, valid_mask, image_size):
    """
    Compute reprojection error metrics
    """
    # Project 3D to 2D
    pose_2d_proj = project_3d_to_2d(pose_3d_pred, camera_params_pred, image_size)
    
    # Extract GT 2D coordinates
    pose_2d_gt_xy = pose_2d_gt[:, :, :2]
    
    # Compute errors
    errors = torch.norm(pose_2d_proj - pose_2d_gt_xy, dim=2)  # L2 distance
    
    if valid_mask is not None:
        errors = errors * valid_mask.float()
        num_valid = valid_mask.sum(dim=1).float()
        num_valid = torch.clamp(num_valid, min=1.0)
        mean_errors = errors.sum(dim=1) / num_valid  # Per sample
    else:
        mean_errors = errors.mean(dim=1)
    
    return {
        'per_keypoint_errors': errors,
        'per_sample_errors': mean_errors,
        'overall_error': mean_errors.mean()
    }


def test_model(model, test_loader, device, output_dir, num_vis_samples=10):
    """Test the model and generate visualizations"""
    logger = logging.getLogger(__name__)
    
    model.eval()
    all_errors = []
    
    # Create output directories
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    logger.info("Starting evaluation...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            # Move data to device
            images = batch['image'].to(device)
            pose_2d = batch['pose_2d'].to(device)
            
            # Create valid mask from visibility
            valid_mask = pose_2d[:, :, 2] > 0
            
            # Forward pass
            predictions = model(images, pose_2d, valid_mask)
            pose_3d_pred = predictions['pose_3d']
            camera_params_pred = predictions['camera_params']
            
            # Compute reprojection errors
            error_metrics = evaluate_reprojection_error(
                pose_3d_pred, camera_params_pred, pose_2d, valid_mask, 
                image_size=(256, 256)
            )
            
            all_errors.extend(error_metrics['per_sample_errors'].cpu().numpy())
            
            # Visualize first few samples
            if batch_idx == 0 and num_vis_samples > 0:
                batch_size = min(num_vis_samples, images.shape[0])
                
                for i in range(batch_size):
                    sample_id = batch_idx * test_loader.batch_size + i
                    
                    # Get keypoint names
                    keypoint_names = test_loader.dataset.get_keypoint_names()
                    
                    # Visualize 2D pose on image
                    image_2d_path = os.path.join(vis_dir, f'sample_{sample_id:04d}_2d.jpg')
                    visualize_2d_pose(
                        images[i], pose_2d[i], keypoint_names,
                        valid_mask[i], image_2d_path
                    )
                    
                    # Visualize predicted 3D pose
                    pose_3d_path = os.path.join(vis_dir, f'sample_{sample_id:04d}_3d.png')
                    fig = visualize_3d_pose(
                        pose_3d_pred[i], keypoint_names, valid_mask[i],
                        pose_3d_path, f"Sample {sample_id} - Predicted 3D Pose"
                    )
                    plt.close(fig)
                    
                    # Visualize reprojected 2D pose
                    pose_2d_proj = project_3d_to_2d(
                        pose_3d_pred[i:i+1], 
                        {k: v[i:i+1] for k, v in camera_params_pred.items()},
                        image_size=(256, 256)
                    )
                    
                    # Create comparison image
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Original image with GT 2D pose
                    image_np = images[i].permute(1, 2, 0).cpu().numpy()
                    if image_np.max() <= 1.0:
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        image_np = image_np * std + mean
                        image_np = np.clip(image_np, 0, 1)
                    
                    axes[0].imshow(image_np)
                    axes[0].set_title('Ground Truth 2D Pose')
                    
                    # Plot GT keypoints
                    pose_2d_np = pose_2d[i].cpu().numpy()
                    for j, (x, y, vis) in enumerate(pose_2d_np):
                        if valid_mask[i, j]:
                            axes[0].plot(x, y, 'ro', markersize=4)
                    
                    # Reprojected pose
                    axes[1].imshow(image_np)
                    axes[1].set_title('Reprojected 2D Pose')
                    
                    # Plot reprojected keypoints
                    pose_2d_proj_np = pose_2d_proj[0].cpu().numpy()
                    for j, (x, y) in enumerate(pose_2d_proj_np):
                        if valid_mask[i, j]:
                            axes[1].plot(x, y, 'bo', markersize=4)
                    
                    comparison_path = os.path.join(vis_dir, f'sample_{sample_id:04d}_comparison.png')
                    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    
                    logger.info(f"Saved visualizations for sample {sample_id}")
    
    # Compute final statistics
    all_errors = np.array(all_errors)
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    median_error = np.median(all_errors)
    
    logger.info(f"Evaluation Results:")
    logger.info(f"Mean reprojection error: {mean_error:.2f} pixels")
    logger.info(f"Std reprojection error: {std_error:.2f} pixels")
    logger.info(f"Median reprojection error: {median_error:.2f} pixels")
    
    # Save results
    results = {
        'mean_error': float(mean_error),
        'std_error': float(std_error),
        'median_error': float(median_error),
        'all_errors': all_errors.tolist()
    }
    
    results_path = os.path.join(output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_path}")
    
    return results


def create_test_dir(test_json_path, dataset_name=None, base_dir="test_results"):
    """
    Create a timestamped test results directory
    Args:
        test_json_path: Path to test JSON file
        dataset_name: Custom dataset name (if None, extracts from JSON path)
        base_dir: Base directory for test results
    Returns:
        str: Path to the created test directory
    """
    # Use provided dataset name or extract from JSON path
    if dataset_name is None:
        json_filename = os.path.basename(test_json_path)
        dataset_name = json_filename.replace('.json', '').replace('_test', '')
    
    # Create timestamp string (YYYYMMDD_HHMM)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Create test directory name
    test_name = f"{dataset_name}_test_{timestamp}"
    test_dir = os.path.join(base_dir, test_name)
    
    # Create directory
    os.makedirs(test_dir, exist_ok=True)
    
    return test_dir


def main():
    parser = argparse.ArgumentParser(description='Test 3D Pose Estimation Model')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_json', type=str,
                       default='/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.3/splitted_test_datasets/ap10k_test.json',
                       help='Path to test JSON file')
    parser.add_argument('--image_root', type=str, default='/home/ti_wang/data/tiwang/v8_coco/images',
                       help='Root directory for images')
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='Dataset name for test results folder (if not provided, extracts from JSON filename)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results and visualizations. If not provided, will create timestamped directory')
    
    # Model configuration
    parser.add_argument('--backbone', type=str, default='resnet50',
                       choices=['resnet18', 'resnet34', 'resnet50'],
                       help='Backbone architecture')
    parser.add_argument('--num_keypoints', type=int, default=37,
                       help='Number of keypoints')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                       help='Image size [width, height]')
    
    # Testing arguments
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--num_vis_samples', type=int, default=10,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    # Create timestamped output directory if not provided
    if args.output_dir is None:
        args.output_dir = create_test_dir(args.test_json, args.dataset_name)
        print(f"Created test results directory: {args.output_dir}")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging()
    logger.info(f"Test results directory: {args.output_dir}")
    logger.info(f"Starting testing with arguments: {vars(args)}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from: {args.checkpoint}")
    model = load_model(
        args.checkpoint,
        num_keypoints=args.num_keypoints,
        backbone=args.backbone,
        device=device
    )
    
    # Create test dataset and loader
    logger.info("Creating test data loader...")
    test_dataset = PrimateDataset(
        json_file=args.test_json,
        image_root=args.image_root,
        image_size=tuple(args.image_size)
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Test the model
    results = test_model(
        model, test_loader, device, args.output_dir, args.num_vis_samples
    )
    
    logger.info("Testing completed successfully!")

if __name__ == "__main__":
    main() 