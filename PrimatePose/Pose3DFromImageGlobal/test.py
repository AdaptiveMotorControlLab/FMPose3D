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
import shutil

# Import our modules
from dataset import PrimateDataset, create_data_loaders
from models.model import Pose3DEstimator, project_3d_to_2d
from loss import CombinedLoss
from interactive_3d_viewer import create_interactive_3d_pose
import plotly.offline as pyo


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def copy_source_files(test_dir):
    """
    Copy source files to test directory for reproducibility
    Args:
        test_dir: Path to test results directory
    """
    # Create source code backup directory
    source_dir = os.path.join(test_dir, 'source_code')
    os.makedirs(source_dir, exist_ok=True)
    
    # Files to copy
    files_to_copy = [
        'test.py',
        'models/model.py',
        'loss.py',
        'dataset.py',
        'interactive_3d_viewer.py'
    ]
    
    for file_path in files_to_copy:
        if os.path.exists(file_path):
            # Create subdirectories if needed
            dest_path = os.path.join(source_dir, file_path)
            dest_dir = os.path.dirname(dest_path)
            os.makedirs(dest_dir, exist_ok=True)
            
            # Copy file
            shutil.copy2(file_path, dest_path)
            print(f"Copied {file_path} to {dest_path}")
        else:
            print(f"Warning: {file_path} not found, skipping")


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
            cv2.circle(image_vis, (int(x), int(y)), 3, (0, 0, 255), -1)
            # Add keypoint name
            # cv2.putText(image_vis, f'{i}', (int(x)+5, int(y)+5),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
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
    
    # Fix coordinate system - flip Y and Z axes to correct orientation
    pose_3d_corrected = pose_3d.copy()
    # pose_3d_corrected[:, 1] = -pose_3d[:, 1]  # Flip Y axis
    # pose_3d_corrected[:, 2] = -pose_3d[:, 2]  # Flip Z axis
    
    # Plot keypoints
    if valid_mask is not None:
        valid_pose = pose_3d_corrected[valid_mask]
        ax.scatter(valid_pose[:, 0], valid_pose[:, 1], valid_pose[:, 2], 
                  c='red', s=50, alpha=0.8)
        
        # Add labels for valid keypoints
        for i, (x, y, z) in enumerate(pose_3d_corrected):
            if valid_mask[i]:
                # Use keypoint name if available, otherwise use index
                if keypoint_names is not None and i < len(keypoint_names):
                    label = keypoint_names[i]
                else:
                    label = f'{i}'
                # ax.text(x, y, z, label, fontsize=6, ha='center', va='bottom')
    else:
        ax.scatter(pose_3d_corrected[:, 0], pose_3d_corrected[:, 1], pose_3d_corrected[:, 2], 
                  c='red', s=50, alpha=0.8)
        
        # Add labels
        for i, (x, y, z) in enumerate(pose_3d_corrected):
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
        if joint1 < len(pose_3d_corrected) and joint2 < len(pose_3d_corrected):
            if valid_mask is None or (valid_mask[joint1] and valid_mask[joint2]):
                ax.plot([pose_3d_corrected[joint1, 0], pose_3d_corrected[joint2, 0]],
                       [pose_3d_corrected[joint1, 1], pose_3d_corrected[joint2, 1]],
                       [pose_3d_corrected[joint1, 2], pose_3d_corrected[joint2, 2]], 'b-', alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # 确保3D pose居中显示
    # 计算pose的中心点
    if valid_mask is not None:
        valid_points = pose_3d_corrected[valid_mask]
        if len(valid_points) > 0:
            center_x = valid_points[:, 0].mean()
            center_y = valid_points[:, 1].mean()
            center_z = valid_points[:, 2].mean()
        else:
            center_x = center_y = center_z = 0
    else:
        center_x = pose_3d_corrected[:, 0].mean()
        center_y = pose_3d_corrected[:, 1].mean()
        center_z = pose_3d_corrected[:, 2].mean()
    
    # 计算pose的范围
    if valid_mask is not None:
        valid_points = pose_3d_corrected[valid_mask]
        if len(valid_points) > 0:
            range_x = valid_points[:, 0].max() - valid_points[:, 0].min()
            range_y = valid_points[:, 1].max() - valid_points[:, 1].min()
            range_z = valid_points[:, 2].max() - valid_points[:, 2].min()
        else:
            range_x = range_y = range_z = 1.0
    else:
        range_x = pose_3d_corrected[:, 0].max() - pose_3d_corrected[:, 0].min()
        range_y = pose_3d_corrected[:, 1].max() - pose_3d_corrected[:, 1].min()
        range_z = pose_3d_corrected[:, 2].max() - pose_3d_corrected[:, 2].min()
    
    # 使用最大范围确保等比例显示，并添加一些边距
    max_range = max(range_x, range_y, range_z) / 2.0
    margin = max_range * 0.2  # 添加20%的边距
    
    # 设置坐标轴范围，确保pose居中
    ax.set_xlim(center_x - max_range - margin, center_x + max_range + margin)
    ax.set_ylim(center_y - max_range - margin, center_y + max_range + margin)
    ax.set_zlim(center_z - max_range - margin, center_z + max_range + margin)
    
    # 设置等比例显示
    ax.set_box_aspect([1, 1, 1])
    
    # 设置网格和背景
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Set better viewing angle - rotated 45 degrees clockwise around z-axis
    # elev 仰角: 上下 （x-axis）； azim 方位角: 左右（z-axis）
    # 10, 260
    # ax.view_init(elev=10, azim=260) # default value: elev= 30, azim=-60
    ax.view_init(elev=45, azim=60) # default value: elev= 30, azim=-60
    # ax.view_init(elev=10, azim=-70) # default value: elev= x, azim=60 
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
                    
                    # Visualize predicted 3D pose (static matplotlib)
                    pose_3d_path = os.path.join(vis_dir, f'sample_{sample_id:04d}_3d.png')
                    fig = visualize_3d_pose(
                        pose_3d_pred[i], keypoint_names, valid_mask[i],
                        pose_3d_path, f"Sample {sample_id} - Predicted 3D Pose"
                    )
                    plt.close(fig)
                    
                    # Generate interactive 3D pose (HTML)
                    pose_3d_html_path = os.path.join(vis_dir, f'sample_{sample_id:04d}_3d_interactive.html')
                    try:
                        interactive_fig = create_interactive_3d_pose(
                            pose_3d_pred[i], keypoint_names, valid_mask[i],
                            title=f"Sample {sample_id} - Interactive 3D Pose",
                            show_skeleton=True, point_size=8
                        )
                        pyo.plot(interactive_fig, filename=pose_3d_html_path, auto_open=False)
                        logger.info(f"Saved interactive 3D visualization: {pose_3d_html_path}")
                    except Exception as e:
                        logger.warning(f"Failed to generate interactive 3D visualization for sample {sample_id}: {e}")
                        continue
                    
                    # Visualize reprojected 2D pose
                    pose_2d_proj = project_3d_to_2d(
                        pose_3d_pred[i:i+1], 
                        {k: v[i:i+1] for k, v in camera_params_pred.items()},
                        image_size=(256, 256)
                    )
                    
                    # Create comparison image
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Original image with GT 2D pose - use same processing as visualize_2d_pose
                    image_np = images[i].permute(1, 2, 0).cpu().numpy()
                    # Denormalize image if needed (assuming ImageNet normalization)
                    if image_np.dtype != np.uint8:
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        image_np = (image_np * std) + mean
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
                    
                    logger.info(f"Saved all visualizations for sample {sample_id} (2D, 3D PNG, 3D HTML, comparison)")
    
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
    
    # Information about interactive visualizations
    logger.info("📁 Visualization files generated:")
    logger.info(f"  📊 Static visualizations: {vis_dir}/*_2d.jpg, *_3d.png, *_comparison.png")
    logger.info(f"  🌐 Interactive 3D poses: {vis_dir}/*_3d_interactive.html")
    logger.info("💡 To view interactive 3D poses:")
    logger.info("  1. Start HTTP server: python -m http.server 8080")
    logger.info(f"  2. Open browser: http://localhost:8080/{vis_dir.replace(output_dir + '/', '')}/*_3d_interactive.html")
    logger.info("  3. Or download HTML files to your local machine")
    
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
    
    # Copy source files for reproducibility
    print("Copying source files for reproducibility...")
    copy_source_files(args.output_dir)
    
    # Save test arguments for reproducibility
    args_file = os.path.join(args.output_dir, 'test_args.txt')
    with open(args_file, 'w') as f:
        f.write("Test Arguments:\n")
        f.write("=" * 50 + "\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("Command line:\n")
        f.write(" ".join(sys.argv) + "\n")
    print(f"Saved test arguments to: {args_file}")
    
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
    
    # Save checkpoint information for reproducibility
    checkpoint_info_file = os.path.join(args.output_dir, 'checkpoint_info.txt')
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        with open(checkpoint_info_file, 'w') as f:
            f.write("Checkpoint Information:\n")
            f.write("=" * 50 + "\n")
            f.write(f"Checkpoint path: {args.checkpoint}\n")
            if 'epoch' in checkpoint:
                f.write(f"Epoch: {checkpoint['epoch']}\n")
            if 'best_loss' in checkpoint:
                f.write(f"Best loss: {checkpoint['best_loss']:.6f}\n")
            if 'best_epoch' in checkpoint:
                f.write(f"Best epoch: {checkpoint['best_epoch']}\n")
            f.write(f"Model architecture: {args.backbone}\n")
            f.write(f"Number of keypoints: {args.num_keypoints}\n")
            
            # Count model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Trainable parameters: {trainable_params:,}\n")
        print(f"Saved checkpoint information to: {checkpoint_info_file}")
    except Exception as e:
        logger.warning(f"Could not save checkpoint information: {e}")
    
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