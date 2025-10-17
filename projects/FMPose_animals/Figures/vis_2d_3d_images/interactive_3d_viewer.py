#!/usr/bin/env python3

"""
Interactive 3D Pose Viewer using Plotly
This creates an interactive web-based 3D visualization where you can:
- Rotate and zoom the 3D pose
- Hover over keypoints to see names and coordinates
- Toggle skeleton connections on/off
- View multiple poses side by side
"""

import os
import sys
import torch
import numpy as np
import json
import argparse
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Import our modules
from dataset import PrimateDataset
from model import Pose3DEstimator


def load_model(checkpoint_path, num_keypoints=37, embed_dim=256, num_heads=8, root_joint_idx=11, device='cpu'):
    """Load trained model from checkpoint"""
    model = Pose3DEstimator(
        num_keypoints=num_keypoints,
        embed_dim=embed_dim,
        num_heads=num_heads,
        root_joint_idx=root_joint_idx
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def create_interactive_3d_pose(pose_3d, keypoint_names, valid_mask=None, title="3D Pose", 
                               show_skeleton=True, point_size=8):
    """
    Create interactive 3D pose visualization using Plotly
    
    Args:
        pose_3d: numpy array (num_keypoints, 3) with [x, y, z] coordinates
        keypoint_names: list of keypoint names
        valid_mask: boolean mask for valid keypoints
        title: title for the plot
        show_skeleton: whether to show skeleton connections
        point_size: size of the keypoints
    
    Returns:
        plotly figure object
    """
    # Convert tensor to numpy if needed
    if torch.is_tensor(pose_3d):
        pose_3d = pose_3d.cpu().numpy()
    if valid_mask is not None and torch.is_tensor(valid_mask):
        valid_mask = valid_mask.cpu().numpy()
    
    # Create figure
    fig = go.Figure()
    
    # Filter valid keypoints
    if valid_mask is not None:
        valid_indices = np.where(valid_mask)[0]
        valid_pose = pose_3d[valid_mask]
        valid_names = [keypoint_names[i] if keypoint_names and i < len(keypoint_names) else f'Point_{i}' 
                      for i in valid_indices]
    else:
        valid_pose = pose_3d
        valid_names = [keypoint_names[i] if keypoint_names and i < len(keypoint_names) else f'Point_{i}' 
                      for i in range(len(pose_3d))]
        valid_indices = list(range(len(pose_3d)))
    
    # Create hover text with coordinates
    hover_text = []
    for i, (name, (x, y, z)) in enumerate(zip(valid_names, valid_pose)):
        hover_text.append(f"<b>{name}</b><br>" +
                         f"X: {x:.3f}<br>" +
                         f"Y: {y:.3f}<br>" +
                         f"Z: {z:.3f}<br>" +
                         f"Index: {valid_indices[i]}")
    
    # Add keypoints as scatter plot
    fig.add_trace(go.Scatter3d(
        x=valid_pose[:, 0],
        y=valid_pose[:, 1], 
        z=valid_pose[:, 2],
        mode='markers+text',
        marker=dict(
            size=point_size,
            color='red',
            opacity=0.8,
            line=dict(width=2, color='darkred')
        ),
        text=valid_names,
        textposition="top center",
        textfont=dict(size=10),
        hovertext=hover_text,
        hoverinfo='text',
        name='Keypoints'
    ))
    
    # Add skeleton connections if requested
    if show_skeleton:
        # Define skeleton connections (simplified version)
        connections = [
            (1, 11), (11, 12), (11, 13), (12, 22), (13, 23), 
            (11, 26), (26, 27), (26, 28), (27, 31), (28, 32), 
            (26, 36)
        ]
        
        # Create lines for skeleton
        for joint1, joint2 in connections:
            if (joint1 < len(pose_3d) and joint2 < len(pose_3d) and
                (valid_mask is None or (valid_mask[joint1] and valid_mask[joint2]))):
                
                fig.add_trace(go.Scatter3d(
                    x=[pose_3d[joint1, 0], pose_3d[joint2, 0]],
                    y=[pose_3d[joint1, 1], pose_3d[joint2, 1]],
                    z=[pose_3d[joint1, 2], pose_3d[joint2, 2]],
                    mode='lines',
                    line=dict(color='blue', width=4),
                    opacity=0.6,
                    hoverinfo='none',
                    showlegend=False
                ))
    
    # Update layout for better visualization
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color='black')
        ),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y", 
            zaxis_title="Z",
            xaxis=dict(showgrid=True, zeroline=True),
            yaxis=dict(showgrid=True, zeroline=True),
            zaxis=dict(showgrid=True, zeroline=True),
            bgcolor='white',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Set viewing angle
            )
        ),
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig


def create_multiple_poses_viewer(poses_3d_list, keypoint_names_list, valid_masks_list=None, 
                                titles=None, show_skeleton=True):
    """
    Create a viewer for multiple 3D poses side by side
    
    Args:
        poses_3d_list: list of pose arrays
        keypoint_names_list: list of keypoint names for each pose
        valid_masks_list: list of valid masks
        titles: list of titles for each pose
        show_skeleton: whether to show skeleton connections
    
    Returns:
        plotly figure with subplots
    """
    num_poses = len(poses_3d_list)
    cols = min(3, num_poses)  # Max 3 columns
    rows = (num_poses + cols - 1) // cols
    
    # Create subplot titles
    if titles is None:
        titles = [f"Pose {i+1}" for i in range(num_poses)]
    
    # Create subplots
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'scatter3d'} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=titles[:num_poses],
        horizontal_spacing=0.05,
        vertical_spacing=0.05
    )
    
    # Add each pose to its subplot
    for i, pose_3d in enumerate(poses_3d_list):
        row = i // cols + 1
        col = i % cols + 1
        
        keypoint_names = keypoint_names_list[i] if keypoint_names_list else None
        valid_mask = valid_masks_list[i] if valid_masks_list else None
        
        # Get valid points
        if valid_mask is not None:
            valid_pose = pose_3d[valid_mask]
            valid_names = [keypoint_names[j] if keypoint_names and j < len(keypoint_names) else f'Point_{j}' 
                          for j in np.where(valid_mask)[0]]
        else:
            valid_pose = pose_3d
            valid_names = [keypoint_names[j] if keypoint_names and j < len(keypoint_names) else f'Point_{j}' 
                          for j in range(len(pose_3d))]
        
        # Create hover text
        hover_text = []
        for name, (x, y, z) in zip(valid_names, valid_pose):
            hover_text.append(f"<b>{name}</b><br>" +
                             f"X: {x:.3f}<br>" +
                             f"Y: {y:.3f}<br>" +
                             f"Z: {z:.3f}")
        
        # Add keypoints
        fig.add_trace(
            go.Scatter3d(
                x=valid_pose[:, 0],
                y=valid_pose[:, 1],
                z=valid_pose[:, 2],
                mode='markers+text',
                marker=dict(size=6, color='red', opacity=0.8),
                text=valid_names,
                textposition="top center",
                textfont=dict(size=8),
                hovertext=hover_text,
                hoverinfo='text',
                name=f'Pose {i+1}',
                showlegend=(i == 0)
            ),
            row=row, col=col
        )
        
        # Add skeleton if requested
        if show_skeleton:
            connections = [
                (1, 11), (11, 12), (11, 13), (12, 22), (13, 23), 
                (11, 26), (26, 27), (26, 28), (27, 31), (28, 32), 
                (26, 36)
            ]
            
            for joint1, joint2 in connections:
                if (joint1 < len(pose_3d) and joint2 < len(pose_3d) and
                    (valid_mask is None or (valid_mask[joint1] and valid_mask[joint2]))):
                    
                    fig.add_trace(
                        go.Scatter3d(
                            x=[pose_3d[joint1, 0], pose_3d[joint2, 0]],
                            y=[pose_3d[joint1, 1], pose_3d[joint2, 1]],
                            z=[pose_3d[joint1, 2], pose_3d[joint2, 2]],
                            mode='lines',
                            line=dict(color='blue', width=3),
                            opacity=0.6,
                            hoverinfo='none',
                            showlegend=False
                        ),
                        row=row, col=col
                    )
    
    # Update layout
    fig.update_layout(
        height=400 * rows,
        width=1200,
        title_text="Interactive 3D Pose Viewer",
        paper_bgcolor='white'
    )
    
    return fig


def test_model_and_visualize(checkpoint_path, test_json, image_root, num_samples=5, 
                           output_html='interactive_3d_poses.html'):
    """
    Load model, run inference, and create interactive visualizations
    """
    print("Loading model and running inference...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(
        checkpoint_path, 
        num_keypoints=37,
        embed_dim=256,
        num_heads=8,
        root_joint_idx=11,
        device=device
    )
    
    # Create test dataset
    test_dataset = PrimateDataset(
        json_file=test_json,
        image_root=image_root,
        image_size=(256, 256)
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )
    
    keypoint_names = test_dataset.get_keypoint_names()
    
    # Run inference on first few samples
    poses_3d_list = []
    keypoint_names_list = []
    valid_masks_list = []
    titles = []
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
                
            images = batch['image'].to(device)
            pose_2d = batch['pose_2d'].to(device)
            valid_mask = pose_2d[:, :, 2] > 0
            
            # Forward pass
            predictions = model(pose_2d, valid_mask)
            pose_3d_pred = predictions['pose_3d'][0]  # Take first sample
            
            poses_3d_list.append(pose_3d_pred.cpu().numpy())
            keypoint_names_list.append(keypoint_names)
            valid_masks_list.append(valid_mask[0].cpu().numpy())
            titles.append(f"Sample {i+1}")
            
            print(f"Processed sample {i+1}/{num_samples}")
    
    # Create interactive visualization
    print("Creating interactive visualization...")
    fig = create_multiple_poses_viewer(
        poses_3d_list, keypoint_names_list, valid_masks_list, titles
    )
    
    # Save as HTML
    pyo.plot(fig, filename=output_html, auto_open=False)
    print(f"Interactive visualization saved to: {output_html}")
    print("\n🌐 To view the interactive plot:")
    print("1. Start HTTP server: python -m http.server 8080")
    print(f"2. Open browser: http://localhost:8080/{output_html}")
    print("3. Or download the file to your local machine")
    
    return fig


def create_demo_pose():
    """Create a demo 3D pose for testing"""
    # Simple demo pose
    demo_pose = np.array([
        [0, 0, 0],      # 0: root
        [0, 0, 1],      # 1: head
        [-0.2, 0, 0.8], # 2: left_eye
        [0.2, 0, 0.8],  # 3: right_eye
        # Add more points as needed...
    ])
    
    demo_names = ['root', 'head', 'left_eye', 'right_eye']
    
    return demo_pose, demo_names


def main():
    parser = argparse.ArgumentParser(description='Interactive 3D Pose Viewer')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='demo', 
                       choices=['demo', 'model'],
                       help='Demo mode or model inference mode')
    
    # Model arguments (for model mode)
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--test_json', type=str,
                       default='/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.3/splitted_test_datasets/oms_test.json',
                       help='Path to test JSON file')
    parser.add_argument('--image_root', type=str,
                       default='/home/ti_wang/data/tiwang/v8_coco/images',
                       help='Root directory for images')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--output_html', type=str, default='interactive_3d_poses.html',
                       help='Output HTML file path')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        print("Creating demo interactive 3D pose...")
        demo_pose, demo_names = create_demo_pose()
        
        fig = create_interactive_3d_pose(
            demo_pose, demo_names, 
            title="Demo 3D Pose - Interactive Viewer"
        )
        
        # Save without auto-opening
        pyo.plot(fig, filename='demo_3d_pose.html', auto_open=False)
        print("Demo visualization saved to: demo_3d_pose.html")
        print("\n🌐 To view the interactive plot:")
        print("1. Start HTTP server: python -m http.server 8080")
        print("2. Open browser: http://localhost:8080/demo_3d_pose.html")
        print("3. Or download the file to your local machine")
        
    elif args.mode == 'model':
        if not args.checkpoint:
            print("Error: --checkpoint required for model mode")
            return
        
        test_model_and_visualize(
            args.checkpoint, args.test_json, args.image_root,
            args.num_samples, args.output_html
        )
    
    print("\nInteractive features:")
    print("- Hover over points to see coordinates and names")
    print("- Drag to rotate the 3D view")
    print("- Scroll to zoom in/out")
    print("- Use toolbar for pan, zoom, and other controls")


if __name__ == "__main__":
    main() 