#!/usr/bin/env python3

"""
Interactive 3D Pose Viewer for Rat7M using Plotly
This creates an interactive web-based 3D visualization where you can:
- Rotate and zoom the 3D pose
- Hover over keypoints to see names and coordinates
- Toggle skeleton connections on/off
"""

import os
import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from scipy.io import loadmat


def load_rat7m_skeleton(skeleton_file='jesse_skeleton.mat'):
    """
    Load Rat7M skeleton information from .mat file
    
    Returns:
        joint_names: list of joint names (strings)
        skeleton_connections: list of (joint1, joint2) tuples for skeleton connections
    """
    skeleton_mat = loadmat(skeleton_file)
    
    # Get joint names and clean up the format
    # joint_names comes as nested array structure, extract actual strings
    joint_names_raw = skeleton_mat['joint_names']
    joint_names = []
    for item in joint_names_raw:
        if isinstance(item, np.ndarray) and len(item) > 0:
            # Extract the actual string from nested array
            name = item[0]
            if isinstance(name, np.ndarray):
                name = str(name[0]) if len(name) > 0 else f"Joint_{len(joint_names)}"
            else:
                name = str(name)
            joint_names.append(name)
        else:
            joint_names.append(f"Joint_{len(joint_names)}")
    
    # Get skeleton connections (joints_idx is 1-based, convert to 0-based)
    joints_idx = np.array(skeleton_mat['joints_idx']) - 1
    
    # joints_idx shape: (num_connections, 2), each row is [joint1, joint2]
    skeleton_connections = [(int(joints_idx[i, 0]), int(joints_idx[i, 1])) 
                           for i in range(joints_idx.shape[0])]
    
    return joint_names, skeleton_connections


def create_interactive_3d_pose_rat7m(pose_3d, joint_names, skeleton_connections, 
                                     valid_mask=None, title="Rat7M 3D Pose", 
                                     show_skeleton=True, point_size=8, 
                                     show_joint_labels=True):
    """
    Create interactive 3D pose visualization for Rat7M using Plotly
    
    Args:
        pose_3d: numpy array (num_keypoints, 3) with [x, y, z] coordinates
        joint_names: list of joint names (length=num_keypoints)
        skeleton_connections: list of (joint1, joint2) tuples
        valid_mask: boolean mask for valid keypoints
        title: title for the plot
        show_skeleton: whether to show skeleton connections
        point_size: size of the keypoints
        show_joint_labels: whether to show joint name labels
    
    Returns:
        plotly figure object
    """
    # Ensure pose_3d is numpy array
    if not isinstance(pose_3d, np.ndarray):
        pose_3d = np.array(pose_3d)
    
    # Create figure
    fig = go.Figure()
    
    # Filter valid keypoints
    if valid_mask is not None:
        if not isinstance(valid_mask, np.ndarray):
            valid_mask = np.array(valid_mask)
        valid_indices = np.where(valid_mask)[0]
        valid_pose = pose_3d[valid_mask]
        valid_names = [joint_names[i] if i < len(joint_names) else f'Joint_{i}' 
                      for i in valid_indices]
    else:
        valid_pose = pose_3d
        valid_names = [joint_names[i] if i < len(joint_names) else f'Joint_{i}' 
                      for i in range(len(pose_3d))]
        valid_indices = list(range(len(pose_3d)))
    
    # Create hover text with coordinates
    hover_text = []
    for i, (name, (x, y, z)) in enumerate(zip(valid_names, valid_pose)):
        hover_text.append(
            f"<b>{name}</b><br>" +
            f"Index: {valid_indices[i]}<br>" +
            f"X: {x:.2f} mm<br>" +
            f"Y: {y:.2f} mm<br>" +
            f"Z: {z:.2f} mm"
        )
    
    # Add skeleton connections first (so they appear behind points)
    if show_skeleton and skeleton_connections:
        for joint1, joint2 in skeleton_connections:
            if (joint1 < len(pose_3d) and joint2 < len(pose_3d) and
                (valid_mask is None or (valid_mask[joint1] and valid_mask[joint2]))):
                
                # Check for invalid coordinates (NaN, Inf, or very large values)
                p1 = pose_3d[joint1]
                p2 = pose_3d[joint2]
                
                # Skip connection if either point has invalid coordinates
                if (np.any(np.isnan(p1)) or np.any(np.isnan(p2)) or
                    np.any(np.isinf(p1)) or np.any(np.isinf(p2)) or
                    np.linalg.norm(p1) > 10000 or np.linalg.norm(p2) > 10000):
                    # Print warning about skipped connection
                    name1 = joint_names[joint1] if joint1 < len(joint_names) else f'Joint_{joint1}'
                    name2 = joint_names[joint2] if joint2 < len(joint_names) else f'Joint_{joint2}'
                    print(f"⚠️ Skipping connection {name1}-{name2}: invalid coordinates")
                    continue
                
                fig.add_trace(go.Scatter3d(
                    x=[pose_3d[joint1, 0], pose_3d[joint2, 0]],
                    y=[pose_3d[joint1, 1], pose_3d[joint2, 1]],
                    z=[pose_3d[joint1, 2], pose_3d[joint2, 2]],
                    mode='lines',
                    line=dict(color='lightblue', width=6),
                    opacity=0.7,
                    hoverinfo='none',
                    showlegend=False,
                    name='Skeleton'
                ))
    
    # Add keypoints as scatter plot
    fig.add_trace(go.Scatter3d(
        x=valid_pose[:, 0],
        y=valid_pose[:, 1], 
        z=valid_pose[:, 2],
        mode='markers+text' if show_joint_labels else 'markers',
        marker=dict(
            size=point_size,
            color='red',
            opacity=0.9,
            line=dict(width=2, color='darkred'),
            symbol='circle'
        ),
        text=valid_names if show_joint_labels else None,
        textposition="top center",
        textfont=dict(size=9, color='black', family='Arial Bold'),
        hovertext=hover_text,
        hoverinfo='text',
        name='Keypoints'
    ))
    
    # Calculate axis ranges for equal aspect ratio
    # Filter out invalid coordinates for range calculation
    valid_coords_mask = np.all(~np.isnan(pose_3d), axis=1) & np.all(~np.isinf(pose_3d), axis=1)
    valid_coords_mask = valid_coords_mask & (np.linalg.norm(pose_3d, axis=1) < 10000)
    
    if np.any(valid_coords_mask):
        valid_coords_for_range = pose_3d[valid_coords_mask]
        all_coords = valid_coords_for_range.reshape(-1)
        coord_range = np.max(all_coords) - np.min(all_coords)
        center = np.mean(valid_coords_for_range, axis=0)
        range_pad = coord_range * 0.6  # Add padding
    else:
        # Fallback if all coordinates are invalid
        center = np.array([0, 0, 0])
        range_pad = 1000
    
    # Update layout for better visualization
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color='black', family='Arial Bold'),
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(
                title="X (mm)",
                showgrid=True,
                zeroline=True,
                gridcolor='lightgray',
                range=[center[0] - range_pad, center[0] + range_pad]
            ),
            yaxis=dict(
                title="Y (mm)",
                showgrid=True,
                zeroline=True,
                gridcolor='lightgray',
                range=[center[1] - range_pad, center[1] + range_pad]
            ),
            zaxis=dict(
                title="Z (mm)",
                showgrid=True,
                zeroline=True,
                gridcolor='lightgray',
                range=[center[2] - range_pad, center[2] + range_pad]
            ),
            bgcolor='white',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),  # Set viewing angle
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='cube'  # Equal aspect ratio
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=60),
        paper_bgcolor='white',
        plot_bgcolor='white',
        showlegend=True
    )
    
    return fig


def create_comparison_viewer_rat7m(poses_3d_list, joint_names, skeleton_connections,
                                   titles=None, show_skeleton=True):
    """
    Create a viewer for multiple Rat7M 3D poses side by side
    
    Args:
        poses_3d_list: list of pose arrays, each (num_keypoints, 3)
        joint_names: list of joint names
        skeleton_connections: list of (joint1, joint2) tuples
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
        horizontal_spacing=0.08,
        vertical_spacing=0.1
    )
    
    # Add each pose to its subplot
    for idx, pose_3d in enumerate(poses_3d_list):
        row = idx // cols + 1
        col = idx % cols + 1
        
        # Add skeleton connections first
        if show_skeleton and skeleton_connections:
            for joint1, joint2 in skeleton_connections:
                if joint1 < len(pose_3d) and joint2 < len(pose_3d):
                    fig.add_trace(
                        go.Scatter3d(
                            x=[pose_3d[joint1, 0], pose_3d[joint2, 0]],
                            y=[pose_3d[joint1, 1], pose_3d[joint2, 1]],
                            z=[pose_3d[joint1, 2], pose_3d[joint2, 2]],
                            mode='lines',
                            line=dict(color='lightblue', width=4),
                            opacity=0.7,
                            hoverinfo='none',
                            showlegend=False
                        ),
                        row=row, col=col
                    )
        
        # Create hover text
        hover_text = []
        for i, (name, (x, y, z)) in enumerate(zip(joint_names, pose_3d)):
            hover_text.append(
                f"<b>{name}</b><br>" +
                f"X: {x:.2f}<br>" +
                f"Y: {y:.2f}<br>" +
                f"Z: {z:.2f}"
            )
        
        # Add keypoints
        fig.add_trace(
            go.Scatter3d(
                x=pose_3d[:, 0],
                y=pose_3d[:, 1],
                z=pose_3d[:, 2],
                mode='markers+text',
                marker=dict(size=6, color='red', opacity=0.9),
                text=[name[:6] for name in joint_names],  # Abbreviated names
                textposition="top center",
                textfont=dict(size=7),
                hovertext=hover_text,
                hoverinfo='text',
                showlegend=(idx == 0),
                name='Keypoints'
            ),
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        height=500 * rows,
        width=1400,
        title_text="Rat7M 3D Pose Comparison",
        title_font=dict(size=20),
        paper_bgcolor='white'
    )
    
    return fig


def visualize_rat7m_pose(pose_3d, skeleton_file='jesse_skeleton.mat', 
                        output_html='rat7m_3d_pose.html', 
                        title="Rat7M 3D Pose",
                        auto_open=False):
    """
    Visualize a single Rat7M 3D pose and save as HTML
    
    Args:
        pose_3d: numpy array (20, 3) with [x, y, z] coordinates
        skeleton_file: path to jesse_skeleton.mat
        output_html: output HTML file path
        title: plot title
        auto_open: whether to automatically open the HTML file
    
    Returns:
        plotly figure object
    """
    # Load skeleton information
    joint_names, skeleton_connections = load_rat7m_skeleton(skeleton_file)
    
    # Create visualization
    fig = create_interactive_3d_pose_rat7m(
        pose_3d, 
        joint_names, 
        skeleton_connections,
        title=title,
        show_skeleton=True,
        show_joint_labels=True
    )
    
    # Save as HTML
    pyo.plot(fig, filename=output_html, auto_open=auto_open)
    print(f"✓ Interactive visualization saved to: {output_html}")
    
    if not auto_open:
        print("\n🌐 To view the interactive plot:")
        print(f"  Open in browser: {os.path.abspath(output_html)}")
        print("  Or start HTTP server: python -m http.server 8080")
    
    return fig


def main():
    """Demo: create visualization with example data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Interactive Rat7M 3D Pose Viewer')
    parser.add_argument('--skeleton', type=str, default='jesse_skeleton.mat',
                       help='Path to jesse_skeleton.mat file')
    parser.add_argument('--output', type=str, default='rat7m_3d_pose.html',
                       help='Output HTML file')
    parser.add_argument('--demo', action='store_true',
                       help='Create demo visualization with random pose')
    
    args = parser.parse_args()
    
    if args.demo:
        print("Creating demo Rat7M 3D pose...")
        # Create random demo pose (20 joints)
        demo_pose = np.random.randn(20, 3) * 50 + np.array([500, 300, 1800])
        
        visualize_rat7m_pose(
            demo_pose,
            skeleton_file=args.skeleton,
            output_html=args.output,
            title="Demo Rat7M 3D Pose",
            auto_open=False
        )
        
        print("\nInteractive features:")
        print("  - Hover over points to see joint names and coordinates")
        print("  - Drag to rotate the 3D view")
        print("  - Scroll to zoom in/out")
        print("  - Use toolbar for pan, zoom, and other controls")
    else:
        print("Use --demo to create a demo visualization")
        print("Or import this module to use the functions in your code")


if __name__ == "__main__":
    main()

