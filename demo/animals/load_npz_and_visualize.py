#!/usr/bin/env python3
"""
简单示例：加载 .npz 文件中的 3D 姿态并生成交互式 HTML 可视化

使用方法:
    python load_npz_and_visualize.py --npz_file your_pose.npz --output pose_viewer.html
"""

import numpy as np
import argparse
import plotly.graph_objects as go
import plotly.offline as pyo
import os


def load_pose_from_npz(npz_file, key=None):
    """
    从 .npz 文件加载 3D 姿态数据
    
    Args:
        npz_file: .npz 文件路径
        key: npz 文件中的键名。如果为 None，会列出所有可用的键
    
    Returns:
        pose_3d: numpy array of shape (num_keypoints, 3) or (num_frames, num_keypoints, 3)
    """
    print(f"📂 Loading {npz_file}...")
    data = np.load(npz_file, allow_pickle=True)
    
    # 列出所有可用的键
    print(f"✓ Available keys in NPZ file: {list(data.keys())}")
    
    if key is None:
        # 尝试常见的键名
        possible_keys = ['reconstruction', 'pose_3d', 'keypoints', 'positions_3d', 'data']
        for k in possible_keys:
            if k in data.keys():
                key = k
                print(f"📌 Auto-selected key: '{key}'")
                break
        
        if key is None:
            # 使用第一个键
            key = list(data.keys())[0]
            print(f"📌 Using first available key: '{key}'")
    
    pose_data = data[key]
    print(f"✓ Loaded data shape: {pose_data.shape}")
    
    return pose_data, key


def create_interactive_3d_pose(pose_3d, keypoint_names=None, title="3D Pose Viewer", 
                               show_skeleton=True, point_size=8):
    """
    创建交互式 3D 姿态可视化
    
    Args:
        pose_3d: numpy array (num_keypoints, 3) with [x, y, z] coordinates
        keypoint_names: list of keypoint names (optional)
        title: title for the plot
        show_skeleton: whether to show skeleton connections
        point_size: size of the keypoints
    
    Returns:
        plotly figure object
    """
    # 确保是 2D 数组 (num_keypoints, 3)
    if pose_3d.ndim == 3:
        print(f"⚠️  Input has {pose_3d.shape[0]} frames. Visualizing first frame only.")
        pose_3d = pose_3d[0]
    
    # 创建关键点名称（如果未提供）
    if keypoint_names is None:
        keypoint_names = [f'Joint_{i}' for i in range(len(pose_3d))]
    
    # 创建 Plotly figure
    fig = go.Figure()
    
    # 创建悬停文本
    hover_text = []
    for i, (name, (x, y, z)) in enumerate(zip(keypoint_names, pose_3d)):
        hover_text.append(
            f"<b>{name}</b><br>" +
            f"X: {x:.3f}<br>" +
            f"Y: {y:.3f}<br>" +
            f"Z: {z:.3f}<br>" +
            f"Index: {i}"
        )
    
    # 添加关键点
    fig.add_trace(go.Scatter3d(
        x=pose_3d[:, 0],
        y=pose_3d[:, 1], 
        z=pose_3d[:, 2],
        mode='markers+text',
        marker=dict(
            size=point_size,
            color='red',
            opacity=0.8,
        ),
        text=[f"{i}" for i in range(len(pose_3d))],  # 显示索引号
        textposition="top center",
        textfont=dict(size=8, color='black'),
        hovertext=hover_text,
        hoverinfo='text',
        name='Keypoints'
    ))
    
    # 添加骨架连接
    if show_skeleton:
        # 使用自定义骨架连接 (26个关键点)
        I = np.array([24, 24, 1, 0, 24, 2, 2, 24, 18, 18, 12, 13, 8, 9, 14, 15, 18, 7, 7, 10, 11, 16, 17, 7, 25])
        J = np.array([0, 1, 21, 20, 2, 22, 23, 18, 12, 13, 8, 9, 14, 15, 3, 4, 7, 10, 11, 16, 17, 5, 6, 25, 19])
        
        # 创建骨架连接列表
        connections = list(zip(I, J))
        
        # 绘制骨架
        for joint1, joint2 in connections:
            if joint1 < len(pose_3d) and joint2 < len(pose_3d):
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
    
    # 计算数据范围以设置合适的坐标轴
    x_range = [pose_3d[:, 0].min(), pose_3d[:, 0].max()]
    y_range = [pose_3d[:, 1].min(), pose_3d[:, 1].max()]
    z_range = [pose_3d[:, 2].min(), pose_3d[:, 2].max()]
    
    # 添加一些边距
    x_margin = (x_range[1] - x_range[0]) * 0.1 or 0.1
    y_margin = (y_range[1] - y_range[0]) * 0.1 or 0.1
    z_margin = (z_range[1] - z_range[0]) * 0.1 or 0.1
    
    print(f"📊 Data range: X:[{x_range[0]:.3f}, {x_range[1]:.3f}], "
          f"Y:[{y_range[0]:.3f}, {y_range[1]:.3f}], "
          f"Z:[{z_range[0]:.3f}, {z_range[1]:.3f}]")
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color='black')
        ),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y", 
            zaxis_title="Z",
            xaxis=dict(
                showgrid=True, 
                zeroline=True,
                range=[x_range[0] - x_margin, x_range[1] + x_margin]
            ),
            yaxis=dict(
                showgrid=True, 
                zeroline=True,
                range=[y_range[0] - y_margin, y_range[1] + y_margin]
            ),
            zaxis=dict(
                showgrid=True, 
                zeroline=True,
                range=[z_range[0] - z_margin, z_range[1] + z_margin]
            ),
            bgcolor='white',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # 设置视角
            ),
            aspectmode='cube'  # 使用立方体模式以保持清晰显示
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig


def visualize_multiple_frames(poses_3d, keypoint_names=None, frame_indices=None, 
                              output_html='multi_frame_poses.html'):
    """
    可视化多帧 3D 姿态
    
    Args:
        poses_3d: numpy array (num_frames, num_keypoints, 3)
        keypoint_names: list of keypoint names
        frame_indices: which frames to visualize (e.g., [0, 10, 20])
        output_html: output HTML file
    """
    if frame_indices is None:
        # 自动选择几帧
        num_frames = min(6, len(poses_3d))
        frame_indices = np.linspace(0, len(poses_3d)-1, num_frames, dtype=int)
    
    print(f"🎬 Visualizing {len(frame_indices)} frames: {frame_indices}")
    
    from plotly.subplots import make_subplots
    
    # 创建子图
    cols = min(3, len(frame_indices))
    rows = (len(frame_indices) + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        specs=[[{'type': 'scatter3d'} for _ in range(cols)] for _ in range(rows)],
        subplot_titles=[f"Frame {i}" for i in frame_indices],
    )
    
    for idx, frame_idx in enumerate(frame_indices):
        row = idx // cols + 1
        col = idx % cols + 1
        
        pose_3d = poses_3d[frame_idx]
        
        # 添加关键点
        fig.add_trace(
            go.Scatter3d(
                x=pose_3d[:, 0],
                y=pose_3d[:, 1],
                z=pose_3d[:, 2],
                mode='markers',
                marker=dict(size=5, color='red', opacity=0.8),
                name=f'Frame {frame_idx}',
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=400 * rows,
        width=1200,
        title_text="Multi-Frame 3D Pose Viewer",
    )
    
    # 保存
    pyo.plot(fig, filename=output_html, auto_open=False)
    print(f"✓ Multi-frame visualization saved to: {output_html}")


def main():
    parser = argparse.ArgumentParser(description='Load 3D pose from NPZ and create interactive HTML viewer')
    
    parser.add_argument('--npz_file', type=str, default='/home/xiaohang/FMpose_review/Ti_workspace/projects/FMPose_clean/demo/animals/predictions/000000119761_horse/pose3D/0000_3D.npz',
                       help='Path to .npz file containing 3D pose data')
    parser.add_argument('--key', type=str, default=None,
                       help='Key name in NPZ file (auto-detect if not specified)')
    parser.add_argument('--no_skeleton', action='store_true',
                       help='Disable skeleton connections')
    parser.add_argument('--point_size', type=int, default=8,
                       help='Size of keypoint markers')
    parser.add_argument('--multi_frame', action='store_true',
                       help='Visualize multiple frames if data is sequential')
    parser.add_argument('--frame_indices', type=int, nargs='+',
                       help='Specific frame indices to visualize (e.g., 0 10 20)')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.npz_file):
        print(f"❌ Error: File not found: {args.npz_file}")
        return
    
    # 生成输出 HTML 路径：与 npz 同目录且同名
    npz_dir = os.path.dirname(args.npz_file)
    npz_stem = os.path.splitext(os.path.basename(args.npz_file))[0]
    output_html = os.path.join(npz_dir, f"{npz_stem}.html")

    # 加载 NPZ 文件
    pose_data, key = load_pose_from_npz(args.npz_file, args.key)
    
    # 根据数据维度决定可视化方式
    if pose_data.ndim == 3 and args.multi_frame:
        # 多帧可视化
        visualize_multiple_frames(
            pose_data, 
            frame_indices=args.frame_indices,
            output_html=output_html
        )
    else:
        # 单帧可视化
        if pose_data.ndim == 3:
            print(f"ℹ️  Using first frame from {pose_data.shape[0]} frames")
            pose_data = pose_data[0]
        
        # 创建可视化
        fig = create_interactive_3d_pose(
            pose_data,
            title=f"3D Pose from {os.path.basename(args.npz_file)}",
            show_skeleton=not args.no_skeleton,
            point_size=args.point_size
        )
        
        # 保存为 HTML
        pyo.plot(fig, filename=output_html, auto_open=False)
        print(f"\n✓ Interactive visualization saved to: {output_html}")
    
    # 显示使用说明
    print(f"\n🌐 To view the interactive HTML:")
    print(f"  1. Open directly: {os.path.abspath(output_html)}")
    print(f"  2. Or start HTTP server: python -m http.server 8080")
    print(f"  3. Then open: http://localhost:8080/{os.path.basename(output_html)}")
    
    print("\n🎮 Interactive features:")
    print("  • Drag to rotate the 3D view")
    print("  • Scroll to zoom in/out")
    print("  • Hover over points to see coordinates")
    print("  • Use toolbar for additional controls")


if __name__ == "__main__":
    main()
