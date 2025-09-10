"""
3D Pose Estimation and Visualization API

This module provides a comprehensive API for 3D pose estimation and visualization
in multi-camera setups, designed for primate pose estimation systems.

Main API Functions:
- data_processing(): Main pipeline for processing pose data
- Pose3D(): Extract and process 3D pose coordinates
- Pose3D_Vis(): Visualize 3D poses in 3D space
- Pose2D_vis(): Visualize reprojected poses on 2D images

Key Computer Vision Concepts:
- Camera calibration (intrinsic/extrinsic parameters)
- 3D-to-2D projection using perspective geometry
- Lens distortion correction
- Multi-camera pose visualization
"""

import numpy as np
import cv2
import sys, os, getopt
import json
from numpy.linalg import inv
from scipy.io import loadmat
from collections import defaultdict
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# === GLOBAL CONFIGURATIONS ===
# Define skeletal connections for 13 body joints (primate pose model)
# Each tuple (i, j) represents a bone connecting joint i to joint j
JOINT_PAIRS = [(0,1),(1,2),(2,3),(3,4),(2,5),(5,6),(2,7),(7,8),(8,9),(7,10),(10,11),(7,12)]

# Joint names for better understanding (optional)
JOINT_NAMES = [
    'Head', 'Neck', 'Spine_Mid', 'Spine_Base', 'Tail_Base',
    'Left_Shoulder', 'Left_Elbow', 'Right_Shoulder', 'Right_Elbow', 'Right_Wrist',
    'Left_Hip', 'Left_Knee', 'Right_Hip'
]


# === DATA LOADING FUNCTIONS ===

def load_camera_parameters(batch_num):
    """
    Load camera intrinsic and extrinsic parameters from calibration files.
    
    Args:
        batch_num (str/int): Batch number to load camera parameters for
    
    Returns:
        dict: Dictionary containing camera parameters for each camera
              Format: {cam_id: {'K': matrix, 'd1': float, 'd2': float, 
                               'R': matrix, 'C': vector, 'P': matrix}}
    """
    cameras = {}
    
    # Load intrinsic parameters
    intrinsic_file = f'../dataset/Batch{batch_num}/intrinsic.txt'
    with open(intrinsic_file) as f:
        lines = f.readlines()
        
        # Parse intrinsic file format: every 5 lines define one camera
        for i in range(0, len(lines), 5):
            cam_line = lines[i]          # Camera ID line
            K_lines = lines[i + 1:i + 4] # 3x3 intrinsic matrix K
            ds = lines[i + 4].rstrip('\n')  # Distortion coefficients
            
            # Extract distortion parameters
            d = ds.split(' ')
            d1 = float(d[0])  # First radial distortion coefficient
            d2 = float(d[1])  # Second radial distortion coefficient
            
            # Extract camera ID
            cam = cam_line.strip().split(' ')[1]
            
            # Reconstruct 3x3 intrinsic matrix K from the file
            K = np.reshape(np.array([float(f) for K_line in K_lines for f in K_line.strip().split(' ')]), [3, 3])
            
            # Store camera intrinsic parameters
            cameras[cam] = {'K': K, 'd1': d1, 'd2': d2}
    
    # Load extrinsic parameters
    extrinsic_file = f'../dataset/Batch{batch_num}/camera.txt'
    with open(extrinsic_file) as f:
        lines = f.readlines()
        
        # Parse extrinsic file format: starting from line 3, every 5 lines define one camera
        for i in range(3, len(lines), 5):
            cam_line = lines[i]          # Camera ID line
            C_line = lines[i + 1]        # Camera center coordinates
            R_lines = lines[i + 2:i + 5] # 3x3 rotation matrix
            
            # Extract camera ID
            cam = cam_line.strip().split(' ')[1]
            
            if cam not in cameras:
                continue
                
            # Extract camera center (3D position)
            C = np.array([float(f) for f in C_line.strip().split(' ')])
            
            # Reconstruct 3x3 rotation matrix
            R = np.reshape(np.array([float(f) for R_line in R_lines for f in R_line.strip().split(' ')]), [3, 3])
            
            # Compute projection matrix P = K * [R | -RC]
            # This combines intrinsic and extrinsic parameters for direct 3D-to-2D projection
            P = cameras[cam]['K'] @ (R @ (np.concatenate((np.identity(3), -np.reshape(C, [3, 1])), axis=1)))
            
            # Store extrinsic parameters and computed projection matrix
            cameras[cam]['R'] = R
            cameras[cam]['C'] = C
            cameras[cam]['P'] = P
    
    return cameras


def load_annotations(batch_num):
    """
    Load 3D pose annotations and crop parameters from MATLAB files.
    
    Args:
        batch_num (str/int): Batch number to load annotations for
    
    Returns:
        tuple: (annotations, parameters) - MATLAB data structures
    """
    # Load 3D pose annotations from MATLAB file
    annotations = loadmat(f'../dataset/Batch{batch_num}/coords_3D.mat')
    
    # Load image cropping parameters from MATLAB file
    cropping_parameters = loadmat(f'../dataset/Batch{batch_num}/crop_para.mat')
    
    return annotations, cropping_parameters


def load_images(batch_num, frame_num, camera_ids, parameters):
    """
    Load images for specified frame and cameras.
    
    Args:
        batch_num (str/int): Batch number
        frame_num (int): Frame number to load
        camera_ids (list): List of camera IDs to load images for
        parameters: Crop parameters from MATLAB file
    
    Returns:
        dict: Dictionary mapping camera_id to loaded image
    """
    images = {}
    
    # Extract frame information from crop parameters
    pt = parameters['crop'].transpose()[0]
    u = np.unique(pt, axis=0)
    q = np.where(pt == u[frame_num])
    
    for i, cam_id in enumerate(camera_ids):
        if i * 2 >= len(q[0]):
            break
            
        frame = parameters['crop'][q[0][2*i]][0]
        cmr = parameters['crop'][q[0][2*i]][1]
        
        # Construct image filename
        img_name = f'../dataset/Images/batch{batch_num}_{str(frame).zfill(9)}_{cmr}.jpg'
        
        # Load image from file
        image = cv2.imread(img_name)
        if image is not None:
            images[str(cmr)] = image
        else:
            print(f"Warning: Could not load image {img_name}")
    
    return images


# === UTILITY FUNCTIONS ===

def distort_point(u_x, u_y, camera_params):
    """
    Apply lens distortion to projected 2D points.
    
    Args:
        u_x, u_y: Undistorted 2D pixel coordinates
        camera_params: Dictionary containing camera parameters
    
    Returns:
        numpy.array: Distorted 2D pixel coordinates [x, y]
    """
    K = camera_params['K']
    d1 = camera_params['d1']
    d2 = camera_params['d2']

    # Convert to normalized coordinates
    invK = inv(K)
    z = np.array([u_x, u_y, 1])
    nx = invK.dot(z)

    # Apply radial distortion model
    r_squared = nx[0] * nx[0] + nx[1] * nx[1]
    distortion_factor = 1 + d1 * r_squared + d2 * r_squared * r_squared
    
    x_dn = nx[0] * distortion_factor
    y_dn = nx[1] * distortion_factor

    # Convert back to pixel coordinates
    z2 = np.array([x_dn, y_dn, 1])
    x_d = K.dot(z2)

    return np.array([x_d[0], x_d[1]])


def project_3d_to_2d(coords_3d, camera_params):
    """
    Project 3D world coordinates to 2D image coordinates.
    
    Args:
        coords_3d: 3D coordinates in world space [x, y, z]
        camera_params: Dictionary containing camera parameters
    
    Returns:
        numpy.array: Final 2D pixel coordinates [x, y] with distortion applied
    """
    # Get projection matrix
    P = camera_params['P']
    
    # Project 3D point to 2D using homogeneous coordinates
    u = P @ np.append(coords_3d, [1])
    
    # Perspective division
    u = u[0:2] / u[2]
    
    # Apply lens distortion correction
    proj = distort_point(u[0], u[1], camera_params)
    return proj


def get_projection(cam_id, coords_3d, cameras_dict):
    """
    Project 3D world coordinates to 2D image coordinates (original function signature).
    
    This implements the full camera projection pipeline:
    3D World → Camera Coordinates → Image Plane → Distorted Image
    
    Args:
        cam: Camera identifier string
        coords_3d: 3D coordinates in world space [x, y, z]
        cameras_dict: Dictionary containing all camera parameters
    
    Returns:
        numpy.array: Final 2D pixel coordinates [x, y] with distortion applied
    """
    # Get projection matrix for this camera
    P = cameras_dict[cam_id]['P']
    
    # Project 3D point to 2D using homogeneous coordinates
    # Convert 3D point to homogeneous coordinates [x, y, z, 1]
    u = P @ np.append(coords_3d, [1])
    
    # Perspective division: convert from homogeneous to Cartesian coordinates
    u = u[0:2] / u[2]  # [x/z, y/z]
    
    # Apply lens distortion correction
    proj = distort_point(u[0], u[1], cameras_dict[cam_id])
    return proj


# === MAIN API FUNCTIONS ===

def extract_pose_data(batch_num, frame_num):
    """
    Extract 3D pose data for a specific frame.
    
    Args:
        batch_num (str/int): Batch number to process
        frame_num (int): Frame number to extract pose from
    
    Returns:
        dict: Processed 3D pose data containing:
              - 'joints_3d': 3D coordinates of all joints
              - 'joint_names': Names of joints
              - 'joint_pairs': Skeletal connections
              - 'frame_info': Frame metadata
    """
    print(f'Extracting 3D pose data for frame {frame_num} of batch {batch_num}')
    
    
    # Load camera parameters and images
    cameras = load_camera_parameters(batch_num)
    annotations, cropping_parameters = load_annotations(batch_num)
    
    # Extract 3D coordinates for the specified frame
    ii = frame_num * 13  # Starting index for this frame's joints (13 joints per frame)
    joints_3d = []
    
    for jt in range(13):
        coords = annotations['coords'][ii + jt, 1:4]  # [x, y, z] coordinates
        if coords is not None:
            joints_3d.append(coords)
        else:
            joints_3d.append(None)
    
    # Package the data
    pose_data = {
        'joints_3d': joints_3d,
        'cameras': cameras,
        'cropping_parameters': cropping_parameters,
        'joint_names': JOINT_NAMES,
        'joint_pairs': JOINT_PAIRS,
        'frame_info': {
            'batch_num': batch_num,
            'frame_num': frame_num,
            'total_joints': 13
        }
    }
    
    return pose_data


def Pose3D(batch_num, frame_num, output_dir='results', visualize_3d=True, visualize_2d=True, 
           rotation_angle=-90, rotation_axis='x'):
    """
    Main pipeline for processing 3D pose data with comprehensive visualization.
    
    This is the primary function that orchestrates the entire workflow:
    1. Extracts 3D pose data from annotations
    2. Creates 3D visualization in 3D space
    3. Creates 2D reprojection visualization on camera images
    4. Creates 2D skeleton visualization on white background (clean skeleton images)
    5. Optionally saves 2D pose coordinates without plotting on images
    6. Saves all results to organized output directory
    
    Args:
        batch_num (str/int): Batch number to process
        frame_num (int): Frame number to process
        output_dir (str): Base directory to save results
        visualize_3d (bool): Whether to create 3D visualization
        visualize_2d (bool): Whether to create 2D visualization
        rotation_angle (float): Rotation angle for 3D visualization (degrees, default: -90)
        rotation_axis (str): Rotation axis for 3D visualization ('x', 'y', or 'z', default: 'x')
    
    Returns:
        dict: Complete pose data and processing results
    """
    print(f"=== Starting 3D Pose Processing Pipeline ===")
    print(f"Processing Batch {batch_num}, Frame {frame_num}")
    
    # Extract pose data from annotations 
    pose_data = extract_pose_data(batch_num, frame_num)
    
    # Create specific output directory for this batch and frame
    specific_output_dir = os.path.join(output_dir, f'pose_batch{batch_num}_frame{frame_num}')
    os.makedirs(specific_output_dir, exist_ok=True)
    
    # Generate visualizations
    if visualize_3d:
        save_path_3d = os.path.join(specific_output_dir, f"pose_3d_batch{batch_num}_frame{frame_num}.png")
        Pose3D_Vis(pose_data, save_path=save_path_3d, show_plot=False, 
                   rotation_angle=rotation_angle, rotation_axis=rotation_axis)
    
    if visualize_2d:
        save_path_2d = os.path.join(specific_output_dir, f"pose_2d_batch{batch_num}_frame{frame_num}.png")
        Pose2D_vis(pose_data, batch_num, frame_num, save_path=save_path_2d, show_plot=False)
        
        # Also create 2D skeleton on white background
        save_path_white = os.path.join(specific_output_dir, f"skeleton_white_batch{batch_num}_frame{frame_num}.png")
        plot_2d_skeleton_on_white(pose_data, batch_num, frame_num, save_path=save_path_white)
    
    print(f"=== Processing Complete ===")
    print(f"Results saved to: {specific_output_dir}")
    
    return pose_data


def rotate_3d_points(points, angle_degrees, axis='z'):
    """
    Rotate 3D points around a specified axis.
    
    Args:
        points (np.array): Nx3 array of 3D points
        angle_degrees (float): Rotation angle in degrees
        axis (str): Rotation axis ('x', 'y', or 'z')
    
    Returns:
        np.array: Rotated 3D points
    """
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    if axis.lower() == 'x':
        # Rotation around X-axis
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])
    elif axis.lower() == 'y':
        # Rotation around Y-axis
        rotation_matrix = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
    elif axis.lower() == 'z':
        # Rotation around Z-axis
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    
    # Apply rotation to all points
    rotated_points = np.dot(points, rotation_matrix.T)
    return rotated_points


def Pose3D_Vis(pose_data, save_path=None, show_plot=True, rotation_angle=-90, rotation_axis='x'):
    """
    Visualize 3D poses in 3D space.
    
    Args:
        pose_data (dict): 3D pose data from Pose3D() function
        save_path (str, optional): Path to save the 3D visualization
        show_plot (bool): Whether to display the plot
        rotation_angle (float): Rotation angle in degrees (default: -90)
        rotation_axis (str): Axis to rotate around ('x', 'y', or 'z') (default: 'x')
    """
    joints_3d = pose_data['joints_3d']
    joint_pairs = pose_data['joint_pairs']
    joint_names = pose_data['joint_names']
    frame_info = pose_data['frame_info']
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot joints as points
    valid_joints = [(i, joint) for i, joint in enumerate(joints_3d) if joint is not None]
    
    if valid_joints:
        joint_indices, joint_coords = zip(*valid_joints)
        joint_coords = np.array(joint_coords)
        
        # Apply rotation transformation
        joint_coords = rotate_3d_points(joint_coords, rotation_angle, rotation_axis)
        
        # Plot joint points
        # ax.scatter(joint_coords[:, 0], joint_coords[:, 1], joint_coords[:, 2], 
        #           c='red', alpha=0.8, label='Joints')
        ax.scatter(joint_coords[:, 0], joint_coords[:, 1], joint_coords[:, 2], 
                  c='red', alpha=0.8, label='Joints')
        
        # Plot skeleton connections (using rotated coordinates)
        # First create a mapping from original indices to rotated coordinates
        rotated_joints = {}
        for i, (idx, _) in enumerate(valid_joints):
            rotated_joints[idx] = joint_coords[i]
        
        for joint1_idx, joint2_idx in joint_pairs:
            if (joint1_idx in rotated_joints and joint2_idx in rotated_joints):
                joint1 = rotated_joints[joint1_idx]
                joint2 = rotated_joints[joint2_idx]
                
                ax.plot([joint1[0], joint2[0]], 
                       [joint1[1], joint2[1]], 
                       [joint1[2], joint2[2]], 
                       'b-', linewidth=2, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'3D Pose Visualization (Rotated {rotation_angle}° around {rotation_axis.upper()}-axis)\nBatch {frame_info["batch_num"]}, Frame {frame_info["frame_num"]}')
    
    ax.legend()
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D visualization saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_2d_skeleton_on_white(pose_data, batch_num, frame_num, save_path=None):
    """
    Plot 2D skeleton on white background images (same layout as Pose2D_vis but with white background).
    
    Args:
        pose_data (dict): 3D pose data from Pose3D() function
        batch_num (str/int): Batch number
        frame_num (int): Frame number
        save_path (str, optional): Path to save the 2D skeleton visualization
    """
    cropping_parameters = pose_data['cropping_parameters']
    cameras = pose_data['cameras']
    
    # Extract frame information from crop parameters
    pt = cropping_parameters['crop'].transpose()[0]
    unique_frames = np.unique(pt, axis=0)
    q = np.where(pt == unique_frames[frame_num])
    
    print(f"Creating 2D skeleton on white background for frame {frame_num} (frame_id: {unique_frames[frame_num]})")
    
    # Process each camera (exactly like original Pose2D_vis)
    processed_images = {}
    
    for i in range(4):  # Process 4 cameras exactly like original
        if i * 2 >= len(q[0]):
            break
        
        # Extract camera ID from crop parameters
        index = 2*i
        frame = cropping_parameters['crop'][q[0][index]][0]  # Frame number
        cmr = cropping_parameters['crop'][q[0][index]][1]    # Camera number
        cam_id = str(cmr)  # Convert to string like original
        
        # Get crop parameters (exactly like original)
        crop_top = cropping_parameters['crop'][q[0][index]][2]
        crop_left = cropping_parameters['crop'][q[0][index]][3]
        crop_width = cropping_parameters['crop'][q[0][index]][4]
        crop_height = cropping_parameters['crop'][q[0][index]][5]
        
        # Create white background image with same dimensions as crop region
        image = np.ones((crop_height, crop_width, 3), dtype=np.uint8) * 255  # White background
        
        print(f"Processing camera {i} (cam_id: {cam_id}) - White background: {crop_width}x{crop_height}")
        
        # Reproject 3D joints to 2D (following original approach)
        reprojected_joints = {}
        
        for jt_idx, joint_3d in enumerate(pose_data['joints_3d']):
            if joint_3d is not None:
                # Project 3D joint to 2D image coordinates (exactly like original)
                x, y = get_projection(cam_id, joint_3d, cameras)
                
                # Apply additional distortion (matching original approach exactly)
                proj = distort_point(x, y, cameras[cam_id])
                
                # Store reprojected coordinates (note: y,x order matching original exactly)
                reprojected_joints[jt_idx] = {'reprojected': (int(y), int(x))}
            else:
                # Handle missing joint data exactly like original
                reprojected_joints[jt_idx] = {'reprojected': None}
                
        # Draw skeleton connections (following original approach exactly)
        for joint1_idx, joint2_idx in pose_data['joint_pairs']:
            if (joint1_idx in reprojected_joints and joint2_idx in reprojected_joints):
                # Get reprojected coordinates for both joints (exactly like original)
                if (reprojected_joints[joint1_idx]['reprojected'] is not None and 
                    reprojected_joints[joint2_idx]['reprojected'] is not None):
                    
                    coords1 = reprojected_joints[joint1_idx]['reprojected']
                    x, y = coords1
                    
                    # Adjust coordinates relative to crop region (exactly like original)
                    x = x - crop_left
                    y = y - crop_top
                    pt1 = (y, x)  # OpenCV format exactly like original
                    
                    coords2 = reprojected_joints[joint2_idx]['reprojected']
                    x, y = coords2
                    x = x - crop_left
                    y = y - crop_top
                    pt2 = (y, x)
                    
                    # Check boundaries exactly like original
                    if not (pt1[0] < 1 or pt1[1] < 1 or pt2[0] < 1 or pt2[1] < 1 or 
                            pt1[0] > crop_width or pt1[1] > crop_height or 
                            pt2[0] > crop_width or pt2[1] > crop_height):
                        cv2.line(image, pt1, pt2, [0, 0, 0], 3)  # Black lines exactly like original
        
        # Draw joint points (additional visualization, not in original)
        for jt_idx, joint_data in reprojected_joints.items():
            if joint_data['reprojected'] is not None:
                coords = joint_data['reprojected']
                x, y = coords
                x = x - crop_left
                y = y - crop_top
                pt = (y, x)
                # Use same boundary check as original
                if not (pt[0] < 1 or pt[1] < 1 or pt[0] > crop_width or pt[1] > crop_height):
                    cv2.circle(image, pt, 3, [0, 255, 0], -1)  # Green circles exactly like original
        
        processed_images[i] = image
    
    # Save individual camera images if save_path is provided
    if save_path:
        save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else save_path
        os.makedirs(save_dir, exist_ok=True)
        
        # Save individual camera images
        for i, image in processed_images.items():
            individual_image_path = os.path.join(save_dir, f'skeleton_white_camera_{i}_batch{batch_num}_frame{frame_num}.jpg')
            cv2.imwrite(individual_image_path, image)
            print(f"Skeleton on white camera {i} image saved to: {individual_image_path}")
    
    # Create combined visualization
    fig = plt.figure(figsize=(15, 10))
    
    for i, image in processed_images.items():
        plt.subplot(2, 2, i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f'Camera {i} - Skeleton on White Background')
        plt.axis('off')
    
    plt.suptitle(f'2D Skeleton on White Background\nBatch {batch_num}, Frame {frame_num}')
    plt.tight_layout()
    
    # Save combined visualization
    if save_path:
        combined_save_path = save_path if save_path.endswith('.png') else os.path.join(save_path, f'skeleton_white_combined_batch{batch_num}_frame{frame_num}.png')
        fig.savefig(combined_save_path, dpi=300, bbox_inches='tight')
        print(f"Combined skeleton on white background saved to: {combined_save_path}")
    
    plt.close(fig)


def Pose2D_vis(pose_data, batch_num, frame_num, save_path=None, show_plot=False):
    """
    Visualize reprojected 2D poses on camera images.
    
    Args:
        pose_data (dict): 3D pose data from Pose3D() function
        batch_num (str/int): Batch number
        frame_num (int): Frame number
        save_path (str, optional): Path to save the 2D visualization
        show_plot (bool): Whether to display the plot
    """

    # each row follow the format: [frame, camera, top, left, width, height]
    cropping_parameters = pose_data['cropping_parameters'] # [14150, 6]
    
    # print("cropping_parameters:", cropping_parameters['crop'].shape) 
    
    cameras = pose_data['cameras']
    
    # Extract frame information from crop parameters 
    # bug: the original processing code just select the frame with index equal to frame_num in the unique frame numbers
    # the frame_num is not the frame_id in the cropping_parameters
    
    pt = cropping_parameters['crop'].transpose()[0] # [6, 14150] -> [14150] 
    unique_frames = np.unique(pt, axis=0)
    print("frame_num:", frame_num)
    print("unique_frames:", unique_frames[frame_num]) #
    q = np.where(pt == unique_frames[frame_num])
    print("q:",q[0]) # q[0] means the indices of the rows with frame_id = unique_frames[frame_num]
    
    print("frame_id = cropping_parameters['crop'][q[0]]:", cropping_parameters['crop'][q[0]]) # here we get the row with frame_id = unique_frames[frame_num]
    
    # Process each camera (exactly like original)
    processed_images = {}
    
    for i in range(4):  # Process 4 cameras exactly like original
        if i * 2 >= len(q[0]):
            break
        
        # Extract camera ID from crop parameters
        index = 2*i
        frame = cropping_parameters['crop'][q[0][index]][0]  # Frame number
        cmr = cropping_parameters['crop'][q[0][index]][1]    # Camera number
        cam_id = str(cmr)  # Convert to string like original
        
        # Get crop parameters (exactly like original)
        crop_top = cropping_parameters['crop'][q[0][index]][2]
        crop_left = cropping_parameters['crop'][q[0][index]][3]
        crop_width = cropping_parameters['crop'][q[0][index]][4]
        crop_height = cropping_parameters['crop'][q[0][index]][5]
        
        # Construct image filename (exactly like original)
        img_name = f'../dataset/Images/batch{batch_num}_{str(frame).zfill(9)}_{cmr}.jpg'
        
        # Load image from file (exactly like original)
        image = cv2.imread(img_name)
        if image is None:
            print(f"Warning: Could not load image {img_name}")
            continue
        
        # Save original image without pose overlay if save_path is provided
        if save_path:
            save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else save_path
            os.makedirs(save_dir, exist_ok=True)
            original_image_path = os.path.join(save_dir, f'original_camera_{i}_batch{batch_num}_frame{frame_num}.jpg')
            cv2.imwrite(original_image_path, image)
            print(f"Original camera {i} image saved to: {original_image_path}")
        
        # Reproject 3D joints to 2D (following original approach)
        reprojected_joints = {}
        
        for jt_idx, joint_3d in enumerate(pose_data['joints_3d']):
            if joint_3d is not None:
                # Project 3D joint to 2D image coordinates (exactly like original)
                x, y = get_projection(cam_id, joint_3d, cameras)
                
                # Apply additional distortion (matching original approach exactly)
                proj = distort_point(x, y, cameras[cam_id])
                
                # Store reprojected coordinates (note: y,x order matching original exactly)
                reprojected_joints[jt_idx] = {'reprojected': (int(y), int(x))}
            else:
                # Handle missing joint data exactly like original
                reprojected_joints[jt_idx] = {'reprojected': None}
                
        # Draw skeleton connections (following original approach exactly)
        for joint1_idx, joint2_idx in pose_data['joint_pairs']:
            if (joint1_idx in reprojected_joints and joint2_idx in reprojected_joints):
                # Get reprojected coordinates for both joints (exactly like original)
                if (reprojected_joints[joint1_idx]['reprojected'] is not None and 
                    reprojected_joints[joint2_idx]['reprojected'] is not None):
                    
                    coords1 = reprojected_joints[joint1_idx]['reprojected']
                    x, y = coords1
                    
                    # Adjust coordinates relative to crop region (exactly like original)
                    x = x - crop_left
                    y = y - crop_top
                    pt1 = (y, x)  # OpenCV format exactly like original
                    
                    coords2 = reprojected_joints[joint2_idx]['reprojected']
                    x, y = coords2
                    x = x - crop_left
                    y = y - crop_top
                    pt2 = (y, x)
                    
                    # Check boundaries exactly like original
                    if not (pt1[0] < 1 or pt1[1] < 1 or pt2[0] < 1 or pt2[1] < 1 or 
                            pt1[0] > crop_width or pt1[1] > crop_height or 
                            pt2[0] > crop_width or pt2[1] > crop_height):
                        cv2.line(image, pt1, pt2, [0, 0, 0], 3)  # Black lines exactly like original
        
        # Draw joint points (additional visualization, not in original)
        for jt_idx, joint_data in reprojected_joints.items():
            coords = joint_data['reprojected']
            x, y = coords
            x = x - crop_left
            y = y - crop_top
            pt = (y, x)
            # Use same boundary check as original
            if not (pt[0] < 1 or pt[1] < 1 or pt[0] > crop_width or pt[1] > crop_height):
                cv2.circle(image, pt, 3, [0, 255, 0], -1)  # Small green circles
        
        processed_images[i] = image
    
    # Save individual camera images if save_path is provided
    if save_path:
        save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else save_path
        os.makedirs(save_dir, exist_ok=True)
        
        # Save individual camera images
        for i, image in processed_images.items():
            individual_image_path = os.path.join(save_dir, f'camera_{i}_batch{batch_num}_frame{frame_num}.jpg')
            cv2.imwrite(individual_image_path, image)
            print(f"Camera {i} image saved to: {individual_image_path}")
    
    # Create combined visualization
    fig = plt.figure(figsize=(15, 10))
    
    print(len(processed_images))
    for i, image in processed_images.items():
        plt.subplot(2, 2, i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f'Camera {i}')
        plt.axis('off')
    
    plt.suptitle(f'2D Pose Reprojection\nBatch {batch_num}, Frame {frame_num}')
    plt.tight_layout()
    
    # Save combined visualization or show the plot
    if save_path:
        combined_save_path = save_path if save_path.endswith('.png') else os.path.join(save_path, f'pose_2d_combined_batch{batch_num}_frame{frame_num}.png')
        fig.savefig(combined_save_path, dpi=300, bbox_inches='tight')
        print(f"Combined 2D visualization saved to: {combined_save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def data_processing(batch_num, frame_num, output_dir='results', visualize_3d=True, visualize_2d=True,
                   rotation_angle=-90, rotation_axis='x'):
    """
    Wrapper function for the main Pose3D() pipeline.
    
    This function simply calls Pose3D() which contains the main processing logic.
    
    Args:
        batch_num (str/int): Batch number to process
        frame_num (int): Frame number to process
        output_dir (str): Base directory to save results
        visualize_3d (bool): Whether to create 3D visualization
        visualize_2d (bool): Whether to create 2D visualization
        rotation_angle (float): Rotation angle for 3D visualization (degrees, default: -90)
        rotation_axis (str): Rotation axis for 3D visualization ('x', 'y', or 'z', default: 'x')
    
    Returns:
        dict: Complete pose data and processing results
    """
    # Call the main Pose3D() function which now contains the full pipeline
    return Pose3D(batch_num, frame_num, output_dir, visualize_3d, visualize_2d, 
                  rotation_angle, rotation_axis)


# === LEGACY FUNCTION (for backward compatibility) ===

def display_plot(I, image_name):
    """
    Legacy function: Save reprojection results from 4 cameras in a 2x2 grid to file.
    
    Args:
        I: Dictionary containing processed images from 4 cameras
        image_name: Output filename to save the visualization
    """
    fig = plt.figure()
    
    # Create 2x2 subplot layout for 4 camera views
    sub1 = plt.subplot(2, 2, 1)
    sub1.set_xticks(())  # Remove axis ticks for cleaner display
    sub1.set_yticks(())
    sub1.imshow(I[0])

    sub2 = plt.subplot(2, 2, 2)
    sub2.set_xticks(())
    sub2.set_yticks(())
    sub2.imshow(I[1])

    sub3 = plt.subplot(2, 2, 3)
    sub3.set_xticks(())
    sub3.set_yticks(())
    sub3.imshow(I[2])

    sub4 = plt.subplot(2, 2, 4)
    sub4.set_xticks(())
    sub4.set_yticks(())
    sub4.imshow(I[3])

    fig.tight_layout()
    # plt.show()
    fig.savefig(image_name)  # Save figure to file


# === COMMAND LINE INTERFACE ===

def main():
    """Command line interface for the pose estimation API."""
    print("=== 3D Pose Estimation API ===")
    print("Main function: Pose3D()")
    
    # Default parameters
    frm = 1  # Default frame number
    btch = 7  # Default batch number
    
    # Parse command line arguments
    full_cmd_arguments = sys.argv
    argument_list = full_cmd_arguments[1:]
    short_options = "hb:f:"
    long_options = ["help", "batch=", "frame="]
    
    try:
        arguments, values = getopt.getopt(argument_list, short_options, long_options)
    except getopt.error as err:
        print(str(err))
        sys.exit(2)
    
    # Process command line arguments
    for current_argument, current_value in arguments:
        if current_argument in ("-h", "--help"):
            print("\n=== 3D Pose Estimation API ===")
            print("Main function: Pose3D(batch_num, frame_num)")
            print("-b or --batch ==> Select batch number (default value is 7)")
            print("-f or --frame ==> Select frame number (default value is 1)")
            print("\nAPI Functions:")
            print("- Pose3D(batch_num, frame_num) - Main processing pipeline")
            print("- Pose3D_Vis(pose_data) - Create 3D visualization")
            print("- Pose2D_vis(pose_data, batch_num, frame_num) - Create 2D visualization")
            print("- plot_2d_skeleton_on_white(pose_data, batch_num, frame_num) - Create 2D skeleton on white background")
            print("- extract_pose_data(batch_num, frame_num) - Extract raw 3D pose data")
            print("- data_processing(batch_num, frame_num) - Wrapper for Pose3D()")
            print("\nExample usage:")
            print("python reprojection.py -b 7 -f 1")
            print("or")
            print("from reprojection import Pose3D")
            print("Pose3D(batch_num=7, frame_num=1)")
            sys.exit(0)
        elif current_argument in ("-b", "--batch"):
            btch = str(current_value)
        elif current_argument in ("-f", "--frame"):
            frm = int(current_value)
    
    # Run the main Pose3D pipeline
    print(f"Running Pose3D(batch_num={btch}, frame_num={frm})")
    Pose3D(btch, frm, output_dir='results', visualize_3d=True, visualize_2d=True)


if __name__ == '__main__':
    main()