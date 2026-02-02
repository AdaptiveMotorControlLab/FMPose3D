"""
FMPose3D: monocular 3D Pose Estimation via Flow Matching

Official implementation of the paper:
"FMPose3D: monocular 3D Pose Estimation via Flow Matching"
by Ti Wang, Xiaohang Yu, and Mackenzie Weygandt Mathis
Licensed under Apache 2.0
"""

# SuperAnimal Demo: https://github.com/DeepLabCut/DeepLabCut/blob/main/examples/COLAB/COLAB_YOURDATA_SuperAnimal.ipynb
import sys
import os
import numpy as np
import glob
from tqdm import tqdm
import cv2
import torch
import copy
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import matplotlib.gridspec as gridspec
import imageio
from fmpose.animals.common.arguments import opts as parse_args
from fmpose.common.camera import normalize_screen_coordinates, camera_to_world

sys.path.append(os.getcwd())

args = parse_args().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

CFM = None

if getattr(args, "model_path", ""):
    # Load model from local file path (for custom models)
    import importlib.util
    import pathlib

    model_abspath = os.path.abspath(args.model_path)
    module_name = pathlib.Path(model_abspath).stem
    spec = importlib.util.spec_from_file_location(module_name, model_abspath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    CFM = getattr(module, "Model")
else:
    # Load model from installed fmpose package
    from fmpose.models import Model as CFM

from deeplabcut.pose_estimation_pytorch.apis import superanimal_analyze_images

superanimal_name = "superanimal_quadruped"
model_name = "hrnet_w32"
detector_name = "fasterrcnn_resnet50_fpn_v2"
max_individuals = 1

def compute_limb_regularization_matrix(gt_3d):
    """
    Compute regularization matrix to align limb directions to vertical (0,0,1).
    
    Args:
        gt_3d: numpy array of shape (J, 3) - 3D pose
        
    Returns:
        R: 3x3 rotation matrix to align limbs to vertical
    """
    # Define limb connection pairs (start, end)
    limb_connections = [
        (8, 14),   # left_front_thigh → left_front_knee
        (9, 15),   # right_front_thigh → right_front_knee
        (10, 16),  # left_back_thigh → left_back_knee
        (11, 17),  # right_back_thigh → right_back_knee
    ]
    
    # Compute direction vectors for all connections.
    # These connections go from proximal (thigh/knee) to distal (paw), so they
    # point downward; we reverse them to point upward.
    limb_vectors = []
    for start_idx, end_idx in limb_connections:
        # Reverse direction: end -> start (from paw toward body, upward).
        vec = gt_3d[start_idx] - gt_3d[end_idx]
        # Normalize.
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 1e-6:  # Avoid division by zero.
            vec = vec / vec_norm
            limb_vectors.append(vec)
    
    if len(limb_vectors) == 0:
        return np.eye(3)  # No valid vectors, return identity.
    
    # Compute average direction.
    avg_direction = np.mean(limb_vectors, axis=0)
    avg_direction = avg_direction / (np.linalg.norm(avg_direction) + 1e-8)
    
    # Target direction: vertical up (0, 0, 1).
    target_direction = np.array([0.0, 0.0, 1.0])
    
    # Compute rotation matrix to align avg_direction to target_direction.
    # Use Rodrigues' rotation formula.
    v = np.cross(avg_direction, target_direction)
    c = np.dot(avg_direction, target_direction)
    
    # If the two vectors are already aligned or opposite.
    if np.linalg.norm(v) < 1e-6:
        if c > 0:
            return np.eye(3)  # Already aligned.
        else:
            # Opposite direction, rotate 180 degrees.
            # Choose a perpendicular axis.
            if abs(avg_direction[0]) < 0.9:
                axis = np.array([1.0, 0.0, 0.0])
            else:
                axis = np.array([0.0, 1.0, 0.0])
            axis = axis - avg_direction * np.dot(axis, avg_direction)
            axis = axis / np.linalg.norm(axis)
            return 2 * np.outer(axis, axis) - np.eye(3)
    
    # Rodrigues rotation formula.
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    
    R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    
    return R

def apply_regularization(pose_3d, R):
    """
    Apply regularization matrix to 3D pose.
    
    Args:
        pose_3d: numpy array of shape (J, 3)
        R: 3x3 rotation matrix
        
    Returns:
        regularized pose_3d: numpy array of shape (J, 3)
    """
    return (R @ pose_3d.T).T

def get_pose2D(path, output_dir, type):

    print('\nGenerating 2D pose...')
    
    # Check if this is the special debug case for 000000119761_horse
    filename = Path(path).stem
    is_debug_case = "000000119761_horse" in filename
    
    if is_debug_case:
        print(f"DEBUG MODE: Using provided 2D pose for {filename}")
        # User provided 2D pose (26 keypoints, x, y coordinates, ignoring the last dimension)
        provided_pose = np.array([
            [361, 230], [361, 237], [363, 279], [257, 359], [251, 374],
            [164, 365], [68, 372], [99, 206], [247, 266], [253, 285],
            [127, 275], [101, 285], [267, 217], [268, 229], [273, 318],
            [250, 340], [128, 311], [76, 305], [313, 220], [48, 310],
            [351, 203], [352, 210], [340, 257], [340, 261], [373, 276],
            [55, 247]
        ], dtype=np.float32)
        
        # Reshape to match expected format: (1, 26, 2) for single individual
        provided_pose = provided_pose.reshape(1, 26, 2)
        
        # Create xy_preds dict with the provided pose
        xy_preds = {path: provided_pose}
        print(f"Using provided 2D pose with shape: {provided_pose.shape}")
    else:
        # Normal prediction flow
        predictions = superanimal_analyze_images(
            superanimal_name,
            model_name,
            detector_name,
            path,
            max_individuals,
            out_folder=output_dir
        )
        print("predictions:", predictions)
        
        # get the 2D keypoints from the predictions
        xy_preds = {}
        # predictions is a dict: {image_path: {"bodyparts": (N, K, 3), "bboxes": ..., "bbox_scores": ...}}
        for img_path, payload in predictions.items():
            bodyparts = payload.get("bodyparts")
            if bodyparts is None:
                continue
            # bodyparts shape: (num_individuals, num_keypoints, 3) -> [:, :, :2] keeps x,y
            xy_preds[img_path] = bodyparts[..., :2]

    print("2D keypoints (x,y) by image:")
    for img_path, xy in xy_preds.items():
        print(f"{img_path}: shape {xy.shape}")
    
    # For debug case, the provided pose is already in Animal3D format (26 keypoints)
    # So we skip the mapping step
    if is_debug_case:
        print("DEBUG MODE: Skipping keypoint mapping (already in Animal3D format)")
        mapped_keypoints = xy_preds
    else:
        # now map the keypoints to a different set of keypoints (used in Animal3D)
        # keypoint mapping from quadruped80K super keypotints to animal3d keypoints
        keypoint_mapping = {"quadruped80k":[10, 5, -1, 26, 29, 30, 35, 22, 24, 27, 31, 32, -1, -1, 25, 28, 33, 34, 15, 23, 11, 6, 4, 3, 0, -1]}
        
        # for the keypoint_mapping, -1 indicates that there is no corresponding keypoint in the source set, but we can interpolate 
        # for index 2, we can interpolate between keypoints 3 and 4 in the source set to get a better estimate of the missing keypoint
        # for index 25, we can interpolate between keypoints 22 and 23 in the source set
        # for index 12, we can interpolate between keypoints 24 and 19 in the source set
        # for index 13, we can interpolate between keypoints 27 and 19 in the source set
        
        # Define interpolation rules for -1 indices: {target_idx: (source_idx1, source_idx2)}
        interpolation_rules = {
            2: (3, 4),      # interpolate between source keypoints 3 and 4
            12: (24, 19),   # interpolate between source keypoints 24 and 19
            13: (27, 19),   # interpolate between source keypoints 27 and 19
            25: (22, 23),   # interpolate between source keypoints 22 and 23
        }
        
        # map the keypoints
        mapped_keypoints = {}
        mapping_indices = keypoint_mapping["quadruped80k"]

        for img_path, xy in xy_preds.items():
            # xy shape: (num_individuals, num_keypoints, 2)
            num_individuals, num_keypoints, _ = xy.shape
            num_target_keypoints = len(mapping_indices)
            
            # Initialize mapped array with NaN or zeros
            mapped_xy = np.full((num_individuals, num_target_keypoints, 2), np.nan)
            
            for target_idx, source_idx in enumerate(mapping_indices):
                if source_idx != -1 and source_idx < num_keypoints:
                    # Copy the keypoint from source to target position
                    mapped_xy[:, target_idx, :] = xy[:, source_idx, :]
                elif source_idx == -1 and target_idx in interpolation_rules:
                    # Perform interpolation for -1 indices
                    src1, src2 = interpolation_rules[target_idx]
                    if src1 < num_keypoints and src2 < num_keypoints:
                        # Interpolate as the average of the two source keypoints
                        mapped_xy[:, target_idx, :] = (xy[:, src1, :] + xy[:, src2, :]) / 2.0
                        print(f"Interpolated keypoint {target_idx} from source keypoints {src1} and {src2}")
            
            mapped_keypoints[img_path] = mapped_xy
            print(f"Mapped {img_path}: {xy.shape} -> {mapped_xy.shape}")

    print('Generating 2D pose successful!')

    # Save mapped keypoints for later use in get_pose3D
    output_dir_2D = output_dir + 'input_2D/'
    os.makedirs(output_dir_2D, exist_ok=True)

    # Get the first (and likely only) mapped keypoints
    for img_path, mapped_xy in mapped_keypoints.items():
        # Save in the same format as vis_in_the_wild.py for compatibility
        output_npz = output_dir_2D + 'keypoints.npz'
        np.savez_compressed(output_npz, reconstruction=mapped_xy)
        print(f"Saved keypoints to {output_npz}")
        
        # Also save as npy for backup
        img_name = Path(img_path).stem
        output_file = Path(output_dir_2D) / f"{img_name}_mapped_keypoints.npy"
        np.save(output_file, mapped_xy)
        
        # Optionally save as CSV for readability
        for ind_idx in range(mapped_xy.shape[0]):
            csv_file = Path(output_dir_2D) / f"{img_name}_individual_{ind_idx}_mapped_keypoints.csv"
            df = pd.DataFrame(
                mapped_xy[ind_idx],
                columns=['x', 'y'],
                index=[f'keypoint_{i}' for i in range(mapped_xy.shape[1])]
            )
            df.to_csv(csv_file)
            print(f"Saved individual {ind_idx} keypoints to {csv_file}")
        
        # Visualize mapped keypoints on image
        img = Image.open(img_path)
        img_array = np.array(img)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img_array)
        
        # Define colors for different individuals
        colors = plt.cm.rainbow(np.linspace(0, 1, mapped_xy.shape[0]))
        
        for ind_idx in range(mapped_xy.shape[0]):
            keypoints = mapped_xy[ind_idx]
            
            # Plot keypoints
            valid_mask = ~np.isnan(keypoints[:, 0])
            if np.any(valid_mask):
                ax.scatter(
                    keypoints[valid_mask, 0],
                    keypoints[valid_mask, 1],
                    c=[colors[ind_idx]],
                    s=100,
                    alpha=0.8,
                    edgecolors='white',
                    linewidths=2,
                    label=f'Individual {ind_idx}'
                )
                
                # Annotate keypoint indices
                for kp_idx in np.where(valid_mask)[0]:
                    ax.annotate(
                        str(kp_idx),
                        (keypoints[kp_idx, 0], keypoints[kp_idx, 1]),
                        fontsize=8,
                        color='white',
                        weight='bold',
                        ha='center',
                        va='center'
                    )
        
        ax.set_title(f'Mapped Keypoints for {img_name}', fontsize=14)
        ax.axis('off')
        if mapped_xy.shape[0] > 1:
            ax.legend(loc='upper right')
        
        # Save visualization to pose2D folder (following vis_in_the_wild pattern)
        output_dir_pose2D = output_dir + 'pose2D/'
        os.makedirs(output_dir_pose2D, exist_ok=True)
        vis_file = Path(output_dir_pose2D) / f"{img_name}_2D.png"
        plt.tight_layout()
        plt.savefig(vis_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved visualization to {vis_file}")


def get_pose3D(path, output_dir, type='image'):
    """
    Generate 3D pose from 2D keypoints using the model.
    This function reads the 2D keypoints saved by get_pose2D and generates 3D poses.
    """
    print('\nGenerating 3D pose...')
    print(f"args.n_joints: {args.n_joints}, args.out_joints: {args.out_joints}")
    
    ## Reload model
    model = {}
    model['CFM'] = CFM(args).cuda()
    
    model_dict = model['CFM'].state_dict()
    model_path = args.saved_model_path
    print(f"Loading model from: {model_path}")
    pre_dict = torch.load(model_path)
    for name, key in model_dict.items():
        model_dict[name] = pre_dict[name]
    model['CFM'].load_state_dict(model_dict)
    print("Model loaded successfully!")
    
    model = model['CFM'].eval()

    ## Load input 2D keypoints
    keypoints = np.load(output_dir + 'input_2D/keypoints.npz', allow_pickle=True)['reconstruction']
    print(f"Loaded keypoints shape: {keypoints.shape}")

    ## Generate 3D poses
    if type == "image":
        i = 0
        img = cv2.imread(path)
        get_3D_pose_from_image(args, keypoints, i, img, model, output_dir)
    elif type == "video":
        cap = cv2.VideoCapture(path)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in tqdm(range(video_length)):
            ret, img = cap.read()
            if not ret:
                break
            get_3D_pose_from_image(args, keypoints, i, img, model, output_dir)
        cap.release()

    print('Generating 3D pose successful!')


def get_3D_pose_from_image(args, keypoints, i, img, model, output_dir):
    """
    Generate 3D pose for a single image frame.
    Adapted from vis_in_the_wild.py for animal pose estimation.
    """
    img_size = img.shape
    
    ## Input frames
    if args.type == 'image':
        input_2D_no = keypoints[i] if keypoints.ndim > 2 else keypoints
        if input_2D_no.ndim == 2:
            input_2D_no = np.expand_dims(input_2D_no, axis=0)
    else:
        input_2D_no = keypoints[0][i]
        input_2D_no = np.expand_dims(input_2D_no, axis=0)
    
    # Save original 2D coordinates for visualization (before normalization)
    input_2D_original = input_2D_no.copy()
    
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]

    input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])

    # Remove TTA (Test-Time Augmentation) for better consistency
    # Ensure input_2D has shape (1, J, 2) before conversion
    if input_2D.ndim == 2:  # (J, 2)
        input_2D = np.expand_dims(input_2D, axis=0)  # (1, J, 2)
    
    # Convert to tensor format matching visualize_animal_poses.py
    input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()  # (1, J, 2)
    input_2D = input_2D.unsqueeze(0)  # (1, 1, J, 2)

    # Euler sampler for CFM
    def euler_sample(c_2d, y_local, steps, model_3d):
        dt = 1.0 / steps
        for s in range(steps):
            t_s = torch.full((c_2d.size(0), 1, 1, 1), s * dt, device=c_2d.device, dtype=c_2d.dtype)
            v_s = model_3d(c_2d, y_local, t_s)
            y_local = y_local + dt * v_s
        return y_local
    
    ## Estimation (without TTA for better results)
    print("input_2D.shape:", input_2D.shape)
    print("input_2D:", input_2D[0, 0])
    
    # Single inference without flip augmentation
    # Create 3D random noise with shape (1, 1, J, 3)
    y = torch.randn(input_2D.size(0), input_2D.size(1), input_2D.size(2), 3).cuda()
    output_3D = euler_sample(input_2D, y, steps=args.sample_steps, model_3d=model)
    
    output_3D = output_3D[0:, args.pad].unsqueeze(1)
    # output_3D[:, :, 0, :] = 0
    post_out = output_3D[0, 0].cpu().detach().numpy()

    # rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
    # rot = np.array(rot, dtype='float32')
    # post_out = camera_to_world(post_out, R=rot, t=0)
    # post_out[:, 2] -= np.min(post_out[:, 2])

    input_2D_no = input_2D_no[args.pad]

    # Apply limb regularization
    R_reg = compute_limb_regularization_matrix(post_out)
    post_out = apply_regularization(post_out, R_reg)
    
    # Print debug info for first frame
    # if i == 0:
    #     print("\n=== Regularization Debug Info ===")
    #     print(f"Rotation matrix R:\n{R_reg}")
        
    #     # Validate average limb direction after regularization.
    #     limb_connections = [
    #         (8, 14), (9, 15), (10, 16), (11, 17)
    #     ]
    #     limb_vectors_after = []
    #     for start_idx, end_idx in limb_connections:
    #         vec = post_out[start_idx] - post_out[end_idx]
    #         vec_norm = np.linalg.norm(vec)
    #         if vec_norm > 1e-6:
    #             vec = vec / vec_norm
    #             limb_vectors_after.append(vec)
        
    #     if len(limb_vectors_after) > 0:
    #         avg_dir_after = np.mean(limb_vectors_after, axis=0)
    #         avg_dir_after = avg_dir_after / (np.linalg.norm(avg_dir_after) + 1e-8)
    #         print(f"Average limb direction after regularization: {avg_dir_after}")
    #         print(f"Target direction: [0, 0, 1]")
    #         print(f"Alignment (dot product): {np.dot(avg_dir_after, np.array([0, 0, 1])):.4f}")
    #     print("================================\n")

    ## Save 2D pose on image (similar to visualize_animal_poses.py)
    if img is not None:
        height, width = img_size[0], img_size[1]
        img_copy = img.copy()

        # Use original 2D coordinates (before normalization)
        vals = input_2D_original[args.pad] if input_2D_original.shape[0] > args.pad else input_2D_original[0]
        vals = np.reshape(vals, (26, 2))
        
        # Animal skeleton connections (26 joints)
        I = np.array([24, 24, 1, 0, 24, 2, 2, 24, 18, 18, 12, 13, 8, 9, 14, 15, 18, 7, 7, 10, 11, 16, 17, 7, 25])
        J = np.array([0, 1, 21, 20, 2, 22, 23, 18, 12, 13, 8, 9, 14, 15, 3, 4, 7, 10, 11, 16, 17, 5, 6, 25, 19])
        
        for j in np.arange(len(I)):
            pt1 = (int(vals[I[j], 0]), int(vals[I[j], 1]))
            pt2 = (int(vals[J[j], 0]), int(vals[J[j], 1]))
            cv2.line(img_copy, pt1, pt2, (240, 176, 0), 2, cv2.LINE_AA)  # Anti-aliasing
        
        # Save 2D image directly using cv2
        output_dir_2D_img = output_dir + 'pose2D_on_image/'
        os.makedirs(output_dir_2D_img, exist_ok=True)
        cv2.imwrite(f'{output_dir_2D_img}{i:04d}_2d.png', img_copy)
        print(f"Saved 2D pose on image to {output_dir_2D_img}{i:04d}_2d.png")

    ## Save 3D pose as npz
    output_dir_3D = output_dir + 'pose3D/'
    os.makedirs(output_dir_3D, exist_ok=True)
    
    # Save 3D pose data
    npz_filename = output_dir_3D + str(('%04d' % i)) + '_3D.npz'
    np.savez_compressed(npz_filename, pose_3d=post_out)
    print(f"Saved 3D pose to {npz_filename}")
    
    ## Save 3D visualization
    fig = plt.figure(figsize=(9.6, 5.4))
    gs = gridspec.GridSpec(1, 1)
    gs.update(wspace=-0.00, hspace=0.05)
    ax = plt.subplot(gs[0], projection='3d')
    show3Dpose(post_out, ax, color=(0/255, 176/255, 240/255), world=True, linewidth=2.5)

    plt.savefig(output_dir_3D + str(('%04d' % i)) + '_3D.png', dpi=200, format='png', bbox_inches='tight')
    # Save raw 3D pose alongside the image
    pose_npz = Path(output_dir_3D) / f"{i:04d}_3D.npz"
    np.savez_compressed(pose_npz, pose3d=post_out)
    plt.close(fig)


def show3Dpose(vals, ax, color=(0/255, 176/255, 240/255), world=True, linewidth=2.5):
    """
    Visualize 3D pose skeleton for 26-joint animal poses.
    Adapted from visualize_animal_poses.py
    """
    # Reshape to (26, 3) if needed
    if vals.ndim == 1:
        vals = np.reshape(vals, (26, 3))
    
    # Animal skeleton connections (26 joints)
    I = np.array([24, 24, 1, 0, 24, 2, 2, 24, 18, 18, 12, 13, 8, 9, 14, 15, 18, 7, 7, 10, 11, 16, 17, 7, 25])
    J = np.array([0, 1, 21, 20, 2, 22, 23, 18, 12, 13, 8, 9, 14, 15, 3, 4, 7, 10, 11, 16, 17, 5, 6, 25, 19])
    
    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=linewidth, color=color)
    
    # Compute dynamic limits based on pose size
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    max_range = np.max(np.abs(vals - np.array([xroot, yroot, zroot])), axis=(0, 1))
    RADIUS = max(np.max(max_range) * 0.9, 0.4)  # Scale up and ensure minimum size
    
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_aspect('auto')

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)
    
    # Optional: set view angle (can be commented out if you want default view)
    # ax.view_init(elev=15., azim=70)


def img2video(video_path, filename, output_dir):
    """Convert pose images to video"""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    if len(names) == 0:
        print("No pose images found to create video")
        return
        
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    videoWrite = cv2.VideoWriter(output_dir + filename + '.mp4', fourcc, fps, size)

    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()
    cap.release()
    print(f"Video saved to {output_dir + filename + '.mp4'}")

def img2gif(video_path, name, output_dir, duration=0.25):
    """Convert pose images to GIF"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    image_list = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    
    if len(image_list) == 0:
        print("No pose images found to create GIF")
        return

    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    
    output_path = output_dir + name + '.gif'
    imageio.mimsave(output_path, frames, 'GIF', duration=1/fps)
    print(f"GIF saved to {output_path}")


if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    path = args.path # file path or folder path
    
    # Check if path is a directory
    if os.path.isdir(path):
        # Get all image files in the directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(path, ext)))
        image_files.sort()
        
        if len(image_files) == 0:
            print(f"No image files found in {path}")
            exit(0)
        
        print(f"Found {len(image_files)} images in {path}")
        
        # Process each image
        for img_path in tqdm(image_files, desc="Processing images"):
            filename = img_path.split('/')[-1].split('.')[0]
            output_dir = './predictions/' + filename + '/'
            
            print(f"\nProcessing: {img_path}")
            get_pose2D(img_path, output_dir, args.type)
            get_pose3D(img_path, output_dir, args.type)
        
        print(f'\nAll {len(image_files)} images processed successfully!')
    else:
        # Single file processing
        filename = path.split('/')[-1].split('.')[0]
        output_dir = './predictions/' + filename + '/'

        get_pose2D(path, output_dir, args.type)
        get_pose3D(path, output_dir, args.type)

        if args.type=="video":
            img2video(path, filename, output_dir)
            img2gif(path, filename, output_dir)

        print('Generating demo successful!')