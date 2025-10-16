
import torch.utils.data as data
import numpy as np
import cv2

from common.utils import deterministic_random
from common.camera import normalize_screen_coordinates
from common.generator import ChunkedGenerator

class Rat7MFusion(data.Dataset):
    """Rat7M Fusion dataset for single-view 2D-to-3D lifting"""
    def __init__(self, opt, dataset, root_path, train=True):
        self.data_type = opt.dataset  # rat7m
        self.train = train
        self.root_path = root_path
        self.root_joint = opt.root_joint
        # For Rat7M, subjects are like 's1d1', 's2d1', etc.
        # Use first 4 subjects for both train and test
        self.train_list = dataset.train_list  # ['s1d1', 's2d1', 's2d2', 's3d1']
        self.test_list = dataset.test_list    # Same as train_list
        
        # Rat7M doesn't have actions, but we use 'rat_motion' as placeholder for compatibility
        self.action_filter = None
        
        self.downsample = opt.downsample  # 1
        self.subset = opt.subset  # 1
        self.stride = opt.stride  # 1
        self.crop_uv = opt.crop_uv  # 0
        self.pad = opt.pad  # 0
        
        # Frame ranges for train and test
        self.train_start_frame = 50
        self.train_end_frame = np.inf  # Use all frames for training
        self.test_start_frame = 0
        self.test_end_frame = np.inf  # Use all frames for test
        
        if self.train:
            # Prepare training data
            self.cameras_train, self.poses_train, self.poses_train_2d, self.vis_train = self.fetch(
                dataset, self.train_list, 
                subset=self.subset, 
                views=getattr(opt, 'train_views', [0, 1, 2, 3, 4, 5]),  # All 6 cameras
                start_frame=self.train_start_frame,
                end_frame=self.train_end_frame
            )
            
            self.generator = ChunkedGenerator(
                opt.batch_size // opt.stride, 
                self.cameras_train, 
                self.poses_train,
                self.poses_train_2d, 
                self.stride, 
                pad=self.pad,
                augment=False,  # No data augmentation
                reverse_aug=False,  # No reverse augmentation
                kps_left=self.kps_left, 
                kps_right=self.kps_right,
                joints_left=self.joints_left,
                joints_right=self.joints_right, 
                out_all=opt.out_all
            )
            print('INFO: Training on {} frames'.format(self.generator.num_frames()))
        else:
            # Prepare test data
            self.cameras_test, self.poses_test, self.poses_test_2d, self.vis_test = self.fetch(
                dataset, self.test_list,
                subset=self.subset, 
                views=getattr(opt, 'test_views', [0, 1, 2, 3, 4, 5]),
                start_frame=self.test_start_frame,
                end_frame=self.test_end_frame
            )
            
            self.generator = ChunkedGenerator(
                opt.batch_size // opt.stride, 
                self.cameras_test, 
                self.poses_test,
                self.poses_test_2d,
                pad=self.pad, 
                augment=False, 
                kps_left=self.kps_left,
                kps_right=self.kps_right, 
                joints_left=self.joints_left,
                joints_right=self.joints_right
            )
            self.key_index = self.generator.saved_index
            print('INFO: Testing on {} frames'.format(self.generator.num_frames()))

    def fetch(self, dataset, subjects, views=[0, 1, 2, 3, 4, 5], subset=1, 
              start_frame=0, end_frame=np.inf):
        """Fetch data for specified subjects and views
        Note: Rat7M doesn't have actions, so we use subject directly
        """
        out_poses_3d = {}
        out_poses_2d = {}
        out_camera_params = {}
        out_vis_3d = {}  # Add visibility labels

        # Get joint symmetry from skeleton (only needed for flip augmentation)
        # Since we disabled augmentation, these are not used but kept for compatibility
        self.joints_left = list(dataset.skeleton().joints_left())
        self.joints_right = list(dataset.skeleton().joints_right())
        self.kps_left = self.joints_left  # For Rat7M, use same as joints
        self.kps_right = self.joints_right

        for subject in subjects:
            if subject not in dataset._data:
                print(f"Warning: Subject {subject} not found in dataset")
                continue
            
            # Rat7M doesn't have action level, access data directly
            anim = dataset._data[subject]
            
            # Get positions and cameras
            positions_world = anim['positions']  # [frames, joints, 3]
            cameras = anim['cameras']  # List of camera dicts
            
            # Determine actual frame range
            total_frames = positions_world.shape[0]
            # Handle infinity case for end_frame
            if np.isinf(end_frame):
                actual_end_frame = total_frames
            else:
                actual_end_frame = min(int(end_frame), total_frames)
            actual_start_frame = max(int(start_frame), 0)
            
            # Calculate visibility for all frames (check for nan values)
            # vis_3d: [frames, joints, 1] - 1.0 for valid, 0.0 for nan
            positions_subset = positions_world[actual_start_frame:actual_end_frame]
            vis_3d = np.ones((positions_subset.shape[0], positions_subset.shape[1], 1), dtype='float32')
            
            # Check for nan values along the coordinate axis (axis=2)
            # If any coordinate (x,y,z) is nan, mark joint as invisible
            nan_mask = np.isnan(np.sum(positions_subset, axis=2))  # [frames, joints]
            vis_3d[nan_mask, 0] = 0.0
            
            # Filter out frames where root joint is not visible (following rat7m_dataset.py)
            # Skip poses where root joint (SpineM, index 4) has nan values
            root_joint_index = self.root_joint  # SpineM is the root joint for Rat7M
            valid_frames_mask = vis_3d[:, root_joint_index, 0] == 1.0
            
            if not valid_frames_mask.any():
                print(f"Warning: No valid frames for {subject} (all root joint have nan)")
                continue
            
            # Apply valid frames filter
            positions_subset = positions_subset[valid_frames_mask]
            vis_3d = vis_3d[valid_frames_mask]
            
            # Process each camera view
            for cam_idx in views:
                if cam_idx >= len(cameras):
                    continue
                    
                cam = cameras[cam_idx]
                
                # Convert world coordinates to camera coordinates
                # Following rat7m_dataset.py approach: use matrix multiplication
                frames, joints, _ = positions_subset.shape
                pos_world_reshaped = positions_subset.reshape(-1, 3)  # [frames*joints, 3]
                
                # World to camera: X_cam = X_world @ R.T + t
                pos_cam_reshaped = np.dot(pos_world_reshaped, cam['orientation'].T) + cam['translation']
                pos_3d = pos_cam_reshaped.reshape(frames, joints, 3)  # [frames, joints, 3]
                
                # Make root-relative (subtract root joint from all joints)
                # Following rat7m_dataset.py: use SpineM (index 4) as root
                root_joint_idx = self.root_joint  # SpineM is the root joint for Rat7M
                pos_3d_root = pos_3d[:, root_joint_idx:root_joint_idx+1, :].copy()  # Save root
                pos_3d = pos_3d - pos_3d_root  # All joints relative to root
                # Note: root joint itself is now [0, 0, 0] after subtraction
                
                # Replace nan with 0 after root-relative transformation
                pos_3d = np.nan_to_num(pos_3d, nan=0.0)
                
                # Project 3D to 2D using camera parameters
                pos_2d = self.project_to_2d(
                    positions_subset, 
                    cam
                )
                
                # Store data with key (subject, 'rat_motion', cam_idx) to maintain compatibility
                # Use 'rat_motion' as a placeholder action name for compatibility with generator
                key = (subject, 'rat_motion', cam_idx)
                out_poses_3d[key] = pos_3d
                out_poses_2d[key] = pos_2d
                out_vis_3d[key] = vis_3d
                
                if 'intrinsic' in cam:
                    out_camera_params[key] = cam['intrinsic']

        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None

        # Apply subsampling
        stride = self.downsample
        if subset < 1:
            for key in out_poses_2d.keys():
                n_frames = int(round(len(out_poses_2d[key]) // stride * subset) * stride)
                start = deterministic_random(0, len(out_poses_2d[key]) - n_frames + 1, str(len(out_poses_2d[key])))
                out_poses_2d[key] = out_poses_2d[key][start:start + n_frames:stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][start:start + n_frames:stride]
                if out_vis_3d is not None:
                    out_vis_3d[key] = out_vis_3d[key][start:start + n_frames:stride]
        elif stride > 1:
            for key in out_poses_2d.keys():
                out_poses_2d[key] = out_poses_2d[key][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][::stride]
                if out_vis_3d is not None:
                    out_vis_3d[key] = out_vis_3d[key][::stride]

        return out_camera_params, out_poses_3d, out_poses_2d, out_vis_3d

    def project_to_2d(self, positions_world, cam):
        """Project 3D world coordinates to 2D image coordinates"""
        # positions_world: [frames, joints, 3]
        n_frames, n_joints, _ = positions_world.shape
        
        # Reshape for projection
        pos_world_reshaped = positions_world.reshape(-1, 3)  # [frames*joints, 3]
        
        # Get camera parameters
        R = cam['orientation']
        t = cam['translation']
        K = cam['K']
        
        # Check if distortion parameters exist
        if 'radial_distortion' in cam and 'tangential_distortion' in cam:
            # Combine distortion parameters for OpenCV
            # OpenCV expects: [k1, k2, p1, p2, k3] where k are radial, p are tangential
            radial = cam['radial_distortion'][:2]  # Use first 2 radial distortion coeffs
            tangential = cam['tangential_distortion']
            dist_coeffs = np.array([radial[0], radial[1], tangential[0], tangential[1]])
        else:
            dist_coeffs = np.zeros(4)
        
        # Project using OpenCV
        rvec, _ = cv2.Rodrigues(R)
        pts_2d, _ = cv2.projectPoints(pos_world_reshaped, rvec, t, K, dist_coeffs)
        pts_2d = pts_2d[:, 0, :]  # Remove middle dimension: [frames*joints, 2]
        
        # Reshape back
        pts_2d = pts_2d.reshape(n_frames, n_joints, 2)
        
        # Normalize to [-1, 1] if crop_uv is 0
        if self.crop_uv == 0:
            pts_2d = normalize_screen_coordinates(pts_2d, w=cam['res_w'], h=cam['res_h'])
        
        return pts_2d

    def __len__(self):
        return len(self.generator.pairs)

    def __getitem__(self, index):
        seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[index]
        
        # Convert seq_name to tuple if it's a numpy array (for dictionary key)
        if isinstance(seq_name, np.ndarray):
            seq_name = tuple(seq_name)
        
        # Convert numpy types to Python native types for dictionary key matching
        # seq_name format: (subject, action, cam_idx)
        seq_name = (str(seq_name[0]), str(seq_name[1]), int(seq_name[2]))

        cam, gt_3D, input_2D, action, subject, cam_ind = self.generator.get_batch(
            seq_name, start_3d, end_3d, flip, reverse
        )
        
        # Get visibility labels
        if self.train:
            vis_dict = self.vis_train
        else:
            vis_dict = self.vis_test
        
        # Extract vis_3D for current sequence
        vis_3D = vis_dict[seq_name].copy()
        # Apply same slicing as poses
        low_3d = max(start_3d, 0)
        high_3d = min(end_3d, vis_3D.shape[0])
        pad_left_3d = low_3d - start_3d
        pad_right_3d = end_3d - high_3d
        
        if pad_left_3d != 0 or pad_right_3d != 0:
            vis_3D = np.pad(vis_3D[low_3d:high_3d], 
                           ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
        else:
            vis_3D = vis_3D[low_3d:high_3d]
        
        # No flip operation needed for vis_3D (visibility is same for left/right joints)
        # Only handle reverse if needed
        if reverse:
            vis_3D = vis_3D[::-1].copy()
        
        bb_box = np.array([0, 0, 1, 1])
        scale = 1.0
        
        return cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind, vis_3D

