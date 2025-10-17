from torch.utils.data import Dataset
import os
import numpy as np
from torch import from_numpy as FN, tensor as PT
import copy
import cv2
from tqdm import tqdm
import sys
import os
from common.utils import loadmat
from common.camera import normalize_screen_coordinates, world_to_camera
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
sys.path.append(os.path.dirname(sys.path[0]))


# Rat7M skeleton definition (20 joints)
rat7m_skeleton = Skeleton(
    parents=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    joints_left=[8, 10, 11, 17, 18],  # HipL, ElbowL, ArmL, KneeL, ShinL
    joints_right=[9, 14, 15, 16, 19]  # HipR, ElbowR, ArmR, KneeR, ShinR
)


class Rat7MDataset(MocapDataset):
    """Rat7M Dataset - Similar structure to Human36M"""
    def __init__(self, path, opt):
        super().__init__(fps=210, skeleton=rat7m_skeleton)  # Rat7M is 210fps
        
        self.opt = opt
        self.root_index = 4  # SpineM as root
        self.joint_num = 20
        self.norm_rate = 100.0
        
        # Camera names
        self.cam_names = ['Camera1', 'Camera2', 'Camera3', 'Camera4', 'Camera5', 'Camera6']
        
        # Get train/test subjects directly from args (no auto-discovery)
        if not hasattr(opt, 'train_list') or not hasattr(opt, 'test_list'):
            raise ValueError("train_list and test_list must be provided in args!")
        
        self.train_list = opt.train_list
        self.test_list = opt.test_list
        
        print(f'Train/Test split from args:')
        print(f'  - Train subjects: {self.train_list}')
        print(f'  - Test subjects: {self.test_list}')
        
        # Verify that all specified subjects exist
        all_subjects = list(set(self.train_list + self.test_list))
        missing_subjects = []
        
        for subject in all_subjects:
            subject_folder = os.path.join(path, subject)
            mocap_file = os.path.join(subject_folder, f'mocap-{subject}.mat')
            
            if not os.path.exists(subject_folder):
                missing_subjects.append(f"{subject} (folder not found)")
            elif not os.path.exists(mocap_file):
                missing_subjects.append(f"{subject} (mocap-{subject}.mat not found)")
        
        if missing_subjects:
            raise ValueError(
                f"The following subjects are missing or invalid:\n" +
                "\n".join(f"  - {s}" for s in missing_subjects) +
                f"\n\nDataset path: {path}\n" +
                f"Please check your --train_list and --test_list arguments."
            )
        
        print(f'  ✓ All {len(all_subjects)} subjects verified')
        
        # Initialize camera parameters and data storage
        self._cameras = {}
        self._data = {}
        
        print('Loading Rat7M dataset...')
        # Load all subjects (train + test)
        subjects_to_load = list(set(self.train_list + self.test_list))
        for subject_name in tqdm(subjects_to_load):
            subject_folder = os.path.join(path, subject_name)
            s_label_path = os.path.join(subject_folder, f'mocap-{subject_name}.mat')
            
            if not os.path.exists(s_label_path):
                print(f"Warning: {s_label_path} not found, skipping...")
                continue
                
            s_label_mat = loadmat(s_label_path)
            
            # Process camera parameters
            self._cameras[subject_name] = []
            for cam in self.cam_names:
                cam_params = {}
                cam_params['id'] = cam
                cam_params['orientation'] = s_label_mat['cameras'][cam]['rotationMatrix'].T
                cam_params['translation'] = s_label_mat['cameras'][cam]['translationVector'] / self.norm_rate
                
                K_tmp = s_label_mat['cameras'][cam]['IntrinsicMatrix']
                K_tmp[1, 0] = 0.0
                cam_params['K'] = K_tmp.T
                
                # Focal length and center
                cam_params['focal_length'] = np.array([K_tmp[0, 0], K_tmp[1, 1]], dtype='float32')
                cam_params['center'] = np.array([K_tmp[2, 0], K_tmp[2, 1]], dtype='float32')
                
                # Distortion
                cam_params['radial_distortion'] = np.array([
                    s_label_mat['cameras'][cam]['RadialDistortion'][0],
                    s_label_mat['cameras'][cam]['RadialDistortion'][1],
                    0.0
                ], dtype='float32')
                cam_params['tangential_distortion'] = np.array([
                    s_label_mat['cameras'][cam]['TangentialDistortion'][0],
                    s_label_mat['cameras'][cam]['TangentialDistortion'][1]
                ], dtype='float32')
                
                # Resolution (from original code)
                cam_params['res_w'] = 1328
                cam_params['res_h'] = 1048
                
                # Normalize screen coordinates if needed
                if opt.crop_uv == 0:
                    cam_params['center'] = normalize_screen_coordinates(
                        cam_params['center'], w=cam_params['res_w'], h=cam_params['res_h']
                    ).astype('float32')
                    cam_params['focal_length'] = cam_params['focal_length'] / cam_params['res_w'] * 2
                
                # Create intrinsic vector (focal_length, center, radial, tangential)
                cam_params['intrinsic'] = np.concatenate((
                    cam_params['focal_length'],
                    cam_params['center'],
                    cam_params['radial_distortion'],
                    cam_params['tangential_distortion']
                ))
                
                self._cameras[subject_name].append(cam_params)
            
            # Process 3D mocap data
            total_frames = s_label_mat['mocap']['HeadF'].shape[0]
            
            # Create positions array [frames, joints, 3]
            positions = np.zeros((total_frames, self.joint_num, 3), dtype='float32')
            
            for joint_id, joint_name in enumerate(s_label_mat['mocap'].keys()):
                joint_coor = s_label_mat['mocap'][joint_name]  # [frames, 3]
                positions[:, joint_id, :] = joint_coor / self.norm_rate
            
            # Store data directly under subject (no action level for Rat7M)
            self._data[subject_name] = {
                'positions': positions,
                'cameras': self._cameras[subject_name]
            }
        
        # Check if any data was loaded
        if len(self._data) == 0:
            raise ValueError(
                f"No valid subjects were loaded from {path}.\n"
                f"Expected subjects: {subjects_to_load}\n"
                f"Please check:\n"
                f"  1. Dataset path is correct: {path}\n"
                f"  2. Subject folders exist (e.g., s1d1/, s2d1/)\n"
                f"  3. .mat files exist in each folder (e.g., mocap-s1d1.mat)"
            )
        
        print(f'Successfully loaded {len(self._data)} subjects: {list(self._data.keys())}')


class Rat7MDatasetOld(Dataset):
    def __init__(self, path, split, cam_names, t_pad, root_index = 4, use_2D_gt = True, aug_2D = False,
                       joint_num = 20, sampling_gap = 100, frame_per_video = 3500, norm_rate = 100.0,
                       img_W = 1328, img_H = 1048, arg_views = 1, pose_2D_path = None, resize_2D_scale = 1.):
        self.cam_names = cam_names
        self.joint_num = joint_num
        self.t_pad = t_pad
        self.t_length = (t_pad*2) + 1
        self.root_index = root_index
        self.use_2D_gt = use_2D_gt
        self.aug_2D = aug_2D
        self.img_W = img_W * resize_2D_scale
        self.img_H = img_H * resize_2D_scale
        self.arg_views = arg_views
        self.split = split
        self.pose_2D_path = pose_2D_path

        subject_index = os.listdir(path)
        subject_index.remove('jesse_skeleton.mat')
        # subject_index.remove('s1d1')
        subject_index.sort()

        if split == 'Train':
            self.subject_list = subject_index[:4]
            self.start_frame = 50
            self.end_frame = 54000
        elif split == 'Test':
            self.subject_list = subject_index[:4]
            self.start_frame = 0
            self.end_frame = np.inf
        
        print('Prepare the pose data...')
        self.pose_3D_list = []
        self.pose_2D_list = []
        self.vid2D_list = []
        self.vid3D_list = []
        self.sample_info_list = []
        self.cam_para_list = []
        for sub_idx, subject_name in enumerate(self.subject_list):
            print(subject_name)
            subject_folder = os.path.join(path, subject_name)
            s_label_path = os.path.join(subject_folder, 'mocap-{}.mat'.format(subject_name))
            s_label_mat = loadmat(s_label_path)
            if not use_2D_gt:
                s_pose_2D_path = os.path.join(pose_2D_path, subject_name)
                DLC_predictions = {}
                for cam in cam_names:
                    DLC_predictions[cam] = np.load(os.path.join(s_pose_2D_path, subject_name + '_' + cam.lower() + '_DLC_filtered.npy'))

            tmp_cam_para = {}
            for cam in cam_names:
                tmp_cam_para[cam] = {}
                tmp_cam_para[cam]['R'] = s_label_mat['cameras'][cam]['rotationMatrix'].T

                tmp_cam_para[cam]['t'] = s_label_mat['cameras'][cam]['translationVector'] / norm_rate
                K_tmp = s_label_mat['cameras'][cam]['IntrinsicMatrix']
                K_tmp[1,0] = 0.0
                tmp_cam_para[cam]['K'] = K_tmp.T
                Distort = np.array([s_label_mat['cameras'][cam]['RadialDistortion'][0],
                    s_label_mat['cameras'][cam]['RadialDistortion'][1],
                    s_label_mat['cameras'][cam]['TangentialDistortion'][0],
                    s_label_mat['cameras'][cam]['TangentialDistortion'][1]])
                tmp_cam_para[cam]['Distort'] = Distort

            total_frame_num = s_label_mat['mocap']['HeadF'].shape[0]
            real_end_frame = min(self.end_frame, total_frame_num)
            for idx in tqdm(range(self.start_frame, real_end_frame)):
                idx = max(idx, self.t_pad)
                idx = min(idx, real_end_frame - self.t_pad - 1)
                left_frame_id = idx - self.t_pad
                right_frame_id = idx + self.t_pad + 1
                tmp_3D = np.zeros((self.t_length, self.joint_num, 3, len(self.cam_names)))
                tmp_3D_word = np.zeros((self.t_length, self.joint_num, 3))
                tmp_vis2D = np.zeros((self.t_length, self.joint_num, 1, len(self.cam_names)))
                tmp_vis3D = np.zeros((self.t_length, self.joint_num, 1))
                tmp_info = np.zeros(2)
                tmp_info[0] = sub_idx
                tmp_info[1] = idx

                # if idx == 38600:
                #     aa = 1

                for joint_id, joint in enumerate(s_label_mat['mocap'].keys()): # read 3d keypoints
                    joint_coor = s_label_mat['mocap'][joint][left_frame_id: right_frame_id]
                    tmp_vis3D[np.where(~np.isnan(np.sum(joint_coor, axis=1))), joint_id] = 1.0
                    tmp_3D_word[:, joint_id,:] = joint_coor / norm_rate
                
                if (tmp_vis3D[:, self.root_index] == 0).any():
                    continue
                
                tmp_3D_word_reshaped = np.reshape(tmp_3D_word, (-1, 3))
                for cam_idx, cam in enumerate(cam_names):
                    tmp_3D_cam_reshaped = np.dot(tmp_3D_word_reshaped, tmp_cam_para[cam]['R'].T) + tmp_cam_para[cam]['t']
                    tmp_3D[:,:,:, cam_idx] = np.reshape(tmp_3D_cam_reshaped, (self.t_length, self.joint_num, 3))
                
                tmp_2D = np.zeros((self.t_length, self.joint_num, 2, len(self.cam_names)))
                if use_2D_gt:
                    for j_cam, cam in enumerate(cam_names):
                        tmp_vis2D[:,:,:,j_cam] = copy.deepcopy(tmp_vis3D)
                        rvec, _ = cv2.Rodrigues(tmp_cam_para[cam]['R'])
                        xy, _ = cv2.projectPoints(tmp_3D_word_reshaped, rvec, tmp_cam_para[cam]['t'], tmp_cam_para[cam]['K'], tmp_cam_para[cam]['Distort'])
                        xy_reshaped = np.reshape(xy[:,0,:], (self.t_length, self.joint_num, 2))

                        # cam_id = s_label_mat['cameras'][cam]['frame'][idx]
                        # s_video_dir = os.path.join(subject_folder, subject_name + '_video')
                        # s_c_video_path = os.path.join(s_video_dir, subject_name[:2] + '-' + subject_name[2:] + '-{}-{}.mp4'.format(cam.lower(), cam_id-cam_id%frame_per_video))
                        # cap = VideoReader(s_c_video_path)
                        # cap.set_to_frame(cam_id % frame_per_video)
                        # view_image = cap.read_frame()
                        # fig = plt.figure()
                        # plt.imshow(view_image)
                        # rvec, _ = cv2.Rodrigues(tmp_cam_para[cam]['R'])
                        # s = plt.scatter(xy_reshaped[self.t_pad,:,0], xy_reshaped[self.t_pad,:,1], label=list(s_label_mat['mocap'].keys()), c=list(range(self.joint_num)), cmap='jet', s=3)
                        # cbar = fig.colorbar(mappable=s, cmap='jet', ticks=list(range(len(list(s_label_mat['mocap'].keys())))))
                        # cbar.ax.set_yticklabels(list(s_label_mat['mocap'].keys()))
                        # plt.show()
                        # plt.savefig('{}sd.png'.format(cam), dpi=400.0)
                        # plt.close()
                        # aa = 1

                        tmp_2D[:,:,:,j_cam] = normalize_screen_coordinates(xy_reshaped, self.img_W, self.img_H)
                else:
                    for j_cam, cam in enumerate(cam_names):
                        cam_ids = s_label_mat['cameras'][cam]['frame'][left_frame_id: right_frame_id]
                        pose_2D_vis = copy.deepcopy(DLC_predictions[cam][cam_ids])

                        # s_video_dir = os.path.join(subject_folder, subject_name + '_video')
                        # s_c_video_path = os.path.join(s_video_dir, subject_name[:2] + '-' + subject_name[2:] + '-{}-{}.mp4'.format(cam.lower(), cam_ids[self.t_pad]-cam_ids[self.t_pad]%frame_per_video))
                        # cap = VideoReader(s_c_video_path)
                        # cap.set_to_frame(cam_ids[self.t_pad] % frame_per_video)
                        # view_image = cap.read_frame()
                        # fig = plt.figure()
                        # plt.imshow(view_image)
                        # pose_2D_vis_back_resize = pose_2D_vis.copy() * 2.0
                        # plt.scatter(pose_2D_vis_back_resize[self.t_pad,:,0], pose_2D_vis_back_resize[self.t_pad,:,1], c=list(range(self.joint_num)), cmap='jet', s=3)
                        # plt.savefig('{}sd_new.png'.format(cam), dpi=200.0)
                        # plt.close()

                        tmp_2D[:,:,:,j_cam] = normalize_screen_coordinates(copy.deepcopy(pose_2D_vis[:,:,:2]), self.img_W, self.img_H)
                        tmp_vis2D[:,:,:,j_cam] = copy.deepcopy(pose_2D_vis[:,:,2:])
                    aa = 1
                
                tmp_3D_root = copy.deepcopy(tmp_3D[:,self.root_index:self.root_index+1,:,:])
                tmp_3D = tmp_3D - tmp_3D_root
                tmp_3D[:,self.root_index:self.root_index+1,:,:] = tmp_3D_root
            
                
                # self.pose_3D_list.append(np.nan_to_num(tmp_3D))
                self.pose_3D_list.append(tmp_3D)
                self.pose_2D_list.append(np.nan_to_num(tmp_2D))
                self.vid2D_list.append(tmp_vis2D)
                self.vid3D_list.append(tmp_vis3D)
                self.sample_info_list.append(tmp_info)
            self.cam_para_list.append(tmp_cam_para)
        aa = 1



    def __len__(self):
        return len(self.vid3D_list)

    def __getitem__(self, index):
        return self.getitem(index)

    def getitem(self, index):
        pose_3D = copy.deepcopy(self.pose_3D_list[index])  # T,K,3,N
        pose_2D = copy.deepcopy(self.pose_2D_list[index])  # T,K,2,N
        vid_2D = copy.deepcopy(self.vid2D_list[index])     # T,K,1,N
        vid_3D = copy.deepcopy(self.vid3D_list[index])     # T,K,1
        sample_info = copy.deepcopy(self.sample_info_list[index])  # 2
        
        # if self.use_2D_gt and self.aug_2D:
        if "TRAIN" in self.split.upper() and self.arg_views > 0:
            pose_3D, pose_2D = self.view_aug(pose_3D, pose_2D)
            tmp_vid = np.repeat(np.expand_dims(copy.deepcopy(vid_3D), axis=-1), self.arg_views, axis = -1)
            vid_2D = np.concatenate((vid_2D, tmp_vid), axis = -1)
        
        pose_root = copy.deepcopy(pose_3D[:,self.root_index:self.root_index+1,:,:])
        pose_3D[:,self.root_index:self.root_index+1,:,:] = 0.0
        pose_3D = np.nan_to_num(pose_3D, nan=0)
        pose_2D = np.concatenate((pose_2D, vid_2D), axis=2)

        return FN(pose_3D).float(), FN(pose_root).float(), FN(pose_2D).float(), FN(vid_3D).float(), FN(sample_info).float()


if __name__ == '__main__':
    cam_names = ['Camera1', 'Camera2', 'Camera3', 'Camera4', 'Camera5', 'Camera6']
    data_dir = '/home/xiaohang/Ti_workspace/projects/FMPose_animals/dataset/rat7m/'

    valid_dataset = Rat7MDataset(data_dir, 'Train', cam_names, 3)

    pose_3D, pose_root, pose_2D, vid_3D, rotation, sample_info = valid_dataset.getitem(1)
    # print(pose_3D.shape, pose_root.shape, pose_2D.shape, vid_3D.shape, sample_info)
    