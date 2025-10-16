import numpy as np

class ChunkedGenerator:
    def __init__(self, batch_size, cameras, poses_3d, poses_2d,
                 chunk_length=1, pad=0, causal_shift=0,
                 shuffle=False, random_seed=1234,
                 augment=False, reverse_aug= False,kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, out_all = False, vis_3d=None, root_joint=4):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)
        
        # Store visibility and root joint index for filtering
        self.vis_3d = vis_3d
        self.root_joint = root_joint

        pairs = []
        self.saved_index = {}
        start_index = 0
 
        # Statistics for filtering
        total_frames = 0
        filtered_frames = 0
        
        for key in poses_2d.keys():
            assert poses_3d is None or poses_3d[key].shape[0] == poses_2d[key].shape[0]
            
            n_frames = poses_2d[key].shape[0]
            total_frames += n_frames
            
            # Get root joint visibility for this sequence
            if vis_3d is not None and key in vis_3d:
                # vis_3d shape: [frames, joints, 1]
                root_vis = vis_3d[key][:, root_joint, 0]  # [frames]
            else:
                # No visibility info, assume all frames are valid
                root_vis = np.ones(n_frames)
            
            # Generate pairs, checking each temporal window
            # Following rat7m_dataset.py logic: skip pairs where ANY frame in the window has nan root joint
            num_pairs_for_this_key = 0
            num_skipped_pairs = 0
            
            for start_idx in range(n_frames):
                end_idx = start_idx + chunk_length
                
                if end_idx > n_frames:
                    continue  # Skip if not enough frames
                
                # Check if ALL frames in [start_idx:end_idx] have valid root joint
                # This matches rat7m_dataset.py: if (tmp_vis3D[:, self.root_index] == 0).any(): continue
                window_root_vis = root_vis[start_idx:end_idx]
                if (window_root_vis != 1.0).any():
                    # Skip this pair: at least one frame has nan root joint
                    num_skipped_pairs += 1
                    continue
                
                key_array = np.array(key).reshape([1, 3])
                augment_flag = False
                reverse_flag = False
                
                # Add normal pair
                pairs.append((key_array[0], start_idx, end_idx, augment_flag, reverse_flag))
                num_pairs_for_this_key += 1
                
                # Add augmented pairs if requested
                if reverse_aug:
                    pairs.append((key_array[0], start_idx, end_idx, augment_flag, True))
                    num_pairs_for_this_key += 1
                if augment:
                    if reverse_aug:
                        pairs.append((key_array[0], start_idx, end_idx, True, True))
                        num_pairs_for_this_key += 1
                    else:
                        pairs.append((key_array[0], start_idx, end_idx, True, reverse_flag))
                        num_pairs_for_this_key += 1

            # Statistics
            if num_skipped_pairs > 0:
                print(f"Filtered {num_skipped_pairs}/{n_frames} pairs for {key} due to nan root joint in temporal window")
            filtered_frames += num_skipped_pairs

            # Save index info - use actual number of pairs generated, not just frame count
            end_index = start_index + num_pairs_for_this_key
            self.saved_index[key] = [start_index, end_index]
            start_index = end_index
        
        if vis_3d is not None:
            print(f"Total frames: {total_frames}, Filtered frames: {filtered_frames}, Valid frames: {total_frames - filtered_frames}")
            print(f"Generated {len(pairs)} pairs from valid frames")

        # Get a valid key for initializing batch arrays
        valid_key = None
        for k in poses_2d.keys():
            if k in self.saved_index:  # This key has valid frames
                valid_key = k
                break
        
        if valid_key is None:
            raise ValueError("No valid sequences found after filtering")
        
        if cameras is not None:
            self.batch_cam = np.empty((batch_size, cameras[valid_key].shape[-1]))

        if poses_3d is not None:
            self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[valid_key].shape[-2], poses_3d[valid_key].shape[-1])) # [B,1,17,3]
        self.batch_2d = np.empty((batch_size, chunk_length + 2 * pad, poses_2d[valid_key].shape[-2], poses_2d[valid_key].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        if cameras is not None:
            self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.out_all = out_all

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def get_batch(self, seq_i, start_3d, end_3d, flip, reverse):
        subject, action, cam_index = seq_i
        seq_name = (subject, action, int(cam_index))
        start_2d = start_3d - self.pad - self.causal_shift
        end_2d = end_3d + self.pad - self.causal_shift

        seq_2d = self.poses_2d[seq_name].copy()
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d
        if pad_left_2d != 0 or pad_right_2d != 0:
            self.batch_2d = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
        else:
            self.batch_2d = seq_2d[low_2d:high_2d]

        if flip:
            self.batch_2d[ :, :, 0] *= -1
            self.batch_2d[ :, self.kps_left + self.kps_right] = self.batch_2d[ :,
                                                                  self.kps_right + self.kps_left]
        if reverse:
            self.batch_2d = self.batch_2d[::-1].copy()

        if self.poses_3d is not None:
            seq_3d = self.poses_3d[seq_name].copy()
            if self.out_all:
                low_3d = low_2d
                high_3d = high_2d
                pad_left_3d = pad_left_2d
                pad_right_3d = pad_right_2d
            else:
                low_3d = max(start_3d, 0)
                high_3d = min(end_3d, seq_3d.shape[0])
                pad_left_3d = low_3d - start_3d
                pad_right_3d = end_3d - high_3d
            
            if pad_left_3d != 0 or pad_right_3d != 0:
                self.batch_3d = np.pad(seq_3d[low_3d:high_3d],
                                          ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
            else:
                self.batch_3d = seq_3d[low_3d:high_3d]

            if flip:
                self.batch_3d[ :, :, 0] *= -1
                self.batch_3d[ :, self.joints_left + self.joints_right] = \
                    self.batch_3d[ :, self.joints_right + self.joints_left]
            if reverse:
                self.batch_3d = self.batch_3d[::-1].copy()

        if self.cameras is not None:
            self.batch_cam = self.cameras[seq_name].copy()
            if flip:
                self.batch_cam[2] *= -1
                self.batch_cam[7] *= -1

        if self.poses_3d is None and self.cameras is None:
            return None, None, self.batch_2d.copy(), action, subject, int(cam_index)
        elif self.poses_3d is not None and self.cameras is None:
            return np.zeros(9), self.batch_3d.copy(), self.batch_2d.copy(),action, subject, int(cam_index)
        elif self.poses_3d is None:
            return self.batch_cam, None, self.batch_2d.copy(),action, subject, int(cam_index)
        else:
            return self.batch_cam, self.batch_3d.copy(), self.batch_2d.copy(), action, subject, int(cam_index)



            

