import argparse
import math

def str2bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
    def init(self):
        self.parser.add_argument('--model', default='', type=str)
        self.parser.add_argument('--layers', default=3, type=int)
        self.parser.add_argument('--channel', default=512, type=int)
        self.parser.add_argument('--d_hid', default=1024, type=int)
        self.parser.add_argument('--dataset', type=str, default='rat7m')
        self.parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str)
        self.parser.add_argument('--data_augmentation', type=bool, default=False)
        self.parser.add_argument('--reverse_augmentation', type=bool, default=False)
        self.parser.add_argument('--test_augmentation', type=str2bool, default=False)
        self.parser.add_argument('--test_augmentation_flip_hypothesis', type=str2bool, default=False)
        self.parser.add_argument('--test_augmentation_FlowAug', type=str2bool, default=False)
        self.parser.add_argument('--crop_uv', type=int, default=0)
        self.parser.add_argument('--root_path', type=str, default='Rat7M_data/')
        self.parser.add_argument('-a', '--actions', default='*', type=str)
        self.parser.add_argument('--downsample', default=1, type=int)
        self.parser.add_argument('--subset', default=1, type=float)
        self.parser.add_argument('-s', '--stride', default=1, type=int)
        self.parser.add_argument('--gpu', default='0', type=str, help='')
        self.parser.add_argument('--train', action='store_true')
        # self.parser.add_argument('--test', action='store_true')
        self.parser.add_argument('--test', type=int, default=1) # 
        self.parser.add_argument('--nepoch', type=int, default=41) # 
        self.parser.add_argument('--batch_size', type=int, default=128, help='can be changed depending on your machine') # default 128
        self.parser.add_argument('--lr', type=float, default=1e-3)
        self.parser.add_argument('--lr_decay_large', type=float, default=0.5)
        self.parser.add_argument('--large_decay_epoch', type=int, default=20)
        self.parser.add_argument('--workers', type=int, default=8)
        self.parser.add_argument('-lrd', '--lr_decay', default=0.95, type=float)
        self.parser.add_argument('--frames', type=int, default=1)  #
        self.parser.add_argument('--pad', type=int, default=175) # 175  pad = (self.opt.frames-1) // 2 
        self.parser.add_argument('--reload', action='store_true')
        self.parser.add_argument('--model_dir', type=str, default='')
        # Optional: load model class from a specific file path
        self.parser.add_argument('--model_path', type=str, default='')

        self.parser.add_argument('--post_refine_reload', action='store_true')
        self.parser.add_argument('--checkpoint', type=str, default='')
        self.parser.add_argument('--previous_dir', type=str, default='./pre_trained_model/pretrained')
        self.parser.add_argument('--saved_model_path', type=str, default='')
        
        self.parser.add_argument('--n_joints', type=int, default=26)
        self.parser.add_argument('--out_joints', type=int, default=26)
        self.parser.add_argument('--out_all', type=int, default=1)
        self.parser.add_argument('--in_channels', type=int, default=2)
        self.parser.add_argument('--out_channels', type=int, default=3)
        self.parser.add_argument('-previous_best_threshold', type=float, default= math.inf)
        self.parser.add_argument('-previous_name', type=str, default='')
        self.parser.add_argument('--post_refine', action='store_true', help='if use post_refine model')
        self.parser.add_argument('-previous_post_refine_name', type=str, default='', help='save last saved model name')
        self.parser.add_argument('-norm', '--norm', default=0.01, type=float, metavar='N', help='constraint  of sparsity')
        self.parser.add_argument('--train_views', type=int, nargs='+', default=[0,1,2,3,4,5])
        self.parser.add_argument('--test_views', type=int, nargs='+', default=[0,1,2,3,4,5])
        self.parser.add_argument('--token_dim', type=int, default=256)
        self.parser.add_argument('--create_time', type=str, default='')
        self.parser.add_argument('--filename', type=str, default='')
        self.parser.add_argument('--single', action='store_true')
        self.parser.add_argument('--reload_3d', action='store_true')
        
        # 
        self.parser.add_argument('--create_file', type=int, default=1)
        self.parser.add_argument('--debug', action='store_true')
        self.parser.add_argument('--folder_name', type=str, default='')
        
        # param for refine
        self.parser.add_argument('--lr_refine', type=float, default=1e-5)
        self.parser.add_argument('--refine', action='store_true')
        self.parser.add_argument('--reload_refine', action='store_true')
        self.parser.add_argument('-previous_refine_name', type=str, default='')
        
        self.parser.add_argument('--sample_steps', type=int, default=3)
        # evaluation: run multiple sample steps at test time
        self.parser.add_argument('--eval_sample_steps', type=str, default='1,3,5,7,9')
        # allow multiple hypothesis counts, e.g. --num_hypothesis_list 1 3 5 7 9
        self.parser.add_argument('--num_hypothesis_list', type=str, default='1')
        self.parser.add_argument('--hypothesis_num', type=int, default=1)
        # number of best checkpoints to keep
        self.parser.add_argument('--num_saved_models', type=int, default=3)
        self.parser.add_argument('--sh_file', type=str, default='')
        # uncertainty-aware aggregation threshold factor
        
        self.parser.add_argument('--ua_k', type=float, default=0.9)
        self.parser.add_argument('--topk', type=int, default=3)
        self.parser.add_argument('--weight_softmax_tau', type=float, default=1.0)
        self.parser.add_argument('--exp_temp', type=float, default=0.002)
        self.parser.add_argument('--mode', type=str, default='exp')
        
        # mask joints
        self.parser.add_argument('--mask_prob', type=float, default=0.5)
        self.parser.add_argument('--masked_joints', type=str, default='12,13')

        # General arguments for animal3d
        self.parser.add_argument('--train_dataset_path', type=str, nargs='+', 
                                default=['./dataset/animal3d/train.json'],
                                help='List of training dataset paths (can specify multiple)')
        self.parser.add_argument('--test_dataset_path', type=str, nargs='+',
                                default=['./dataset/animal3d/test.json'],
                                help='List of test dataset paths (can specify multiple)')

        # demo
        self.parser.add_argument('--type', type=str, default='image', help='input type, only support image or video')
        self.parser.add_argument('--path', type=str, default='demo/images/running.png', help='the path of your file')
    

        
        self.parser.set_defaults(align_r=True)
        self.parser.set_defaults(align_t=True)
        self.parser.set_defaults(align_s=True)
        self.parser.set_defaults(align_trj=True)
        self.parser.set_defaults(trj_align_r=True)
        self.parser.set_defaults(trj_align_s=True)
        self.parser.set_defaults(test_flip=True)
        self.parser.set_defaults(test_rot=True)

 
    def parse(self):
        self.init()
        
        self.opt = self.parser.parse_args()
        self.opt.pad = (self.opt.frames-1) // 2

        self.opt.subjects_train = 'S1,S5,S6,S7,S8'
        self.opt.subjects_test = 'S9,S11'
                
        
    
        if self.opt.dataset == 'h36m':
            self.opt.subjects_train = 'S1,S5,S6,S7,S8'
            self.opt.subjects_test = 'S9,S11'

            if self.opt.keypoints.startswith('sh') or self.opt.keypoints.startswith('hr'):
                self.opt.n_joints = 16
                self.opt.out_joints = 16

                self.opt.joints_left = [4, 5, 6, 10, 11, 12]
                self.opt.joints_right = [1, 2, 3, 13, 14, 15]
            else:
                self.opt.n_joints = 17
                self.opt.out_joints = 17

                self.opt.joints_left = [4, 5, 6, 11, 12, 13]  # 左侧
                self.opt.joints_right = [1, 2, 3, 14, 15, 16]
        
        elif self.opt.dataset == 'rat7m':
            # Rat7M dataset configuration
            self.opt.n_joints = 20
            self.opt.out_joints = 20
            self.opt.joints_left = [8, 10, 11, 17, 18]  # HipL, ElbowL, ArmL, KneeL, ShinL
            self.opt.joints_right = [9, 14, 15, 16, 19]  # HipR, ElbowR, ArmR, KneeR, ShinR
            self.opt.root_joint = 4
        elif self.opt.dataset == 'animal3d':
            # Animal3D dataset configuration
            
            '''
            [0: 'left_eye', 1: 'right_eye', 2:'mouth_mid', 3:'left_front_paw',
            4: 'right_front_paw', 5:'left_back_paw', 6: 'right_back_paw', 7: 'tail_base',
            8: 'left_front_thigh', 9: 'right_front_thigh', 10: 'left_back_thigh', 11: 'right_back_thigh',
            12: 'left_shoulder', 13: 'right_shoulder', 14: 'left_front_knee', 15: 'right_front_knee',
            16: 'left_back_knee', 17: 'right_back_knee',18: 'neck', 19: 'tail_end',
            20: 'left_ear', 21: 'right_ear', 22: 'left_mouth', 23: 'right_mouth',
            24: 'nose', 25: 'tail_mid']
            
            skeleton pairs:
                pairs = [0, 24,  1, 24,  2, 24,  3, 14,  4, 15,  5, 16,  6, 17,  7, 18,  8, 12,  9, 13,  10, 7,  11, 7,
                12, 18,  13, 18,  14, 8,  15, 9,  16, 10,  17, 11,  18, 24,  19, 25,  20, 0,  21, 1,  22, 24,
                23, 24,  25, 7]

            '''

            self.opt.n_joints = 26
            self.opt.out_joints = 26
            # Root joint: Body_Center (index 12) is a stable center point
            # Alternative: Hip_Center (index 13)
            self.opt.root_joint = 7  # Body_Center - most stable central point -- use tail_base here 
            self.opt.joints_left = [0, 3, 5, 8, 10, 12, 14, 16, 20, 22]  
            self.opt.joints_right = [1, 4, 6, 9, 11, 13, 15, 17, 21, 23] 
                
        return self.opt