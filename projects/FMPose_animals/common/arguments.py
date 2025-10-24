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

        # General arguments for rat7m
        self.parser.add_argument('--cfg', help="Specify the path of the path of the config(*.yaml)", default='./cfg/rat7m/t_7_dim_4.yaml')
        self.parser.add_argument('--metric', help="eval metric", default='mpjpe') #['mpjpe', 'p_mpjpe', 'n_mpjpe']
        self.parser.add_argument('--no_align_r', dest = 'align_r', action='store_false', help='align rotation(metric)')
        self.parser.add_argument('--no_align_t', dest = 'align_t',action='store_false', help='align translatio(metric)n')
        self.parser.add_argument('--no_align_s', dest = 'align_s', action='store_false', help='align scale(metric)')
        self.parser.add_argument('--no_align_trj', dest = 'align_trj', action='store_false', help='align triangulation')
        self.parser.add_argument('--no_trj_align_r', dest = 'trj_align_r', action='store_false', help='align rotation(triangulation)')
        self.parser.add_argument('--trj_align_t', action='store_true', help='align translation(triangulation)')
        self.parser.add_argument('--no_trj_align_s', dest = 'trj_align_s', action='store_false', help='align scale(triangulation)')
        # Visualization
        self.parser.add_argument('--vis_3d', action='store_true', help='if vis 3d pose')
        self.parser.add_argument('--vis_complexity', action='store_true', help='if vis complexity')
        self.parser.add_argument('--vis_debug', action='store_true', help='save vis fig')
        self.parser.add_argument('--vis_grad', action='store_true', help='')
        self.parser.add_argument('--vis_dataset', help="Specify the name of the vis datast", default='h36m')
                
        # Rat7M dataset split
        self.parser.add_argument('--train_list', type=str, nargs='+', 
                                default=['s1d1','s2d1','s2d2','s3d1','s4d1'],
                                help='List of subjects for training (Rat7M)')
        self.parser.add_argument('--test_list', type=str, nargs='+',
                                default=['s5d1','s5d2'],
                                help='List of subjects for testing (Rat7M)')
        
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
            self.opt.n_joints = 26
            self.opt.out_joints = 26
            # Root joint: Body_Center (index 12) is a stable center point
            # Alternative: Hip_Center (index 13)
            self.opt.root_joint = 12  # Body_Center - most stable central point
            self.opt.joints_left = [8, 9, 10, 11, 17, 18, 19, 2]  # Left_Paw, Wrist, Elbow, Shoulder, Foot, Ankle, Knee, Left_Ear
            self.opt.joints_right = [4, 5, 6, 7, 14, 15, 16, 1]  # Right_Paw, Wrist, Elbow, Shoulder, Foot, Ankle, Knee, Right_Ear
                
        return self.opt