import argparse
import math

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
    def init(self):
        self.parser.add_argument('--model', default='', type=str)
        self.parser.add_argument('--layers', default=3, type=int)
        self.parser.add_argument('--channel', default=512, type=int)
        self.parser.add_argument('--d_hid', default=1024, type=int)
        self.parser.add_argument('--dataset', type=str, default='h36m')
        self.parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str)
        self.parser.add_argument('--data_augmentation', type=bool, default=True)
        self.parser.add_argument('--reverse_augmentation', type=bool, default=False)
        self.parser.add_argument('--test_augmentation', type=bool, default=True)
        self.parser.add_argument('--crop_uv', type=int, default=0)
        self.parser.add_argument('--root_path', type=str, default='dataset/')
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
        self.parser.add_argument('--large_decay_epoch', type=int, default=5)
        self.parser.add_argument('--workers', type=int, default=8)
        self.parser.add_argument('-lrd', '--lr_decay', default=0.95, type=float)
        self.parser.add_argument('--frames', type=int, default=1)  #
        self.parser.add_argument('--pad', type=int, default=175) # 175  pad = (self.opt.frames-1) // 2 
        self.parser.add_argument('--reload', action='store_true')
        self.parser.add_argument('--model_dir', type=str, default='')

        self.parser.add_argument('--post_refine_reload', action='store_true')
        self.parser.add_argument('--checkpoint', type=str, default='')
        self.parser.add_argument('--previous_dir', type=str, default='./pre_trained_model/pretrained')
        self.parser.add_argument('--saved_model_path', type=str, default='')
        
        self.parser.add_argument('--n_joints', type=int, default=17)
        self.parser.add_argument('--out_joints', type=int, default=17)
        self.parser.add_argument('--out_all', type=int, default=1)
        self.parser.add_argument('--in_channels', type=int, default=2)
        self.parser.add_argument('--out_channels', type=int, default=3)
        self.parser.add_argument('-previous_best_threshold', type=float, default= math.inf)
        self.parser.add_argument('-previous_name', type=str, default='')
        self.parser.add_argument('--post_refine', action='store_true', help='if use post_refine model')
        self.parser.add_argument('-previous_post_refine_name', type=str, default='', help='save last saved model name')
        self.parser.add_argument('-norm', '--norm', default=0.01, type=float, metavar='N', help='constraint  of sparsity')
        self.parser.add_argument('--train_views', type=int, nargs='+', default=[0,1,2,3])
        self.parser.add_argument('--test_views', type=int, nargs='+', default=[0,1,2,3])
        
        self.parser.add_argument('--token_dim', type=int, default=256) # 
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
        self.parser.add_argument('--eval_multi_steps', action='store_true')
        self.parser.add_argument('--eval_sample_steps', type=str, default='1,3,5,7,9')
        # allow multiple hypothesis counts, e.g. --num_hypothesis_list 1 3 5 7 9
        self.parser.add_argument('--num_hypothesis_list', type=str, default='1')
        self.parser.add_argument('--hypothesis_num', type=int, default=1)
        # number of best checkpoints to keep
        self.parser.add_argument('--num_saved_models', type=int, default=3)
        
    def parse(self):
        self.init()
        
        self.opt = self.parser.parse_args()
        self.opt.pad = (self.opt.frames-1) // 2

        self.opt.subjects_train = 'S1,S5,S6,S7,S8'
        self.opt.subjects_test = 'S9,S11'
                
        self.opt.root_joint = 0
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
                       
        return self.opt