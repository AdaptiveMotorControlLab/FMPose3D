import argparse
import os
import math
import time
import torch
import socket

def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--debug', action='store_true')
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--model', default='', type=str)
    parser.add_argument('--exit', action='store_true')
    parser.add_argument('--down', default=1, type=int)

    ## mask
    parser.add_argument('--mask_ratio', default=0, type=float)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--dataset', type=str, default='h36m', help='dataset')
    parser.add_argument('--keypoints', default='cpn_ft_h36m_dbb', type=str,help='2D detections to use {gt||cpn_ft_h36m_dbb}')
    parser.add_argument('--data_augmentation', type=int, default=False, help='disable train-time flipping')##  galse
    parser.add_argument('--reverse_augmentation', type=bool, default=False,help='if reverse the video to augment data')
    parser.add_argument('--test_augmentation', type=bool, default=True,help='flip and fuse the output result')
    parser.add_argument('--crop_uv', type=int, default=0,help='if crop_uv to center and do normalization')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='dataset root path')  # liwh/Human3.6M/data/   
    parser.add_argument('--actions', default='*', type=str, help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('--downsample', default=1, type=int, help='downsample frame rate by factor (semi-supervised)')
    parser.add_argument('--subset', default=1, type=float, help='reduce dataset size by fraction')
    parser.add_argument('--stride', default=1, type=int, help='chunk size to use during training')

    parser.add_argument('--gpu', default='0', type=str, help='')
    parser.add_argument('--train', default=1)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--nepoch', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size (256)')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_refine', type=float, default=1e-5)
    parser.add_argument('--lr_decay_large', type=float, default=0.5)
    parser.add_argument('--lr_decay_epoch', type=int, default=5, help='give a large lr decay after how manys epochs')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('-lrd', '--lr_decay', default=0.95, type=float, help='learning rate decay per epoch')
    # parser.add_argument('--token_dim', default=None, type=int)

    parser.add_argument('--frames', type=int, default=1)
    parser.add_argument('--pad', type=int, default=13) 
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--refine_reload', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--out_joints', type=int, default=17, help='number of joints, 16 for human body 21 for hand pose')
    parser.add_argument('--out_all', type=int, default=1, help='output 1 frame or all frames')
    parser.add_argument('--out_channels', type=int, default=3, help='expected input channels here 2')
    parser.add_argument('--previous_best', type=float, default= math.inf)
    parser.add_argument('--previous_name', type=str, default='', help='save last saved model name')
    parser.add_argument('--previous_refine_name', type=str, default='', help='save last saved model name')

    parser.add_argument('--previous_dir', type=str, default='')
    parser.add_argument('--saved_model_path', type=str, default='')
    # Optional: load model class from a specific file path
    parser.add_argument('--model_path', type=str, default='')

    parser.add_argument('--num_saved_models', type=int, default=3)

    # parser.add_argument('--pad', type=int, default=175) # 175  pad = (self.opt.frames-1) // 2 
    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--channel', default=512, type=int)
    parser.add_argument('--d_hid', default=1024, type=int)
    parser.add_argument('--token_dim', type=int, default=256) # 
    parser.add_argument('--n_joints', type=int, default=17)    

    # wandb
    parser.add_argument('--groupname', type=str, default='')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--create_file', type=int, default=1)
    parser.add_argument('--train_views', type=int, nargs='+', default=[0,1,2,3,4,5,6,7])
    parser.add_argument('--test_views', type=int, nargs='+', default=[0,1,2,3,4,5,6,7])
    parser.add_argument('--create_time', type=str, default='')
   
    parser.add_argument('--eval_sample_steps', type=str, default='1,3,5,7,9') 
    
    args = parser.parse_args()

    if args.test:
        args.train = 0

    # if args.previous_dir != '':
    #     args.previous_dir = 'checkpoint/' + args.previous_dir

    if args.frames % 2 == 0:
        print('frames number error')
        # exit()
        
    args.pad = (args.frames-1) // 2

    ##-----------------------dataset-----------------------##
    red   = "\033[1;31m%s\033[0m"
    green = "\033[1;32m%s\033[0m"
    blue  = "\033[1;34m%s\033[0m"

    args.root_joint = 0
    if args.dataset == 'h36m':
        args.subjects_train = 'S1,S5,S6,S7,S8'
        args.subjects_test = 'S9,S11'

        if args.keypoints.startswith('sh') or args.keypoints.startswith('hr'):
            args.n_joints = 16
            args.out_joints = 16

            args.joints_left = [4, 5, 6, 10, 11, 12]
            args.joints_right = [1, 2, 3, 13, 14, 15]
        else:
            args.n_joints = 17
            args.out_joints = 17

            args.joints_left = [4, 5, 6, 11, 12, 13]  # 左侧
            args.joints_right = [1, 2, 3, 14, 15, 16]
            
    elif args.dataset.startswith('3dhp'):  
        args.root_path = './dataset/'

        args.subjects_train = 'S1,S2,S3,S4,S5,S6,S7,S8'    # all      
        # args.subjects_train = 'S1,S2,S3,S4,S5,S6'
        # args.subjects_train = 'S1'
        args.subjects_test = 'TS1,TS2,TS3,TS4,TS5,TS6' # all
        # args.subjects_test = 'TS1,TS2,TS3,TS4' #
        # args.subjects_test = 'TS1'
        # args.subjects_test = 'TS1,TS2' # GS
        # args.subjects_test = 'TS3,TS4' # no GS
        args.subjects_test = 'TS5,TS6' # Outdoor
        print(args.subjects_test)
                 
        if args.keypoints.startswith('gt_14'):
            args.n_joints, args.out_joints = 14, 14
        elif args.keypoints.startswith('gt_16'):
            args.n_joints, args.out_joints = 16, 16
        elif args.keypoints.startswith('gt_17'):
            args.n_joints, args.out_joints = 17, 17

        if args.keypoints.startswith('gt_17'):
            args.joints_left, args.joints_right = [4,5,6,11,12,13], [1,2,3,14,15,16]
        elif args.keypoints.startswith('gt_16'):
            args.joints_left, args.joints_right = [4,5,6,10,11,12], [1,2,3,13,14,15]
        elif args.keypoints.startswith('gt_14'):
            args.root_joint = 6
            args.joints_left, args.joints_right = [3,4,5,8,9,10], [0,1,2,11,12,13]

    ##-----------------------checkpoint-----------------------##
    if args.train:
        # logtime = time.strftime('%m%d_%H%M_%S_')
        # args.checkpoint = 'checkpoint/' + logtime + '%d'%(args.frames) + \
        #     '%s'%('_refine' if args.refine else '') + '_' +  args.model

        if args.dataset.startswith('3dhp'):
            args.checkpoint += '_' + str(args.dataset) + '_' + str(args.keypoints)

        if args.keypoints == 'gt':
            args.checkpoint += '_gt' 

        if args.mask_ratio > 0:
            args.checkpoint += '_pre_' + str(args.mask_ratio)

        if args.finetune:
            args.checkpoint += '_fin'
            
        if args.lr != 0.001:
            args.checkpoint += '_' + str(args.lr)

        if args.batch_size != 256:
            args.checkpoint += '_b_' + str(args.batch_size)
        
        if not torch.__version__.startswith('1.7'):
            args.checkpoint += '_torch_' + str(torch.__version__)

        if args.debug:
            args.checkpoint += '_debug' 

        if args.single and args.frames != 1:
            args.checkpoint += '_single' 

        if args.workers != 8:
            print(red % 'Workers: ' + str(args.workers))

        # if not os.path.exists(args.checkpoint):
        #     os.makedirs(args.checkpoint)

        # args_write = dict((name, getattr(args, name)) for name in dir(args)
        #         if not name.startswith('_'))

        # file_name = os.path.join(args.checkpoint, 'configs.txt')

        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write('==> Args:\n')
        #     for k, v in sorted(args_write.items()):
        #         opt_file.write('  %s: %s\n' % (str(k), str(v)))
        #     opt_file.write('==> Args:\n')
    
    print(green % 'GPU: ' + args.gpu)
    print(green % 'Model: ' + args.model)
    if args.token_dim != None:
        print(green % 'Para:', args.frames, args.layers, args.channel, args.d_hid, args.token_dim, args.lr)
    else:
        print(green % 'Para:', args.frames, args.layers, args.channel, args.d_hid, args.lr)
    print(green % 'Checkpoint: ' + args.checkpoint.split('/')[-1])

    return args




