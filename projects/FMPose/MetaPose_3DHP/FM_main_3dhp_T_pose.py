import os
import time
import torch
import random
import logging
import matplotlib
import numpy as np
from tqdm import tqdm
matplotlib.use('Agg')
from common.utils import *
import torch.optim as optim
from common.camera import *
from common.utils import *
import common.eval_cal as eval_cal
from common.arguments import parse_args
from model.utils.post_refine import post_refine
from common.dataset.load_data_hm36 import Fusion
from common.dataset.load_data_3dhp import Fusion_3dhp
from common.dataset.h36m_dataset import Human36mDataset
from common.dataset.mpi_inf_3dhp_dataset import Mpi_inf_3dhp_Dataset
import numpy as np 
args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
exec('from model.' + args.model + ' import Model')

def train(dataloader, model, model_refine, optimizer):
    model.train()
    loss_all = {'loss': AccumLoss()}

    for i, data in enumerate(tqdm(dataloader, 0)):
        batch_cam, gt_3D, input_2D, input_2D_GT, action, subject, cam_ind = data
        [input_2D, input_2D_GT, gt_3D, batch_cam] = get_varialbe('train', [input_2D, input_2D_GT, gt_3D, batch_cam])

        output_3D = model(input_2D) # B F J 3

        out_target = gt_3D.clone() # B F J 3
        out_target[:, :, args.root_joint] = 0

        if args.single:
            out_target = out_target[:, args.pad].unsqueeze(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.debug and i > 2:
            break

        N = input_2D.shape[0]
        loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

    return loss_all['loss'].avg


def test(actions, dataloader, model, model_refine):
    
    model.eval()

    error_sum_2d_in, error_sum_joints = AccumLoss(), AccumLoss()

    # Load T-pose template once (expected shape: (1,1,J,3))
    tpose_tensor = None
    tpose_path = os.path.join('./dataset', 'mean_3D_pose.npz')
    npz = np.load(tpose_path)
    tpose_arr = npz['mean_3D']
    tpose_tensor = torch.from_numpy(tpose_arr).float().cuda()
  
    # for multi-step eval, maintain per-step accumulators across the whole split
    eval_steps = None
    action_error_sum_multi = None
    eval_steps = sorted({int(s) for s in getattr(args, 'eval_sample_steps', '3').split(',') if str(s).strip()})
    action_error_sum_multi = {s: define_error_list(actions) for s in eval_steps}

    for i, data in enumerate(tqdm(dataloader, 0)):
        batch_cam, gt_3D, input_2D, input_2D_GT, action, subject, cam_ind = data # test  
        [input_2D, input_2D_GT, gt_3D, batch_cam] = get_varialbe('test', [input_2D, input_2D_GT, gt_3D, batch_cam])

        N = input_2D.size(0)

        input_2D_nonflip = input_2D[:, 0]
        input_2D_flip = input_2D[:, 1]
        out_target = gt_3D.clone()
        out_target[:, :, args.root_joint] = 0

        # print("gt_3D.shape:", gt_3D.shape)
        
        # Simple Euler sampler for CFM at test time (independent runs per step if eval_multi_steps)
        def euler_sample(x2d, y_local, steps, model_3d):
            dt = 1.0 / steps
            for s in range(steps):
                t_s = torch.full((gt_3D.size(0), 1, 1, 1), s * dt, device=gt_3D.device, dtype=gt_3D.dtype)
                v_s = model_3d(x2d, y_local, t_s)
                y_local = y_local + dt * v_s
            return y_local

        # for each requested step count, run an independent sampling (no default output here)
        for s_keep in eval_steps:
            # Initialize from T-pose (no added noise here)
            y = tpose_tensor.clone().to(device=gt_3D.device, dtype=gt_3D.dtype)
            y = y.expand_as(gt_3D)
            y[:, :, 0, :] = 0
            y = y + torch.randn_like(y)
            
            y_s = euler_sample(input_2D_nonflip, y, s_keep, model)
            
            if args.test_augmentation:
                joints_left = [4, 5, 6, 11, 12, 13]
                joints_right = [1, 2, 3, 14, 15, 16]
                # Flip-start from T-pose as well
                y_flip = tpose_tensor.clone().to(device=gt_3D.device, dtype=gt_3D.dtype)
                y_flip[:, :, :, 0] *= -1
                y_flip[:, :, joints_left + joints_right, :] = y_flip[:, :, joints_right + joints_left, :]
                y_flip = y_flip.expand_as(gt_3D)
                y_flip[:, :, 0, :] = 0
                y_flip = y_flip + torch.randn_like(y_flip)

                y_flip_s = euler_sample(input_2D_flip, y_flip, s_keep, model)
                y_flip_s = y_flip_s.clone()
                y_flip_s[:, :, :, 0] *= -1
                y_flip_s[:, :, joints_left + joints_right, :] = y_flip_s[:, :, joints_right + joints_left, :]
                y_s = (y_s + y_flip_s) / 2
            # per-step metrics only; do not store per-sample outputs
            output_3D_s = y_s[:, args.pad].unsqueeze(1)
            output_3D_s[:, :, 0, :] = 0

            # accumulate by action across the entire test set
            action_error_sum_multi[s_keep] = eval_cal.test_calculation(output_3D_s, out_target, action, action_error_sum_multi[s_keep], args.dataset, subject)

        # if not args.single:
        #     output_3D = output_3D[:, args.pad].unsqueeze(1)
        # if args.refine:
        #     model_refine.eval()
        #     output_3D = refine_model(model_refine, output_3D, input_2D[:, 0], gt_3D, batch_cam) # B 1 J 3

    per_step_p1 = {}
    per_step_p2 = {}
    per_step_pck = {}
    per_step_auc = {}
    for s_keep in sorted(action_error_sum_multi.keys()):
        p1_s, p2_s, pck, auc = print_error(args.dataset, action_error_sum_multi[s_keep], args.train)
        per_step_p1[s_keep] = float(p1_s)
        per_step_p2[s_keep] = float(p2_s)
        per_step_pck[s_keep] = float(pck)
        per_step_auc[s_keep] = float(auc)

    return per_step_p1, per_step_p2, per_step_pck, per_step_auc

def print_error(data_type, action_error_sum, is_train):
    mean_error_p1, mean_error_p2, pck, auc = 0, 0, 0, 0
    if data_type == 'h36m' or data_type.startswith('3dhp'):
        mean_error_p1, mean_error_p2, pck, auc = print_error_action(action_error_sum, is_train, data_type)

    return mean_error_p1, mean_error_p2, pck, auc

def print_error_action(action_error_sum, is_train, data_type):
    mean_error_each = {'p1': 0.0, 'p2': 0.0, 'pck': 0.0, 'auc': 0.0}
    mean_error_all  = {'p1': AccumLoss(), 'p2': AccumLoss(), 'pck': AccumLoss(), 'auc': AccumLoss()}

    if not is_train:
        if data_type.startswith('3dhp'):
            print("{0:=^12} {1:=^10} {2:=^8} {3:=^8} {4:=^8}".format("Action", "p#1 mm", "p#2 mm", "PCK", "AUC"))
            logging.info("{0:=^12} {1:=^10} {2:=^8} {3:=^8} {4:=^8}".format("Action", "p#1 mm", "p#2 mm", "PCK", "AUC"))
        else:
            print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))

    for action, value in action_error_sum.items():
        if not is_train:
            print("{0:<12} ".format(action), end="")

        mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg * 1000.0
        mean_error_all['p2'].update(mean_error_each['p2'], 1)

        mean_error_each['pck'] = action_error_sum[action]['pck'].avg * 100.0
        mean_error_all['pck'].update(mean_error_each['pck'], 1)

        mean_error_each['auc'] = action_error_sum[action]['auc'].avg * 100.0
        mean_error_all['auc'].update(mean_error_each['auc'], 1)

        if is_train == 0:
            if data_type.startswith('3dhp'):
                print("{0:>6.2f} {1:>10.2f} {2:>10.2f} {3:>10.2f}".format(
                    mean_error_each['p1'], mean_error_each['p2'], 
                    mean_error_each['pck'], mean_error_each['auc']))
                logging.info("{0:<12} {1:>6.2f} {2:>10.2f} {3:>10.2f} {4:>10.2f}".format(
                    action, mean_error_each['p1'], mean_error_each['p2'], 
                    mean_error_each['pck'], mean_error_each['auc']))
            else:
                print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'], mean_error_each['p2']))

    if is_train == 0:
        if data_type.startswith('3dhp'):
            print("{0:<12} {1:>6.2f} {2:>10.2f} {3:>10.2f} {4:>10.2f}".format(
                "Average", mean_error_all['p1'].avg, mean_error_all['p2'].avg,
                mean_error_all['pck'].avg, mean_error_all['auc'].avg))
            logging.info("{0:<12} {1:>6.2f} {2:>10.2f} {3:>10.2f} {4:>10.2f}".format(
                "Average", mean_error_all['p1'].avg, mean_error_all['p2'].avg,
                mean_error_all['pck'].avg, mean_error_all['auc'].avg))
        else:
            print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg, \
                mean_error_all['p2'].avg))

    if data_type.startswith('3dhp'):
        return mean_error_all['p1'].avg, mean_error_all['p2'].avg,  \
                mean_error_all['pck'].avg, mean_error_all['auc'].avg
    else:
        return mean_error_all['p1'].avg, mean_error_all['p2'].avg, 0, 0


def refine_model(model_refine, output_3D, input_2D, gt_3D, batch_cam):
    input_2D_single = input_2D[:, args.pad, :, :].unsqueeze(1)
    
    if output_3D.size(1) > 1:
        output_3D_single = output_3D[:, args.pad, :, :].unsqueeze(1)
    else:
        output_3D_single = output_3D

    if gt_3D.size(1) > 1:
        gt_3D_single = gt_3D[:, args.pad, :, :].unsqueeze(1)
    else:
        gt_3D_single = gt_3D

    uvd = torch.cat((input_2D_single, output_3D_single[:, :, :, 2].unsqueeze(-1)), -1)
    xyz = get_uvd2xyz(uvd, gt_3D_single, batch_cam) # B 1 J 3
    xyz[:, :, args.root_joint, :] = 0
    refine_out = model_refine(output_3D_single, xyz) # B 1 J 3

    return refine_out

if __name__ == '__main__':
    #  10 42 88 1000 1234 3407
    manualSeed = 1 # random.randint(1, 10000)

    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logtime = time.strftime('%y%m%d_%H%M_%S')
    args.create_time = logtime
    args.group = "V1"
    args.job_type = "1"
    args.filename = 'V1.1.1.1_trainview7_trainS1_testS1_' + args.create_time

    if args.create_file:
        # create backup folder
        if args.debug:
            args.checkpoint = './debug/' + logtime
        else:
            args.checkpoint = './test/' + logtime
            
        if not os.path.exists(args.checkpoint):
            os.makedirs(args.checkpoint)
 
        # backup python file
        import shutil
        file_name = os.path.basename(__file__)
        shutil.copyfile(src=file_name, dst = os.path.join( args.checkpoint, args.create_time + "_" + file_name))
        shutil.copyfile(src="common/arguments.py", dst = os.path.join( args.checkpoint, args.create_time + "_arguments.py"))
        # shutil.copyfile(src="model/model_Gaussian.py", dst = os.path.join( args.checkpoint, args.create_time + "_model_Gaussian.py"))
        shutil.copyfile(src="common/utils.py", dst = os.path.join( args.checkpoint, args.create_time + "_utils.py"))
        shutil.copyfile(src="test_3dhp.sh", dst = os.path.join( args.checkpoint, args.filename + "_test_3dhp.sh"))

        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(args.checkpoint, 'train.log'), level=logging.INFO)
        logging.info("Starting the record")
        logging.info("ALL: TS1-TS6")
        # Then, change to a format without timestamp
        logging.root.handlers[0].setFormatter(logging.Formatter('%(message)s'))

    if args.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                            filename=os.path.join(args.checkpoint, 'train.log'), level=logging.INFO)

    ## load data
    if args.dataset == 'h36m':
        dataset_path = args.root_path + 'data_3d_' + args.dataset + '.npz'
        dataset = Human36mDataset(dataset_path, args)
        actions = define_actions(args.actions)
    # elif args.dataset == 'humaneva15':
    #     from common.dataset.humaneva_dataset import HumanEvaDataset
    #     dataset_path = args.root_path + 'data_3d_' + args.dataset + '.npz'
    #     dataset = HumanEvaDataset(dataset_path, args)
    #     actions = define_actions_humaneva(args.actions)
    elif args.dataset.startswith('3dhp'):
        dataset_path = args.root_path + 'data_3d_' + args.dataset + '_' + args.keypoints + '.npz'
        print(dataset_path)
        dataset = Mpi_inf_3dhp_Dataset(dataset_path, args)

    if args.dataset.startswith('3dhp'):
        if args.train:
            train_data = Fusion_3dhp(args, dataset, args.root_path, train=True)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                        shuffle=True, num_workers=int(args.workers), pin_memory=True)
        test_data = Fusion_3dhp(args, dataset, args.root_path, train=False)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                        shuffle=False, num_workers=int(args.workers), pin_memory=True)
    else:
        if args.train:
            train_data = Fusion(args, dataset, args.root_path, train=True)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                        shuffle=True, num_workers=int(args.workers), pin_memory=True)
        test_data = Fusion(args, dataset, args.root_path, train=False)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                        shuffle=False, num_workers=int(args.workers), pin_memory=True)

    ## load model
    model = Model(args).cuda()
    model_refine = post_refine(args).cuda()

    ## Reload model
    if args.saved_model_path != '':
        # model_paths = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))
        # model_path = model_paths[0]
        # model_path = "checkpoint/231030_0033_11/Model_Gaussian_mu_p1_22_4952.pth"
        model_path = args.saved_model_path

        print(model_path)

        pre_dict = torch.load(model_path)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        print("model loaded successfully!")
        # Reload refine model
        # if args.refine_reload:
        #     # refine_path = model_paths[1]
        #     refine_path = "/home/xiu/codes/pose/Baseline/checkpoint/0921_0020_40/refine_9_4796.pth"
        #     print(refine_path)
        #     pre_dict_refine = torch.load(refine_path)
        #     refine_dict = model_refine.state_dict()
        #     state_dict = {k: v for k, v in pre_dict_refine.items() if k in refine_dict.keys()}
        #     refine_dict.update(state_dict)
        #     model_refine.load_state_dict(refine_dict)

    ## Optimizer
    lr = args.lr
    all_param = []
    if args.refine:
        all_param += list(model.parameters()) + list(model_refine.parameters())
    else:
        all_param += list(model.parameters())

    optimizer = optim.Adam(all_param, lr=lr, amsgrad=True)
    
    ##--------------------------------epoch-------------------------------- ##
    best_epoch = 0
    loss_epoches = []
    mpjpes = []

    for epoch in range(1, args.nepoch+1):
        ## train
        if args.train: 
            if args.dataset.startswith('3dhp'):
                actions = define_actions_3dhp(args.actions, 1)
            loss = train(train_dataloader, model, model_refine, optimizer)
            loss_epoches.append(loss * 1000)

        ## test
        if args.dataset.startswith('3dhp'):
            actions = define_actions_3dhp(args.actions, 0)

        ##--------------------------------epoch-------------------------------- ##

        # with torch.no_grad():
        p1_per_step, p2_per_step, pck_per_step, auc_per_step = test(actions, test_dataloader, model, model_refine)
        best_step = min(p1_per_step, key=p1_per_step.get)
        p1 = p1_per_step[best_step]
        p2 = p2_per_step[best_step]
        pck = pck_per_step[best_step]
        auc = auc_per_step[best_step]
        
        ## print
        if args.train:
            if args.dataset.startswith('3dhp'):
                logging.info('epoch: %d, lr: %.6f, loss: %.4f, p1: %.2f, p2: %.2f, pck: %.2f, auc: %.2f, %d: %.2f' % (epoch, lr, loss, p1, p2, pck, auc, best_epoch, args.previous_best))
                print('%d, lr: %.6f, loss: %.4f, p1: %.2f, p2: %.2f, pck: %.2f, auc: %.2f, %d: %.2f' % (epoch, lr, loss, p1, p2, pck, auc, best_epoch, args.previous_best))
            else:
                logging.info('epoch: %d, lr: %.6f, loss: %.4f, p1: %.2f, p2: %.2f, %d: %.2f' % (epoch, lr, loss, p1, p2, best_epoch, args.previous_best))
                print('%d, lr: %.6f, loss: %.4f, p1: %.2f, p2: %.2f, %d: %.2f' % (epoch, lr, loss, p1, p2, best_epoch, args.previous_best))
            
            ## adjust lr
            if epoch % args.lr_decay_epoch == 0:
                lr *= args.lr_decay_large
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay_large
            else:
                lr *= args.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay 
        else:
            if args.dataset.startswith('3dhp'):
                steps_sorted = sorted(p1_per_step.keys())
                step_strs = [
                    f"{s}_p1: {p1_per_step[s]:.4f}, {s}_p2: {p2_per_step[s]:.4f}, {s}_pck: {pck_per_step[s]:.4f}, {s}_auc: {auc_per_step[s]:.4f}"
                    for s in steps_sorted
                ]
                print('e: %d, p1: %.4f, p2: %.4f, pck: %.4f, auc: %.4f | %s' % (epoch, p1, p2, pck, auc, ' | '.join(step_strs)))
                logging.info('epoch: %d, p1: %.4f, p2: %.4f, pck: %.4f, auc: %.4f | %s' % (epoch, p1, p2, pck, auc, ' | '.join(step_strs)))
            