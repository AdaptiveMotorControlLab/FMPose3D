import os
import torch
import random
import logging
import datetime
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from common.arguments import opts as parse_args
from common.utils import *
from common.animal3d_dataset import TrainDataset, EvaluationDataset
import time

args = parse_args().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Support loading the model class from a specific file path if provided
CFM = None
if getattr(args, 'model_path', ''):
    import importlib.util
    import pathlib
    model_abspath = os.path.abspath(args.model_path)
    module_name = pathlib.Path(model_abspath).stem
    spec = importlib.util.spec_from_file_location(module_name, model_abspath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    CFM = getattr(module, 'Model')
    
# wandb logging (assumed available)
import wandb
WANDB_AVAILABLE = False

def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)

def val(opt, actions, val_loader, model, steps=None):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model, steps=steps)

def step(split, args, actions, dataLoader, model, optimizer=None, epoch=None, steps=None):

    loss_all = {'loss': AccumLoss()}
    action_error_sum = define_error_list(actions)
    
    model_3d = model['CFM']
    if split == 'train':
        model_3d.train()
    else:
        model_3d.eval()
        
    # determine steps for single-step evaluation per call
    steps_to_use = steps

    for i, data in enumerate(tqdm(dataLoader, 0)):
        batch_cam, gt_3D, input_2D, action, subject, cam_ind, vis_3D, start_3d, end_3d = data
        [input_2D, gt_3D, batch_cam, vis_3D] = get_varialbe(split, [input_2D, gt_3D, batch_cam, vis_3D])
        
        if split =='train':
            B, F, J, C = input_2D.shape
            
            # Ensure root joint is zero in ground truth (should already be, but enforce it)
            gt_3D = gt_3D.clone()
            gt_3D[:, :, args.root_joint] = 0
            
            # Conditional Flow Matching training
            # gt_3D, input_2D shape: (B,F,J,C)
            # vis_3D shape: (B,F,J,1) - visibility mask
            # x0_noise = torch.randn_like(gt_3D)
            x0_noise = torch.randn(B, F, J, 3, device=gt_3D.device, dtype=gt_3D.dtype)
            x0 = x0_noise
            
            B = gt_3D.size(0)
            # t on correct device/dtype and broadcastable: (B,1,1,1)
            t = torch.rand(B, 1, 1, 1, device=gt_3D.device, dtype=gt_3D.dtype)
            y_t = (1.0 - t) * x0 + t * gt_3D
            v_target = gt_3D - x0
            v_pred = model_3d(input_2D, y_t, t)

            # Apply visibility mask to loss (only compute loss on visible joints)
            # vis_3D: [B, F, J, 1], expand to match v_pred: [B, F, J, 3]
            vis_mask = vis_3D.expand(-1, -1, -1, 3).clone()  # [B, F, J, 3]
            
            # Exclude root joint from loss (user preference + focus on relative positions)
            # Root has absolute camera position which is hard to predict from 2D alone
            vis_mask[:, :, args.root_joint, :] = 0
            
            masked_loss = ((v_pred - v_target)**2) * vis_mask
            
            # Compute mean only over valid (visible) elements
            num_valid = vis_mask.sum()  # Count visible elements
            if num_valid > 0:
                loss = masked_loss.sum() / num_valid
            else:
                # Fallback: if no visible joints (shouldn't happen), compute full loss
                loss = ((v_pred - v_target)**2).mean()
            
            N = input_2D.size(0)
            loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)
            if WANDB_AVAILABLE:
                # log per-batch training loss
                wandb.log({'train_loss': float(loss.detach().cpu().item()), 'epoch': epoch if epoch is not None else -1})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            # No test augmentation - use input_2D directly
            B, F, J, C = input_2D.shape
            out_target = gt_3D.clone()
            out_target[:, :, args.root_joint] = 0

            # Simple Euler sampler for CFM at test time
            def euler_sample(x2d, y_local, steps_local):
                dt = 1.0 / steps_local
                for s in range(steps_local):
                    t_s = torch.full((gt_3D.size(0), 1, 1, 1), s * dt, device=gt_3D.device, dtype=gt_3D.dtype)
                    v_s = model_3d(x2d, y_local, t_s)
                    y_local = y_local + dt * v_s
                return y_local
            
            # Start from noise
            y = torch.randn(B, F, J, 3, device=gt_3D.device, dtype=gt_3D.dtype)
            
            # Run sampling
            y_s = euler_sample(input_2D, y, steps_to_use)
            output_3D = y_s[:, args.pad].unsqueeze(1)
            
            
            output_3D[:, :, args.root_joint, :] = 0
            
            # Apply visibility mask for evaluation (only evaluate visible joints)
            # vis_3D: [B, F, J, 1], expand to match output_3D: [B, F, J, 3]
            vis_mask = vis_3D.expand(-1, -1, -1, 3).clone()  # [B, F, J, 3]
            
            # Exclude root joint from evaluation (consistent with training)
            # Since root is always 0, no need to evaluate it
            vis_mask[:, :, args.root_joint, :] = 0
            
            # Mask both prediction and target for fair comparison
            output_3D_masked = output_3D * vis_mask
            out_target_masked = out_target * vis_mask
            
            action_error_sum = test_calculation(output_3D_masked, out_target_masked, action, action_error_sum, args.dataset, subject, vis_mask)

    if split == 'train':
        return loss_all['loss'].avg

    elif split == 'test':
        # aggregate metrics for the single requested steps
        p1_s, p2_s = print_error(args.dataset, action_error_sum, args.train)
        if WANDB_AVAILABLE and steps_to_use is not None:
            wandb.log({f'test_p1_s{steps_to_use}': float(p1_s), f'test_p2_s{steps_to_use}': float(p2_s)})
        return float(p1_s), float(p2_s)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == '__main__':
    manualSeed = 1
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # allow overriding timestamp folder by user-provided folder_name
    logtime = time.strftime('%y%m%d_%H%M_%S')
    args.create_time = logtime

    if args.folder_name != '':
        folder_name = args.folder_name
    else:
        folder_name = logtime
    
    if WANDB_AVAILABLE:
        wandb.init(project=getattr(args, 'wandb_project', 'Pose3DCFM'),
                   name=f"CFM_Rat7M_{folder_name}",
                   config={k: getattr(args, k) for k in dir(args) if not k.startswith('_')})
     
    if args.create_file:
        # create backup folder
        if args.debug:
            args.checkpoint = './debug/' + folder_name
        elif args.train:
            args.checkpoint = './checkpoint/' + folder_name

        if args.train==False:
            # create a new folder for the test results
            args.folder_dir = os.path.dirname(args.saved_model_path)
            args.checkpoint = os.path.join(args.folder_dir, 'test_results_' + args.create_time)

        if not os.path.exists(args.checkpoint):
            os.makedirs(args.checkpoint)

        # backup files
        import shutil
        file_name = os.path.basename(__file__)
        shutil.copyfile(src=file_name, dst = os.path.join( args.checkpoint, args.create_time + "_" + file_name))
        shutil.copyfile(src="common/arguments.py", dst = os.path.join(args.checkpoint, args.create_time + "_arguments.py"))
        # backup the selected model file (from --model_path if provided)
        if getattr(args, 'model_path', ''):
            model_src_path = os.path.abspath(args.model_path)
            model_dst_name = f"{args.create_time}_" + os.path.basename(model_src_path)
            shutil.copyfile(src=model_src_path, dst=os.path.join(args.checkpoint, model_dst_name))
        # shutil.copyfile(src="common/utils.py", dst = os.path.join(args.checkpoint, args.create_time + "_utils.py"))
        sh_base = os.path.basename(args.sh_file)
        dst_name = f"{args.create_time}_" + sh_base
        shutil.copyfile(src=args.sh_file, dst=os.path.join(args.checkpoint, dst_name))

        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
            filename=os.path.join(args.checkpoint, 'train.log'), level=logging.INFO)
             
        arguments = dict((name, getattr(args, name)) for name in dir(args)
                if not name.startswith('_'))
        file_name = os.path.join(args.checkpoint, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> Args:\n')
            for k, v in sorted(arguments.items()):
                opt_file.write('  %s: %s\n' % (str(k), str(v)))
            opt_file.write('==> Args:\n')

    root_path = args.root_path
    dataset_path = root_path  # Directly use root_path for Rat7M
    
    # All Rat7M dataset configurations (n_joints, joints_left, joints_right, etc.) 
    # are set in arguments.py based on --dataset rat7m parameter
    
    
    dataset_path = '/home/xiaohang/Ti_workspace/projects/FMPose_animals/dataset/animal3d/'
    train_path = os.path.join(dataset_path, 'train_data.json')
    test_path = os.path.join(dataset_path, 'test_data.json')

    # Rat7M doesn't have action labels, use placeholder for error calculation
    actions = ['rat_motion']


    if args.train:
        train_data = TrainDataset(is_train=True, json_file=train_path)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=int(args.workers), pin_memory=True)
    if args.test:
        test_data = EvaluationDataset(is_train = False, json_file=test_path)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size,
                                                      shuffle=False, num_workers=int(args.workers), pin_memory=True)    

    model = {}
    model['CFM'] = CFM(args).cuda()

    if args.reload:
        model_dict = model['CFM'].state_dict()
        # Prefer explicit saved_model_path; otherwise fallback to previous_dir glob
        model_path = args.saved_model_path
        print(model_path)
        pre_dict = torch.load(model_path)
        for name, key in model_dict.items():
            model_dict[name] = pre_dict[name]
        model['CFM'].load_state_dict(model_dict)
        print("Load model Successfully!")

    all_param = []
    all_paramters = 0
    lr = args.lr
    all_param += list(model['CFM'].parameters())
    print(all_paramters)
    logging.info(all_paramters)
    optimizer = optim.Adam(all_param, lr=args.lr, amsgrad=True)
    
    starttime = datetime.datetime.now()
    best_epoch = 0
    
    for epoch in range(1, args.nepoch):
        if args.train:
            loss = train(args, actions, train_dataloader, model, optimizer, epoch)
            if WANDB_AVAILABLE:
                wandb.log({'train_loss_epoch': float(loss), 'epoch': epoch})
        
        # evaluate per step externally (single-step val per call)
        p1_per_step = {}
        p2_per_step = {}
        eval_steps_list = [int(s) for s in str(getattr(args, 'eval_sample_steps', '3')).split(',') if str(s).strip()]
        for s_eval in eval_steps_list:
            p1_s, p2_s = val(args, actions, test_dataloader, model, steps=s_eval)
            p1_per_step[s_eval] = float(p1_s)
            p2_per_step[s_eval] = float(p2_s)
        best_step = min(p1_per_step, key=p1_per_step.get)
        p1 = p1_per_step[best_step]
        p2 = p2_per_step[best_step]
        if WANDB_AVAILABLE:
            log_data = {'test_p1': p1, 'epoch': epoch}
            wandb.log(log_data)
        
        if args.train:
            data_threshold = p1
            saved_path = save_top_N_models(args.previous_name, args.checkpoint, epoch, data_threshold, model['CFM'], "CFM", num_saved_models=getattr(args, 'num_saved_models', 3))
            # update best tracker
            if data_threshold < args.previous_best_threshold:
                args.previous_best_threshold = data_threshold
                args.previous_name = saved_path
                best_epoch = epoch
                
            steps_sorted = sorted(p1_per_step.keys())
            step_strs = [f"{s}_p1: {p1_per_step[s]:.4f}, {s}_p2: {p2_per_step[s]:.4f}" for s in steps_sorted]
            print('e: %d, lr: %.7f, loss: %.4f, p1: %.4f, p2: %.4f | %s' % (epoch, lr, loss, p1, p2, ' | '.join(step_strs)))
            logging.info('epoch: %d, lr: %.7f, loss: %.4f, p1: %.4f, p2: %.4f | %s' % (epoch, lr, loss, p1, p2, ' | '.join(step_strs)))

        else:
            steps_sorted = sorted(p1_per_step.keys())
            step_strs = [f"{s}_p1: {p1_per_step[s]:.4f}, {s}_p2: {p2_per_step[s]:.4f}" for s in steps_sorted]
            print('p1: %.4f, p2: %.4f | %s' % (p1, p2, ' | '.join(step_strs)))
            logging.info('p1: %.4f, p2: %.4f | %s' % (p1, p2, ' | '.join(step_strs)))
            break
                
        if epoch % args.large_decay_epoch == 0: 
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay_large
                lr *= args.lr_decay_large
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= args.lr_decay
    
    endtime = datetime.datetime.now()   
    a = (endtime - starttime).seconds
    h = a//3600
    mins = (a-3600*h)//60
    s = a-3600*h-mins*60
    
    print("best epoch:{}, best result(mpjpe):{}".format(best_epoch, args.previous_best_threshold))
    logging.info("best epoch:{}, best result(mpjpe):{}".format(best_epoch, args.previous_best_threshold))
    print(h,"h",mins,"mins", s,"s")
    logging.info('training time: %dh,%dmin%ds' % (h, mins, s))
    print(args.checkpoint)
    logging.info(args.checkpoint)
    if WANDB_AVAILABLE:
        wandb.finish()

