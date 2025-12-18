import os
import torch
import random
import logging
import datetime
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from fmpose.animals.common.arguments import opts as parse_args
from fmpose.animals.common.utils import *
from fmpose.animals.common.animal3d_dataset import TrainDataset
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
    
# wandb removed

def train(opt, actions, train_loader, model, optimizer, epoch):
    return step('train', opt, actions, train_loader, model, optimizer, epoch)

def val(opt, actions, val_loader, model, steps=None):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model, steps=steps)

def step(split, args, actions, dataLoader, model, optimizer=None, epoch=None, steps=None):

    loss_all = {'loss': AccumLoss()}
    
    model_3d = model['CFM']
    if split == 'train':
        model_3d.train()
    else:
        model_3d.eval()
        
    # determine steps for single-step evaluation per call
    steps_to_use = steps
    p1_error_sum = 0.
    p2_error_sum = 0.
    
    data_lent = 0

    for i, data in enumerate(tqdm(dataLoader, 0)):
        
        # batch_cam, gt_3D, input_2D, action, subject, cam_ind, vis_3D, start_3d, end_3d = data
        input_2D, gt_3D = data['keypoints_2d'], data['keypoints_3d'] 
        # print(input_2D.shape,input_2D)
        # print(gt_3D)
        #  input_2D shape: torch.Size([B, J, 2]) (normalized x,y coordinates)
        #  gt_3D shape: torch.Size([B, J, 4]) (x,y,z + homogeneous coordinate)
        gt_3D = gt_3D[:,:,:3]  # only use x,y,z for 3D ground truth
        
        # [input_2D, gt_3D, batch_cam, vis_3D] = get_varialbe(split, [input_2D, gt_3D, batch_cam, vis_3D])
        
        # unsqueeze frame dimension
        input_2D = input_2D.unsqueeze(1)  # (B,F,J,C)
        gt_3D = gt_3D.unsqueeze(1)  # (B,F,J,C)
        
        device = next(model_3d.parameters()).device

        model_dtype = next(model_3d.parameters()).dtype
        input_2D = input_2D.to(device=device, dtype=model_dtype)
        gt_3D = gt_3D.to(device=device, dtype=model_dtype)
        
        B = input_2D.shape[0]
        data_lent  += B
        
        if split =='train':
            B, F, J, C = input_2D.shape
            
            # Note: gt_3D is already root-relative from the dataloader
            # Root joint should already be [0,0,0]
            gt_3D = gt_3D.clone()
            gt_3D[:, :, args.root_joint] = 0
            
            
            # Conditional Flow Matching training
            # gt_3D, input_2D shape: (B,F,J,C)
            # vis_3D shape: (B,F,J,1) - visibility mask
            # x0_noise = torch.randn_like(gt_3D)
            x0_noise = torch.randn(B, F, J, 3, device=gt_3D.device, dtype=model_dtype)
            x0 = x0_noise
            
            B = gt_3D.size(0)
            # t on correct device/dtype and broadcastable: (B,1,1,1)
            t = torch.rand(B, 1, 1, 1, device=gt_3D.device, dtype=model_dtype)
            y_t = (1.0 - t) * x0 + t * gt_3D
            v_target = gt_3D - x0
            v_pred = model_3d(input_2D, y_t, t)

     
            loss = ((v_pred - v_target)**2).mean()
            
            N = input_2D.size(0)
            loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            # No test augmentation - use input_2D directly
            B, F, J, C = input_2D.shape
            
            # Note: gt_3D is already root-relative from the dataloader
            out_target = gt_3D.clone()
            out_target[:, :, args.root_joint] = 0

            # Simple Euler sampler for CFM at test time
            def euler_sample(x2d, y_local, steps_local):
                dt = 1.0 / steps_local
                for s in range(steps_local):
                    t_s = torch.full((gt_3D.size(0), 1, 1, 1), s * dt, device=gt_3D.device, dtype=model_dtype)
                    v_s = model_3d(x2d, y_local, t_s)
                    y_local = y_local + dt * v_s
                return y_local
            
            # Start from noise
            y = torch.randn(B, F, J, 3, device=gt_3D.device, dtype=model_dtype)
            
            # Run sampling
            y_s = euler_sample(input_2D, y, steps_to_use)
            output_3D = y_s[:, args.pad].unsqueeze(1)
            
            
            output_3D[:, :, args.root_joint, :] = 0
            
            p1 = mpjpe_cal(predicted = output_3D, target = out_target)
            # print("p1_error", p1)
            predicted = output_3D
            output_3D = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1]) # B,17,3
            target = out_target            
            out_target = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1]) # # B,17,3
            
            p2 = p_mpjpe(predicted = output_3D, target = out_target)
            p2 = np.mean(p2)
            # print("p2_error_sum", p2)
            
            p1_error_sum += p1 * B
            p2_error_sum += p2 * B

    if split == 'train':
        return loss_all['loss'].avg

    elif split == 'test':
        # aggregate metrics for the single requested steps
        # p1_s, p2_s = print_error(args.dataset, action_error_sum, args.train)
        p1_s = p1_error_sum / data_lent * 1000.0  # in mm
        p2_s = p2_error_sum / data_lent * 1000.0  # in mm
        
        print("mpjpe: {:.4f} mm".format(p1_s))
        print("p-mpjpe: {:.4f} mm".format(p2_s))

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
        # import shutil
        # file_path = os.path.abspath(__file__)
        # file_name = os.path.basename(file_path)
        # shutil.copyfile(src=file_path, dst=os.path.join(args.checkpoint, args.create_time + "_" + file_name))
        # shutil.copyfile(src=os.path.abspath("common/arguments.py"), dst=os.path.join(args.checkpoint, args.create_time + "_arguments.py"))
        # # backup the selected model file (from --model_path if provided)
        # if getattr(args, 'model_path', ''):
        #     model_src_path = os.path.abspath(args.model_path)
        #     model_dst_name = f"{args.create_time}_" + os.path.basename(model_src_path)
        #     shutil.copyfile(src=model_src_path, dst=os.path.join(args.checkpoint, model_dst_name))
        # # shutil.copyfile(src="common/utils.py", dst = os.path.join(args.checkpoint, args.create_time + "_utils.py"))
        # sh_base = os.path.basename(args.sh_file)
        # dst_name = f"{args.create_time}_" + sh_base
        # sh_src = os.path.abspath(args.sh_file)
        # shutil.copyfile(src=sh_src, dst=os.path.join(args.checkpoint, dst_name))

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

    
    # Support multiple training and test dataset paths from arguments
    train_paths = args.train_dataset_path if isinstance(args.train_dataset_path, list) else [args.train_dataset_path]
    test_paths = args.test_dataset_path if isinstance(args.test_dataset_path, list) else [args.test_dataset_path]

    # Rat7M doesn't have action labels, use placeholder for error calculation
    actions = ['rat_motion']

    if args.train:
        train_datasets = [TrainDataset(is_train=True, json_file=p, root_joint=args.root_joint) for p in train_paths]
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=int(args.workers), pin_memory=True)
    if args.test:
        test_datasets = [TrainDataset(is_train=False, json_file=p, root_joint=args.root_joint) for p in test_paths]
        test_dataset = torch.utils.data.ConcatDataset(test_datasets)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
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
        
        
        # evaluate per step externally (single-step val per call)
        p1_per_step = {}
        p2_per_step = {}
        eval_steps_list = [int(s) for s in str(getattr(args, 'eval_sample_steps', '3')).split(',') if str(s).strip()]
        for s_eval in eval_steps_list:
            p1_s, p2_s = val(args, actions, test_dataloader, model, steps=s_eval)
            p1_per_step[s_eval] = float(p1_s)
            p2_per_step[s_eval] = float(p2_s)
        best_step = min(p2_per_step, key=p2_per_step.get)
        p1 = p1_per_step[best_step]
        p2 = p2_per_step[best_step]
        
        if args.train:
            # Use P2 (P-MPJPE) as the metric for saving best models
            data_threshold = p2  # Changed from p1 to p2
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
    
    print("best epoch:{}, best result(p-mpjpe):{}".format(best_epoch, args.previous_best_threshold))
    logging.info("best epoch:{}, best result(p-mpjpe):{}".format(best_epoch, args.previous_best_threshold))
    print(h,"h",mins,"mins", s,"s")
    logging.info('training time: %dh,%dmin%ds' % (h, mins, s))
    print(args.checkpoint)
    logging.info(args.checkpoint)
    

