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
from common.load_data_rat7m import Rat7MFusion
from common.rat7m_dataset_ti import Rat7MDataset
from common.animal_visualization import save_absolute_3Dpose
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
    skeleton_mat = loadmat('/home/xiaohang/Ti_workspace/projects/FMPose_animals/dataset/rat7m/jesse_skeleton.mat')
    print("skeleton_mat:",skeleton_mat.keys(),skeleton_mat['joint_names'])
    skeleton = np.array(skeleton_mat['joints_idx'])-1
    print("skeleton:",skeleton.shape, skeleton)

    for i, data in enumerate(tqdm(dataLoader, 0)):
        batch_cam, gt_3D, input_2D, action, subject, cam_ind, vis_3D, start_3d, end_3d = data
        [input_2D, gt_3D, batch_cam, vis_3D] = get_varialbe(split, [input_2D, gt_3D, batch_cam, vis_3D])
        
        # Print frame range for tracking
        print(f"Batch {i}, subject: {subject[0]}, cam: {int(cam_ind[0])}, frame: {int(start_3d[0])}")
        
        # print("**********check gt_3d shape:", gt_3D.shape,gt_3D)  [1, 1, 20, 3]
        # break
        if i <=10:
            vis_savepath = args.checkpoint + f"/gt_3d_{i}.png"
            save_absolute_3Dpose(gt_3D[0,0,:,:].cpu().numpy(), skeleton , vis_savepath)
        
        
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

    if split == 'test':
        # aggregate metrics for the single requested steps
        p1_s, p2_s = print_error(args.dataset, action_error_sum, args.train)
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

    dataset = Rat7MDataset(dataset_path, args)
    
    # Rat7M doesn't have action labels, use placeholder for error calculation
    actions = ['rat_motion']
    
    # Verify dataset configuration
    print(f"Dataset: {args.dataset}")
    print(f"Train subjects: {dataset.train_list}")
    print(f"Test subjects: {dataset.test_list}")
    print(f"Train views: {args.train_views}")
    print(f"Test views: {args.test_views}")

    if args.train:
        train_data = Rat7MFusion(opt=args, train=True, dataset=dataset, root_path=root_path)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                       shuffle=True, num_workers=int(args.workers), pin_memory=True)
    if args.test:
        test_data = Rat7MFusion(opt=args, train=False, dataset=dataset, root_path=root_path)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
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
        
        steps_sorted = sorted(p1_per_step.keys())
        step_strs = [f"{s}_p1: {p1_per_step[s]:.4f}, {s}_p2: {p2_per_step[s]:.4f}" for s in steps_sorted]
        print('p1: %.4f, p2: %.4f | %s' % (p1, p2, ' | '.join(step_strs)))
        logging.info('p1: %.4f, p2: %.4f | %s' % (p1, p2, ' | '.join(step_strs)))
        break