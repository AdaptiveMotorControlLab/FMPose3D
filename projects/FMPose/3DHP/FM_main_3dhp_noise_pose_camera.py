import os
import time
import torch
import random
import logging
import matplotlib
import numpy as np
from tqdm import tqdm
matplotlib.use('Agg')
import torch.nn as nn
import torch.utils.data
from common.utils import *
import torch.optim as optim
from common.camera import *
from common.utils import *
import matplotlib.pyplot as plt
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

def cam_mm_to_pix(cam, cam_data):
    """
    Convert camera parameters from mm to pixels.
    
    Args:
        cam: [fx, fy, cx, cy, k1, k2, p1, p2, k3] in mm (torch.Tensor or list)
        cam_data: [width, height, sensorSize_x, sensorSize_y]
        
    Returns:
        cam: [fx, fy, cx, cy, k1, k2, p1, p2, k3] in pixels
    """
    # Convert to tensor if it's a list
    if isinstance(cam, list):
        cam = torch.tensor(cam, dtype=torch.float32)
    else:
        cam = cam.clone()
    
    # Calculate scaling factors (pixels per mm)
    mx = cam_data[0] / cam_data[2]  # width / sensorSize_x
    my = cam_data[1] / cam_data[3]  # height / sensorSize_y
    
    # Convert focal lengths
    cam[0] = cam[0] * mx  # fx in pixels
    cam[1] = cam[1] * my  # fy in pixels
    
    # Convert principal point (add image center offset)
    cam[2] = cam[2] * mx + cam_data[0] / 2  # cx in pixels
    cam[3] = cam[3] * my + cam_data[1] / 2  # cy in pixels
    
    # Distortion coefficients remain unchanged (dimensionless)
    # cam[4:9] stay the same
    
    return cam

def aggregate_hypothesis_camera_weight_3dhp(list_hypothesis, input_2D, gt_3D, cam_mm, cam_data, width, height, topk=3):
    """
    Camera-guided weighted aggregation for 3DHP with manual camera parameters.
    
    Args:
        list_hypothesis: list of (B,1,J,3) tensors (root-relative 3D poses)
        input_2D: (B, F, J, 2) 2D joints in normalized coordinates
        gt_3D: (B, F, J, 3) GT 3D for shape and root position
        cam_mm: camera parameters in mm
        cam_data: [width, height, sensorSize_x, sensorSize_y]
        width, height: image resolution
        topk: number of top hypotheses to aggregate
        
    Returns:
        (B,1,J,3) aggregated 3D pose with root at origin
    """
    if len(list_hypothesis) == 0:
        raise ValueError("list_hypothesis is empty")
    
    device = list_hypothesis[0].device
    dtype = list_hypothesis[0].dtype
    
    # Shapes
    B = gt_3D.size(0)
    J = gt_3D.size(2)
    F = gt_3D.size(1)
    assert F >= 1, "Expected at least one frame"
    
    # Stack hypotheses: (H,B,1,J,3) -> (B,H,J,3)
    stack = torch.stack(list_hypothesis, dim=0)  # (H,B,1,J,3)
    X_hbj3 = stack[:, :, 0, :, :]                # (H,B,J,3)
    X_bhj3 = X_hbj3.transpose(0, 1).contiguous() # (B,H,J,3)
    H = X_bhj3.size(1)
    
    # Convert camera params from mm to pixels
    cam_params_pix = cam_mm_to_pix(cam_mm, cam_data)
    cam_params_batch = cam_params_pix.unsqueeze(0).repeat(B, 1).to(device)
    
    # Target 2D at the same frame index as 3D (args.pad)
    target_2d_norm = input_2D[:, getattr(args, 'pad', 0)].contiguous()  # (B,J,2) normalized
    
    # Convert hypotheses from root-relative to absolute coordinates using GT root
    gt_root = gt_3D[:, getattr(args, 'pad', 0), 0, :].contiguous()  # (B,3)
    X_abs = X_bhj3.clone()
    X_abs[:, :, 1:, :] = X_abs[:, :, 1:, :] + gt_root.unsqueeze(1).unsqueeze(1)
    X_abs[:, :, 0, :] = gt_root.unsqueeze(1)
    
    # Vectorized projection for all hypotheses
    # (B,H,J,3) -> (B*H,J,3)
    X_flat = X_abs.view(B * H, J, 3)
    cam_rep = cam_params_batch.repeat_interleave(H, dim=0)  # (B*H,9)
    
    # Project to 2D (pixel coordinates)
    proj2d_flat_pix = project_to_2d(X_flat, cam_rep)  # (B*H,J,2) in pixels
    
    # Normalize projected 2D to match input_2D coordinate space
    offset = torch.tensor([1, height / width], device=proj2d_flat_pix.device, dtype=proj2d_flat_pix.dtype)
    proj2d_flat_norm = proj2d_flat_pix / width * 2 - offset  # (B*H,J,2) normalized
    
    proj2d_bhj = proj2d_flat_norm.view(B, H, J, 2)
    
    # Per-hypothesis per-joint 2D error (both in normalized coordinates)
    diff = proj2d_bhj - target_2d_norm.unsqueeze(1)    # (B,H,J,2)
    dist = torch.norm(diff, dim=-1)                    # (B,H,J)
    
    # For root joint (0), set equal distances (we set root to 0 later anyway)
    dist[:, :, 0] = 0.0
    
    # Convert 2D losses to weights using exponential over top-k hypotheses
    k = max(1, min(topk, H))
    
    # top-k smallest distances along hypothesis dim
    topk_vals, topk_idx = torch.topk(dist, k=k, dim=1, largest=False)  # (B,k,J)
    
    # Exponential weighting
    temp = args.exp_temp
    max_safe_val = temp * 20
    topk_vals_clipped = torch.clamp(topk_vals, max=max_safe_val)
    exp_vals = torch.exp(-topk_vals_clipped / temp)
    exp_sum = exp_vals.sum(dim=1, keepdim=True)
    topk_weights = exp_vals / torch.clamp(exp_sum, min=1e-10)
    
    # Handle NaN
    nan_mask = torch.isnan(topk_weights).any(dim=1, keepdim=True)
    uniform_weights = torch.ones_like(topk_weights) / k
    topk_weights = torch.where(nan_mask.expand_as(topk_weights), uniform_weights, topk_weights)
    
    # Scatter back to full H with zeros elsewhere
    weights = torch.zeros_like(dist)  # (B,H,J)
    weights.scatter_(1, topk_idx, topk_weights)
    
    # Weighted sum of root-relative 3D hypotheses per joint
    weights_exp = weights.unsqueeze(-1)                     # (B,H,J,1)
    weighted_bj3 = torch.sum(X_bhj3 * weights_exp, dim=1)   # (B,J,3)
    
    # Assemble output (B,1,J,3) and enforce root at origin
    agg = weighted_bj3.unsqueeze(1).to(dtype=dtype)
    agg[:, :, 0, :] = 0
    return agg

# Support loading the model class from a specific file path if provided
Model = None
if getattr(args, 'model_path', ''):
    import importlib.util
    import pathlib
    model_abspath = os.path.abspath(args.model_path)
    module_name = pathlib.Path(model_abspath).stem
    spec = importlib.util.spec_from_file_location(module_name, model_abspath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    Model = getattr(module, 'Model')
else:
    # Fallback to original method if no model_path is provided
    exec('from model.' + args.model + ' import Model')

def test(actions, dataloader, model, model_refine, hypothesis_num=1):
    
    model.eval()

    error_sum_2d_in, error_sum_joints = AccumLoss(), AccumLoss()
  
    # for multi-step eval, maintain per-step accumulators across the whole split
    eval_steps = None
    action_error_sum_multi = None
    eval_steps = sorted({int(s) for s in getattr(args, 'eval_sample_steps', '3').split(',') if str(s).strip()})
    action_error_sum_multi = {s: define_error_list(actions) for s in eval_steps}
    
    print(f"\n{'='*80}")
    print(f"Testing with {hypothesis_num} hypothesis(es), eval_steps: {eval_steps}")
    print(f"{'='*80}\n")

    for i, data in enumerate(tqdm(dataloader, 0)):
        batch_cam, gt_3D, input_2D, input_2D_GT, action, subject, cam_ind = data # test  
        [input_2D, input_2D_GT, gt_3D, batch_cam] = get_varialbe('test', [input_2D, input_2D_GT, gt_3D, batch_cam])

        input_2D_nonflip = input_2D[:, 0]
        input_2D_flip = input_2D[:, 1]
        
        # Select camera parameters based on test sequence
        # TS5 and TS6 are outdoor sequences using Camera 2
        # TS1-TS4 are indoor sequences using Camera 1
        if subject[0] == 'TS5' or subject[0] == 'TS6':
            # Camera 2: Outdoor (1920x1080, 16:9)
            cam_mm = [8.770747185, 8.770747185, -0.104908645, 0.104899704, 
                      -0.276859611, 0.131125256, -0.000360494, -0.001149441, -0.049318332]
            cam_data = [1920, 1080, 10, 5.625]  # [width, height, sensorSize_x, sensorSize_y]
            width, height = 1920, 1080
        else:
            # Camera 1: Indoor (2048x2048, 1:1)
            cam_mm = [7.32506, 7.32506, -0.0322884, 0.0929296, 0, 0, 0, 0, 0]
            cam_data = [2048, 2048, 10, 10]  # [width, height, sensorSize_x, sensorSize_y]
            width, height = 2048, 2048
        
        # Convert camera parameters from mm to pixels (for projection)
        cam_params_pix = cam_mm_to_pix(cam_mm, cam_data)
        
        # Prepare camera params tensor for batched projection
        # project_to_2d expects shape (N, 9)
        N = input_2D.size(0)
        cam_params_batch = cam_params_pix.unsqueeze(0).repeat(N, 1).to(gt_3D.device)
        
        # Optional: Test projection (only for first batch to verify camera setup)
        if i == 0:
            print(f"\n=== Projection Test (Batch {i}) ===")
            print(f"Resolution: {width}x{height}, Camera: {'2 (Outdoor)' if subject[0] in ['TS5', 'TS6'] else '1 (Indoor)'}")
            
            # Test: Project GT 3D to 2D and compare with input 2D
            proj_gt_3D = gt_3D.clone()
            output_3D_nonflip_test = gt_3D.clone()
            output_3D_nonflip_test[:,:,1:] += proj_gt_3D[:,:,:1]
            output_3D_nonflip_test[:,:,:1] = proj_gt_3D[:,:,:1]
            
            output_3D_flat = output_3D_nonflip_test.reshape(-1, output_3D_nonflip_test.shape[-2], 3)
            cam_params_flat = cam_params_batch.unsqueeze(1).repeat(1, output_3D_nonflip_test.shape[1], 1).reshape(-1, 9)
            
            proj_nonflip_2d_pix = project_to_2d(output_3D_flat, cam_params_flat)
            offset = torch.tensor([1, height / width], device=proj_nonflip_2d_pix.device, dtype=proj_nonflip_2d_pix.dtype)
            proj_nonflip_2d_norm = proj_nonflip_2d_pix / width * 2 - offset
            
            proj_nonflip_2d_norm = proj_nonflip_2d_norm.reshape(N, output_3D_nonflip_test.shape[1], -1, 2)
            input_2D_nonflip_reshaped = input_2D_nonflip.reshape(N, output_3D_nonflip_test.shape[1], -1, 2)
            
            loss_nonflip_proj = eval_cal.mpjpe(proj_nonflip_2d_norm, input_2D_nonflip_reshaped)
            print(f"Reprojection Error (normalized): {loss_nonflip_proj.item():.6f}")
            print("="*60 + "\n")
        

        out_target = gt_3D.clone()
        out_target[:, :, args.root_joint] = 0
        
        # Simple Euler sampler for CFM at test time (independent runs per step if eval_multi_steps)
        def euler_sample(x2d, y_local, steps, model_3d):
            dt = 1.0 / steps
            for s in range(steps):
                t_s = torch.full((gt_3D.size(0), 1, 1, 1), s * dt, device=gt_3D.device, dtype=gt_3D.dtype)
                v_s = model_3d(x2d, y_local, t_s)
                y_local = y_local + dt * v_s
            return y_local

        # for each requested step count, run an independent sampling
        if i == 0:  # Only print once
            print(f"eval_steps: {eval_steps}, hypothesis_num: {hypothesis_num}")
        
        for s_keep in eval_steps:
            list_hypothesis = []
            
            for h in range(hypothesis_num):
                # Generate hypothesis from noise
                y = torch.randn_like(gt_3D)
                y_s = euler_sample(input_2D_nonflip, y, s_keep, model)
                
                # Add flip augmentation hypothesis if enabled
                if args.test_augmentation_flip_hypothesis:
                    y_flip = torch.randn_like(gt_3D)
                    y_flip_s = euler_sample(input_2D_flip, y_flip, s_keep, model)
                    y_flip_s[:, :, :, 0] *= -1
                    y_flip_s[:, :, args.joints_left + args.joints_right, :] = \
                        y_flip_s[:, :, args.joints_right + args.joints_left, :]
                    y_flip_s_frame = y_flip_s[:, args.pad].unsqueeze(1)
                    y_flip_s_frame[:, :, 0, :] = 0
                    list_hypothesis.append(y_flip_s_frame)
                
                # Add non-flip hypothesis
                y_s_frame = y_s[:, args.pad].unsqueeze(1)
                y_s_frame[:, :, 0, :] = 0
                list_hypothesis.append(y_s_frame)
            
            # Aggregate using camera-guided weighting
            output_3D_s = aggregate_hypothesis_camera_weight_3dhp(
                list_hypothesis, input_2D_nonflip, gt_3D, 
                cam_mm, cam_data, width, height, args.topk
            )

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
    
    # Use custom folder_name if provided, otherwise use timestamp
    if args.folder_name != '':
        folder_name = args.folder_name
    else:
        folder_name = logtime
    
    args.filename = 'V1.1.1.1_trainview7_trainS1_testS1_' + args.create_time

    if args.create_file:
        # create backup folder
        if args.test and args.saved_model_path != '':
            # For testing, create folder in the model's directory
            args.previous_dir = os.path.dirname(args.saved_model_path)
            args.checkpoint = os.path.join(args.previous_dir, folder_name)
        elif args.debug:
            args.checkpoint = './debug/' + folder_name
        else:
            args.checkpoint = './test/' + folder_name
            
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

        ##-------------------------------- Multi-Hypothesis Testing -------------------------------- ##
        
        # Parse hypothesis list and eval steps
        hypothesis_list = [int(x) for x in args.num_hypothesis_list.split(',')]
        eval_steps_list = [int(s) for s in str(getattr(args, 'eval_sample_steps', '3')).split(',') if str(s).strip()]
        
        # Track global best across all (step, hypothesis) combinations
        best_global_p1 = None
        best_global_p2 = None
        best_global_pck = None
        best_global_auc = None
        best_global_pair = None  # (step, hypothesis)
        
        for s_eval in eval_steps_list:
            p1_by_hyp = {}
            p2_by_hyp = {}
            pck_by_hyp = {}
            auc_by_hyp = {}
            
            for hypothesis_num in hypothesis_list:
                print(f"\n{'='*80}")
                print(f"Evaluating step {s_eval} with {hypothesis_num} hypotheses")
                print(f"{'='*80}\n")
                logging.info(f"Evaluating step {s_eval} with {hypothesis_num} hypotheses")
                
                with torch.no_grad():
                    # Test with specific step and hypothesis count
                    args_backup = args.eval_sample_steps
                    args.eval_sample_steps = str(s_eval)
                    
                    p1_per_step, p2_per_step, pck_per_step, auc_per_step = \
                        test(actions, test_dataloader, model, model_refine, hypothesis_num=hypothesis_num)
                    
                    args.eval_sample_steps = args_backup
                
                p1 = p1_per_step[int(s_eval)]
                p2 = p2_per_step[int(s_eval)]
                pck = pck_per_step[int(s_eval)]
                auc = auc_per_step[int(s_eval)]
                
                p1_by_hyp[int(hypothesis_num)] = float(p1)
                p2_by_hyp[int(hypothesis_num)] = float(p2)
                pck_by_hyp[int(hypothesis_num)] = float(pck)
                auc_by_hyp[int(hypothesis_num)] = float(auc)
                
                if best_global_p1 is None or float(p1) < best_global_p1:
                    best_global_p1 = float(p1)
                    best_global_p2 = float(p2)
                    best_global_pck = float(pck)
                    best_global_auc = float(auc)
                    best_global_pair = (int(s_eval), int(hypothesis_num))
            
            # Print one line per step with all hypotheses results
            hyp_sorted = sorted(p1_by_hyp.keys())
            hyp_strs = [
                f"h{h}_p1: {p1_by_hyp[h]:.4f}, h{h}_p2: {p2_by_hyp[h]:.4f}, h{h}_pck: {pck_by_hyp[h]:.4f}, h{h}_auc: {auc_by_hyp[h]:.4f}"
                for h in hyp_sorted
            ]
            print('\n' + '='*80)
            print(f'Step: {s_eval} | ' + ' | '.join(hyp_strs))
            print('='*80 + '\n')
            logging.info(f'step: {s_eval} | ' + ' | '.join(hyp_strs))
        
        # Print best results across all combinations
        if best_global_p1 is not None:
            print('\n' + '='*80)
            print(f'BEST RESULT: step {best_global_pair[0]}, hyp {best_global_pair[1]}: '
                  f'p1: {best_global_p1:.4f}, p2: {best_global_p2:.4f}, '
                  f'pck: {best_global_pck:.4f}, auc: {best_global_auc:.4f}')
            print('='*80 + '\n')
            logging.info(f'BEST: step {best_global_pair[0]}, hyp {best_global_pair[1]}: '
                        f'p1: {best_global_p1:.4f}, p2: {best_global_p2:.4f}, '
                        f'pck: {best_global_pck:.4f}, auc: {best_global_auc:.4f}')
        
        # Training logic (if training)
        if args.train:
            pass  # Add training save logic if needed
        else:
            break  # Exit after one epoch for testing
            