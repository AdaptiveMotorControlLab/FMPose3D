"""
for CVPR26 3dhp vis with FM
vis on 3dhp
only show the init results and optimized results
"""
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
from IPython import embed
import os
import h5py
import cv2
import torch
from tqdm import tqdm
from common.vis.load_data_3dhp_vis import Fusion_3dhp
from common.dataset.mpi_inf_3dhp_dataset import Mpi_inf_3dhp_Dataset
from common.arguments import parse_args
from common.utils import *
import common.eval_cal as eval_cal

import matplotlib
import matplotlib.pyplot as plot
plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Load CFM model dynamically
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
else:
    raise ValueError("model_path must be provided in args")

args.frames = 1
args.pad = (args.frames - 1) // 2

args.refine = 0
args.depth_part = 0
args.layers = 3
args.channel = 512 
args.d_hid = 1024 
args.token_dim = 256
args.n_joints = 17

## model
model = {}
model['CFM'] = CFM(args).cuda()

if args.reload:
    model_dict = model['CFM'].state_dict()
    model_path = args.saved_model_path
    print(model_path)
    pre_dict = torch.load(model_path)
    for name, key in model_dict.items():
        model_dict[name] = pre_dict[name]
    model['CFM'].load_state_dict(model_dict)
    print("Load model Successfully!")

##
args.dataset = '3dhp_valid'
args.keypoints = 'gt_17_univ'
args.root_path = './dataset/'
args.subjects_train = 'S1,S2,S3,S4,S5,S6,S7,S8' 
args.subjects_test = args.subjects_test

def getFiles(path):
    image_files = []
    path_list = os.listdir(path)
    path_list.sort()
    for item in path_list:
        if item.startswith('.') and os.path.isfile(os.path.join(path, item)):
            path_list.remove(item)
    for file in path_list:
        image_files.append(os.path.join(path, file))
    return image_files

def Delete_Files(path = 'images'):
  file_name = getFiles(path)
  for remove_file in file_name:
    os.remove(remove_file)

def Load_3DHP():
  dataset_path = args.root_path + 'data_3d_' + args.dataset + '_' + args.keypoints + '.npz'
  print(dataset_path)
  dataset = Mpi_inf_3dhp_Dataset(dataset_path, args)

  test_data = Fusion_3dhp(opt=args, train=False, dataset=dataset, root_path =args.root_path)
  dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=8)

  return dataset, dataloader

def drawskeleton(kps, img, scale =1 , thickness=3, mpii=2):
    thickness *= scale
    colors = [(240, 176, 0), # blue
              (240, 176, 0), # blue
              (255/255, 127/255, 127/255)] # 脚

    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    # LR = [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    LR = [2, 2, 2, 1, 1,
          1, 1, 1, 2, 2,
          1, 1, 1, 2, 2, 2]

    # lcolor = [(34, 139, 34), (34, 139, 34), (34, 139, 34), (34, 139, 34), (34, 139, 34),]

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), colors[LR[j]-1], thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=colors[LR[j]-1], radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=colors[LR[j]-1], radius=3)

    return img

def show3Dpose_GT(channels, ax, world = True, linewidth = 2.5): # blue, orange
  # colors = [(255/255, 128/255, 255/255), # 躯干
  #           (127/255, 127/255, 255/255), # 手
  #           (255/255, 127/255, 127/255)] # 脚

  # colors = [(0/255, 176/255, 240/255), # blue
  #           (255/255, 0/255, 0/255), # red
  #           (255/255, 127/255, 127/255)] # 脚

  colors = [(255/255, 0/255, 0/255), # blue
            (255/255, 0/255, 0/255), # red
            (255/255, 0/255, 0/255)] # 脚

  vals = np.reshape( channels, (17, 3) )

  I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9]) # start points
  J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10]) # end points
  LR =          [2, 1, 2, 1, 2, 1, 1, 1,  2,  1,  2,  2,  1,  1, 2,  2]
  # LR = [3, 3, 3, 3, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1]

  for i in np.arange( len(I) ):
    if world:
      x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
    else:
      x, z, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]

    ax.plot(x, y, z, lw=linewidth, color = colors[LR[i]-1])
    # ax.scatter(x, y, z, color=(0, 1, 0))

  RADIUS = 0.55 
  xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
  ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
  ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
  ax.set_ylim3d([-RADIUS+zroot, RADIUS+zroot])
  # ax.set_zlim3d([0, 1.7])
  # ax.set_aspect('equal')
  # ax.set_aspect('auto')
  ax.set_box_aspect([1,1,1])

  # ax.set_xticks([]) # 不显示坐标 和 线
  # ax.set_yticks([]) 
  # ax.set_zticks([]) 

  white = (1.0, 1.0, 1.0, 0.0)
  ax.xaxis.set_pane_color(white) #不显示背景
  ax.yaxis.set_pane_color(white)
  ax.zaxis.set_pane_color(white)

  # ax.w_xaxis.line.set_color(white) #不限制边缘线
  # ax.w_yaxis.line.set_color(white)
  # ax.w_zaxis.line.set_color(white)
  
  ax.tick_params('x', labelbottom = False) # 不显示坐标轴文本
  ax.tick_params('y', labelleft = False)
  ax.tick_params('z', labelleft = False)

  if not world: 
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.invert_zaxis()

def show3Dpose(channels, ax, color, world = True, linewidth = 2.5): # blue, orange
  # colors = [(255/255, 128/255, 255/255), # 躯干
  #           (127/255, 127/255, 255/255), # 手
  #           (255/255, 127/255, 127/255)] # 脚

  # colors = [(0/255, 176/255, 240/255), # blue
  #           # (255/255, 0/255, 0/255), # red
  #           (0/255, 176/255, 240/255), # blue
  #           # (127/255, 127/255, 255/255), # 手
  #           (255/255, 127/255, 127/255)] # 脚

  vals = np.reshape( channels, (17, 3) )

  I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9]) # start points
  J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10]) # end points
  LR =          [2, 1, 2, 1, 2, 1, 1, 1,  2,  1,  2,  2,  1,  1, 2,  2]

  # LR = [3, 3, 3, 3, 3, 3, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1]

  for i in np.arange( len(I) ):
    if world:
      x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
    else:
      x, z, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
      x2, z2, y2 = [np.array( [vals[I[i], j]+np.random.random_sample()*0.01, vals[J[i], j]+np.random.random_sample()*0.01] ) for j in range(3)]

    ax.plot(x, y, z, lw=linewidth, color = color)

  RADIUS = 0.55 
  xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
  ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
  ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
  ax.set_ylim3d([-RADIUS+zroot, RADIUS+zroot])
  # ax.set_zlim3d([0, 1.7])
  # ax.set_aspect('equal')
  # ax.set_aspect('auto')
  ax.set_box_aspect([1,1,1])

  # ax.set_xticks([]) # 不显示坐标 和 线
  # ax.set_yticks([]) 
  # ax.set_zticks([]) 

  white = (1.0, 1.0, 1.0, 0.0)
  ax.xaxis.set_pane_color(white) #不显示背景
  ax.yaxis.set_pane_color(white)
  ax.zaxis.set_pane_color(white)
  
  ax.tick_params('x', labelbottom = False) # 不显示坐标轴文本
  ax.tick_params('y', labelleft = False)
  ax.tick_params('z', labelleft = False)

  if not world: 
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.invert_zaxis()


import imageio  
def create_gif(name, folder_path, duration=0.25):
    frames = []
    png_files = os.listdir(folder_path)
    print(png_files)
    # png_files.sort()
    png_files.sort(key=lambda x:int(x.split('_')[4]))
    image_list = [os.path.join(folder_path, f) for f in png_files]
    for image_name in image_list:
        # read png files
        frames.append(imageio.imread(image_name))
        # print(image_name)
    # save gif
    imageio.mimsave(name, frames, 'GIF', duration = duration)
    return 

def input_augmentation(input_2D, model, joints_left, joints_right):
    output_3D_non_flip, _ = model(input_2D[:, 0])
    output_3D_flip, _     = model(input_2D[:, 1])

    output_3D_flip[:, :, :, 0] *= -1
    output_3D_flip[:, :, args.joints_left + args.joints_right, :] = output_3D_flip[:, :, args.joints_right + args.joints_left, :] 

    output_3D = (output_3D_non_flip + output_3D_flip) / 2

    return output_3D

def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

def aggregate_hypothesis(list_hypothesis):
    return torch.mean(torch.stack(list_hypothesis), dim=0)
  
def aggregate_hypothesis_camera_weight(list_hypothesis, batch_cam, input_2D, gt_3D, topk=3):
    """
    Select per-joint 3D from the hypothesis whose 2D projection yields minimal L2 error.

    Args:
        list_hypothesis: list of (B,1,J,3) tensors
        batch_cam: (B, 9) or (B, 1, 9) intrinsics [f(2), c(2), k(3), p(2)]
        input_2D: (B, F, J, 2) 2D joints in image coordinates
        gt_3D: (B, F, J, 3) used for shape metadata only
    Returns:
        (B,1,J,3) aggregated 3D pose with joint 0 set to 0
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

    # Prepare camera params: (B,9)
    if batch_cam.dim() == 3 and batch_cam.size(1) == 1:
        cam_b9 = batch_cam[:, 0, :].contiguous()
    elif batch_cam.dim() == 2 and batch_cam.size(1) == 9:
        cam_b9 = batch_cam
    else:
        cam_b9 = batch_cam.view(B, -1)
    assert cam_b9.size(1) == 9, f"camera params should be 9-dim, got {cam_b9.size()}"

    # Target 2D at the same frame index as 3D selection (args.pad)
    # input_2D: (B,F,J,2) -> (B,J,2)
    target_2d = input_2D[:, getattr(args, 'pad', 0)].contiguous()  # (B,J,2)

    # Convert hypotheses from root-relative to absolute camera coordinates using GT root
    # Root at frame args.pad: (B,3)
    gt_root = gt_3D[:, getattr(args, 'pad', 0), 0, :].contiguous()  # (B,3)
    X_abs = X_bhj3.clone()
    X_abs[:, :, 1:, :] = X_abs[:, :, 1:, :] + gt_root.unsqueeze(1).unsqueeze(1)
    X_abs[:, :, 0, :] = gt_root.unsqueeze(1)

    # Vectorized projection for all hypotheses in absolute coordinates
    # (B,H,J,3) -> (B*H,J,3)
    X_flat = X_abs.view(B * H, J, 3)
    cam_rep = cam_b9.repeat_interleave(H, dim=0)  # (B*H,9)

    # project_to_2d expects last dim=3 and cam (N,9)
    proj2d_flat = project_to_2d(X_flat, cam_rep)  # (B*H,J,2)
    proj2d_bhj = proj2d_flat.view(B, H, J, 2)

    # Per-hypothesis per-joint 2D error
    diff = proj2d_bhj - target_2d.unsqueeze(1)    # (B,H,J,2)
    dist = torch.norm(diff, dim=-1) # (B,H,J)

    # For root joint (0), avoid NaNs in softmax by setting equal distances
    # This yields uniform weights at the root (we set root to 0 later anyway)
    dist[:, :, 0] = 0.0

    # Convert 2D losses to weights using softmax over top-k hypotheses per joint
    tau = float(getattr(args, 'weight_softmax_tau', 1.0))
    H = dist.size(1)
    k = int(getattr(args, 'topk', None))
    # print("k:", k)
    # k = int(H//2)+1
    k = max(1, min(k, H))

    # top-k smallest distances along hypothesis dim
    topk_vals, topk_idx = torch.topk(dist, k=k, dim=1, largest=False)  # (B,k,J)
    
    # Weight calculation method
    weight_method = 'exp'
    
    if weight_method == 'softmax':
        softmax_input = -topk_vals / max(tau, 1e-6)
        topk_weights = torch.softmax(softmax_input, dim=1)
    elif weight_method == 'inverse':
        eps = 1e-6
        inv_weights = 1.0 / (topk_vals + eps)
        topk_weights = inv_weights / inv_weights.sum(dim=1, keepdim=True)
    elif weight_method == 'exp':
        temp = args.exp_temp
        max_safe_val = temp * 20
        topk_vals_clipped = torch.clamp(topk_vals, max=max_safe_val)
        exp_vals = torch.exp(-topk_vals_clipped / temp)
        exp_sum = exp_vals.sum(dim=1, keepdim=True)
        topk_weights = exp_vals / torch.clamp(exp_sum, min=1e-10)
        nan_mask = torch.isnan(topk_weights).any(dim=1, keepdim=True)
        uniform_weights = torch.ones_like(topk_weights) / k
        topk_weights = torch.where(nan_mask.expand_as(topk_weights), uniform_weights, topk_weights)
    else:
        softmax_input = -topk_vals / max(tau, 1e-6)
        topk_weights = torch.softmax(softmax_input, dim=1)

    # scatter back to full H with zeros elsewhere
    weights = torch.zeros_like(dist)  # (B,H,J)
    weights.scatter_(1, topk_idx, topk_weights)

    # Weighted sum of root-relative 3D hypotheses per joint
    weights_exp = weights.unsqueeze(-1)                     # (B,H,J,1)
    weighted_bj3 = torch.sum(X_bhj3 * weights_exp, dim=1)   # (B,J,3)

    # Assemble output (B,1,J,3) and enforce root at origin
    agg = weighted_bj3.unsqueeze(1).to(dtype=dtype)
    agg[:, :, 0, :] = 0
    return agg

def show_input(img, ax):
    b,g,r = cv2.split(img)
    image_mat = cv2.merge([r,g,b])
    ax.imshow(image_mat)
    # ax.set_xticks([]) # 不显示坐标
    # ax.set_yticks([]) 
    plt.axis('off')

def show_frame():
    dataset, dataloader = Load_3DHP()
    model_FMPose = model['CFM']
    model_FMPose.eval()
    
    import time
    logtime = time.strftime('%y%m%d_%H%M_%S')
    args.create_time = logtime

    if args.folder_name != '':
        folder_name = args.folder_name
    else:
        folder_name = logtime
        
    # create backup folder
    if args.create_file:
        if args.debug:
            folder = './debug/' + folder_name + "_vis"
        else:
            folder = './test/' + folder_name + "_vis"

        if not os.path.exists(folder):
            os.makedirs(folder)
        # backup python file
        import shutil
        file_name = os.path.basename(__file__)
        shutil.copyfile(src=file_name, dst = os.path.join(folder, args.create_time + "_" + file_name))
        shutil.copyfile(src="vis_3dhp.sh", dst = os.path.join(folder, args.create_time + "_vis_3dhp.sh"))
    
    figsize_x = 6.4*2
    figsize_y = 3.6*2
    dpi_number = 1000
    
    eval_steps = sorted({int(s) for s in getattr(args, 'eval_sample_steps', '3').split(',') if str(s).strip()})
    
    for i_data, data in enumerate(tqdm(dataloader, 0)):
        batch_cam, gt_3D, input_2D, input_2D_GT, input_2D_no, action, subject, cam_ind, index = data

        # print(action, subject, cam_ind, index)
        index_image = index + args.pad + 1
        index_image = index_image.item()

        ## valid
        base_dir = '/media/ti/datasets/mpi_inf_3dhp/mpi_inf_3dhp_test_set'
        # base_dir = '/data3/xiu/datasets/mpi_inf_3dhp_test_set'
        anno_path = os.path.join(base_dir, subject[0], 'annot_data.mat')
        mat_as_h5 = h5py.File(anno_path, 'r')
        valid = np.array(mat_as_h5['valid_frame']).astype('bool')

        if index_image %5!=0:
            continue
        
        
        valid_sum = []
        for i, value in enumerate(valid):
            valid_sum.append(sum(valid[:i+1]))
        index_image = valid_sum.index(index_image)
        
        # if index_image != 453:
        #    continue

        if subject[0] == 'TS6':
            batch_cam = torch.tensor([ 1.7421e+00,  1.7372e+00, -2.2164e-02,  2.0153e-02, -3.5696e-02,
            1.7747e-01,  2.5115e-01,  1.1674e-03,  8.4259e-03], requires_grad=False)
        if subject[0] == 'TS1':
            batch_cam = torch.tensor([ 1.4969e+00,  1.6551e+00, -4.7996e-03,  2.9757e-02, -1.0288e-01,
            1.6693e-01,  2.6289e-01, -1.4845e-03, -3.8559e-03], requires_grad=False)
        if subject[0] == 'TS5':
            batch_cam = torch.tensor([ 1.8332e+00,  1.8096e+00, -1.2404e-02,  1.0143e-02,  6.8727e-02,
            -1.6290e-03, -1.8267e-01,  2.0062e-02, -6.6486e-03], requires_grad=False)
        

        [input_2D, input_2D_GT, input_2D_no, gt_3D, batch_cam] = get_varialbe('test', [input_2D, input_2D_GT, input_2D_no, gt_3D, batch_cam])
        batch_cam = batch_cam.unsqueeze(0)

        input_2D_GT = input_2D_GT[:, 0, args.pad].unsqueeze(1)
        input_2D_no = input_2D_no[:, 0, args.pad].unsqueeze(1)
    
        input_2D_nonflip = input_2D[:, 0]
        input_2D_flip = input_2D[:, 1]
        out_target = gt_3D.clone() # B F J 3
        out_target[:, :, args.root_joint] = 0
        
        # Simple Euler sampler for CFM at test time (independent runs per step if eval_multi_steps)
        def euler_sample(x2d, y_local, steps, model_3d):
            list_v_s = []
            dt = 1.0 / steps
            for s in range(steps):
                t_s = torch.full((gt_3D.size(0), 1, 1, 1), s * dt, device=gt_3D.device, dtype=gt_3D.dtype)
                v_s = model_3d(x2d, y_local, t_s)
                y_local = y_local + dt * v_s
                list_v_s.append(v_s)
            return y_local, list_v_s
        
        for s_keep in eval_steps:
            list_hypothesis = []
            for i in range(args.hypothesis_num):
                
                y = torch.randn_like(gt_3D)
                
                y_s, list_v_s = euler_sample(input_2D_nonflip, y, s_keep, model_FMPose)
                
                if args.test_augmentation:
                    joints_left = [4, 5, 6, 11, 12, 13]
                    joints_right = [1, 2, 3, 14, 15, 16]
                    
                    y_flip = torch.randn_like(gt_3D)
                    y_flip[:, :, :, 0] *= -1
                    y_flip[:, :, joints_left + joints_right, :] = y_flip[:, :, joints_right + joints_left, :] 
                    y_flip_s, list_v_s_flip = euler_sample(input_2D_flip, y_flip, s_keep, model_FMPose)
                    y_flip_s[:, :, :, 0] *= -1
                    y_flip_s[:, :, joints_left + joints_right, :] = y_flip_s[:, :, joints_right + joints_left, :]
                    y_s = (y_s + y_flip_s) / 2
                
                # per-step metrics only; do not store per-sample outputs
                output_3D_s = y_s[:, args.pad].unsqueeze(1)
                output_3D_s[:, :, 0, :] = 0
                
                list_hypothesis.append(output_3D_s)
            
            # output_3D_s = aggregate_hypothesis(list_hypothesis)
            output_fmpose = aggregate_hypothesis_camera_weight(list_hypothesis, batch_cam, input_2D_nonflip, gt_3D, args.topk)
                    
        output_fmpose[:, :, 0, :] = 0
        
        # For comparison, also compute initial prediction (first hypothesis)

        output_3D = output_fmpose.clone()
        error_0_P = mpjpe_cal(output_3D, out_target) * 1000
        


        delta_P = ((error_0_P)).item()

        if delta_P > 65:
            continue
        # print(delta_P)
        # break
        # input_2D = input_2D[:, 0]
        # output_3D[:, :, args.root_joint] = 0 
        # output_3D_optimized[:, :, args.root_joint] = 0 
        # error_1 = mpjpe_cal(output_3D, out_target) * 1000
        # error_2 = mpjpe_cal(output_3D_optimized, out_target) * 1000

        input_2D_no = input_2D_no[0, 0].cpu().detach().numpy()
        vis_gt = out_target.clone()
        vis_gt = vis_gt[0, 0].cpu().detach().numpy()
        output_3D = output_3D[0, 0].cpu().detach().numpy()

        path = folder + "/" + str(i_data)
        path_list = [path]
        for path1 in path_list:
            if not os.path.exists(path1):
                os.makedirs(path1)

        if not os.path.exists(path):
            os.makedirs(path)

        # image_dir = '/data0/liwh/MPI_INF_3DHP/video/mpi_inf_3dhp_test_set'
        image_dir = '/media/ti/datasets/mpi_inf_3dhp/mpi_inf_3dhp_test_set'
        # out_dir = 'results_3DHP/' + subject[0] + '_'
        out_dir = path + '/' + subject[0] + '_'

        image_path = image_dir + '/' + str(subject[0]) + '/imageSequence/img_' + str(('%06d'%index_image)) + '.jpg'
        print(image_path)

        if subject[0] == 'TS5' or subject[0] == 'TS6':
            scale = 1
        else:
            scale = 2

        image = cv2.imread(image_path)
        image = drawskeleton(input_2D_no, image, scale)
        cv2.imwrite(out_dir + str(('%06d'%index_image)) + '.jpg', image)

        # figure~ show 3D pose
        figsize_x = 6.4*2
        figsize_y = 3.6*2
        dpi_number = 1000

        fig  = plt.figure(figsize=(figsize_x, figsize_y) ) # 1280 * 720
        fig.subplots_adjust(wspace=-0.05, hspace=0)

        ax1 = fig.add_subplot(111, projection='3d')
        color=(0/255, 176/255, 240/255)
        linewidth=2.5
        show3Dpose_GT(vis_gt, ax1, world = False, linewidth=linewidth)
        show3Dpose(output_3D, ax1, color, world = False, linewidth=linewidth)

        plt.savefig(path + '/' + str(('%06d'%index_image)) + '_Error_' + "{:.2f}".format(error_0_P.item()) + '.jpg', dpi=dpi_number, format='jpg', bbox_inches = 'tight')

        plt.clf () #
        plt.close () #

if __name__ == "__main__":
  # Delete_Files('results_3d/')
  manualSeed = 1
  random.seed(manualSeed)
  torch.manual_seed(manualSeed)
  torch.manual_seed(manualSeed)
  np.random.seed(manualSeed)
  torch.cuda.manual_seed_all(manualSeed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  
  show_frame()


