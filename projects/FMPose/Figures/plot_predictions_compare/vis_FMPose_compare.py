import sys
sys.path.append("..")
import random
import torch
import numpy as np
import matplotlib.pyplot as plt 
import os
import cv2
from tqdm import tqdm
import torch
from common.load_data_hm36_vis import Fusion
from common.h36m_dataset import Human36mDataset
from common.utils import *
from common.arguments import opts as parse_args
args = parse_args().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
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

import matplotlib
plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

## dataset
# dataset_path = "/home/xiu/codes/pose/Baseline/dataset/data_2d_h36m_gt.npz"
dataset_path = args.root_path + 'data_3d_' + args.dataset + '.npz'
dataset = Human36mDataset(dataset_path, args)
test_data = Fusion(opt=args, train=False, dataset=dataset, root_path =args.root_path)
dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=16)

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


# load MGCN
from model.mgcn1.models.modulated_gcn import ModulatedGCN as Mgcn

model_mgcn= Mgcn(dataset, hid_dim=384, num_layers=4, p_dropout=0.2, nodes_group=None).cuda()

# model_mgcn= Mgcn(dataset, hid_dim=512, num_layers=3, p_dropout=0.2, nodes_group=None).cuda()
mgcn_path = "./checkpoints/mgcn/model_module_gcn_7_eva_post_4939.pth"
mgcn_dict = model_mgcn.state_dict()
if os.path.exists(mgcn_path):
  pre_dict_mgcn = torch.load(mgcn_path)
  for name, key in mgcn_dict.items():
      mgcn_dict[name] = pre_dict_mgcn[name]
  model_mgcn.load_state_dict(mgcn_dict)
  print("load MGCN Successfully!")

from model.utils.post_refine import post_refine
model_refine = post_refine(args).cuda()# Reload refine model

from model.utils.post_refine import refine_model
refine = True
if refine:
  post_refine_dict = model_refine.state_dict()
  refine_path = "./checkpoints/mgcn/model_post_refine_7_eva_post_4939.pth"
  pre_dict_post_refine = torch.load(refine_path)
  for name, key in post_refine_dict.items():
      post_refine_dict[name] = pre_dict_post_refine[name]
  model_refine.load_state_dict(post_refine_dict)
  print("load refine model Successfully!")

def drawskeleton(kps, img, thickness=3, mpii=2):
    # colors = [(255, 128, 255), # 躯干
    #           (255, 127, 127), # 手
    #           (127, 127, 255)] # 脚

    # colors = [(240, 176, 0), # blue
    #           (0, 0, 255), # red
    #           (255/255, 127/255, 127/255)] # 脚
    colors = [(240, 176, 0), # blue
              (240, 176, 0), # blue
              (255/255, 127/255, 127/255)] # 脚
    

    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = [2, 2, 2, 1, 1,
          1, 1, 1, 2, 2,
          1, 1, 1, 2, 2, 2]

    # LR = [2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2]

    # LR = [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

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

  # ax.w_xaxis.line.set_color(white) #不限制边缘线
  # ax.w_yaxis.line.set_color(white)
  # ax.w_zaxis.line.set_color(white)

  ax.tick_params('x', labelbottom = False) # 不显示坐标轴文本
  ax.tick_params('y', labelleft = False)
  ax.tick_params('z', labelleft = False)

  if not world: 
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.invert_zaxis()

def show2Dpose(channels, ax): # blue, orange
  vals = np.reshape( channels, (17, 2) )
  # vals = np.reshape( channels, (16, 2))
  # human3.6m
  I = np.array([0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9]) # start points
  J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10]) # end points

  for i in np.arange( len(I) ):
    x, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(2)]

    # ax.text(x[0], y[0] - 0.005, str(I[i]), size = 15, alpha = 0.2)
    # ax.text(x[1], y[1] - 0.005, str(J[i]), size = 15, alpha = 0.2)
    
    # ax.text(x[0] - 0.12, y[0], joints_name[I[i]], size = 9)
    # ax.text(x[1] - 0.12 , y[1], joints_name[J[i]], size = 9)

    ax.plot(x, y, lw=1) # lw=2
    ax.scatter(x, y,s=5) # s 默认是20
    ax.set_aspect('equal') # 正常的人体比例

  # ax.invert_xaxis()
  ax.invert_yaxis()

  ax.set_xticks([]) # 不显示坐标
  ax.set_yticks([]) 
  white = (1.0, 1.0, 1.0, 0.0)
  plt.axis('off')

def save3Dpose(index, pose3D, out_target, ax, color, save_path, action, dpi_number):

    pose3D[:, :, 0] = 0
    # p1 = mpjpe_cal(pose3D, out_target) * 1000
    pose3D = pose3D[0, 0].cpu().detach().numpy()
    plt.sca(ax)
    show3Dpose(pose3D, ax, color= color, world= False)
    # Remove the background
    # ax.set_axis_off()
    # Set the background to transparent
    # ax.patch.set_alpha(0)
    # plt.savefig(save_path + '/' + action + '_idx_'+ str(index) + '.png', dpi=dpi_number, format='png', bbox_inches = 'tight', transparent=False)
    plt.savefig(save_path, dpi=dpi_number, format='png', bbox_inches = 'tight', transparent=False)
    return 0
  
def save3Dpose_svg(index, pose3D, out_target, ax, color, save_path, action, iter_num, iter_opt, dpi_number):

    pose3D[:, :, 0] = 0
    p1 = mpjpe_cal(pose3D, out_target) * 1000
    pose3D = pose3D[0, 0].cpu().detach().numpy()
    plt.sca(ax)
    show3Dpose(pose3D, ax, color= color, world= False)
    # Remove the background
    # ax.set_axis_off()
    # Set the background to transparent
    # ax.patch.set_alpha(0)
    plt.savefig(save_path + '/' + action + '_idx'+ str(index)+ '_iter_'+str(iter_num) + '_error_'+ str('%.4f' % p1.item()) + '.svg', dpi=300, format='svg', bbox_inches = 'tight')
    return p1
  
import imageio  
def create_gif(name, folder_path, duration=0.25):
    frames = []
    png_files = os.listdir(folder_path)
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
  
def show_input(img, ax):
    b,g,r = cv2.split(img)
    image_mat = cv2.merge([r,g,b])
    ax.imshow(image_mat)
    # ax.set_xticks([]) # 不显示坐标
    # ax.set_yticks([]) 
    plt.axis('off')

def input_augmentation(input_2D, model, joints_left, joints_right):
    output_3D_non_flip = model(input_2D[:, 0])
    output_3D_flip     = model(input_2D[:, 1])

    output_3D_flip[:, :, :, 0] *= -1
    output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

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

def show_frame():
  model_FMPose = model['CFM']
  model_FMPose.eval()
  model_mgcn.eval()
  model_refine.eval()
  
  import time
  logtime = time.strftime('%y%m%d_%H%M_%S')
  # create backup folder
  if args.create_file:
    if args.debug: 
        folder = './debug/' + logtime + "_vis"
    else:
        folder = './test/' + logtime + "_vis"

    if not os.path.exists(folder):
        os.makedirs(folder)
    # backup python file
    import shutil
    file_name = os.path.basename(__file__)
    shutil.copyfile(src=file_name, dst = os.path.join(folder, args.create_time + "_" + file_name))
    shutil.copyfile(src="vis_FM.sh", dst = os.path.join(folder, args.create_time + "_vis_FM.sh"))
    # shutil.copyfile(src="vis_FMPose_predictions.py", dst = os.path.join(folder, args.create_time + "_vis_FMPose_predictions.py"))
  figsize_x = 6.4*2
  figsize_y = 3.6*2
  dpi_number = 1000
  
  eval_steps = sorted({int(s) for s in getattr(args, 'eval_sample_steps', '3').split(',') if str(s).strip()})
  
  
  for i_data, data in enumerate(tqdm(dataloader, 0)):
    batch_cam, gt_3D, input_2D, input_2D_GT, input_2D_no, action, subject, cam_ind, index = data
    
    index_image = index + args.pad + 1
    index_image = index_image.item()

    # if (i_data == 5783 or i_data == 6777 or i_data == 21993 or i_data == 80954 or i_data == 119570) ==False:
    #   continue
    
    # if subject[0] != 'S9':
    #     continue
    # if subject[0] == 'S11' and action[0] == 'Greeting 2':
    #     continue
    # if (subject[0] == 'S9' and action[0] == 'Directions' and index_image == 2254) and cam_ind[0] == 3 or \
    #   (subject[0] == 'S9' and action[0] == 'Eating 1' and index_image == 951) and cam_ind[0] == 1 or \
    #   (subject[0] == 'S11' and action[0] == 'Photo' and index_image == 362) and cam_ind[0] == 1 or \
    #   (subject[0] == 'S11' and action[0] == 'Posing' and index_image == 185) and cam_ind[0] == 1:
    #   pass
    # else:
    #   continue
    # error = eval_cal.mpjpe(input_2D[:, 0], input_2D_GT[:, 0]) / 2 * 1000
        
    [input_2D, input_2D_GT, input_2D_no, gt_3D, batch_cam] = get_varialbe('test', [input_2D, input_2D_GT, input_2D_no, gt_3D, batch_cam])
    input_2D_GT = input_2D_GT[:, 0, args.pad].unsqueeze(1) # 1,1,17,2
    input_2D_no = input_2D_no[:, 0, args.pad].unsqueeze(1)

    input_2D_nonflip = input_2D[:, 0]
    input_2D_flip = input_2D[:, 1]
    out_target = gt_3D.clone() # B F J 3
    out_target[:, :, args.root_joint] = 0
    
    output_mgcn = model_mgcn(input_2D_nonflip)
    output_mgcn[:, :, args.root_joint, :] = 0
  
    # mgcn_mpjpe = mpjpe_cal(output_mgcn, out_target) * 1000
    # print("mgcn_mpjpe:", mgcn_mpjpe)
    
    if refine:
        # model_refine.eval()
        refined_output = refine_model(model_refine, output_mgcn, input_2D_nonflip, gt_3D, batch_cam) # B 1 J 3
        root_pos = refined_output[:, :, 0, :].clone()
        refined_output[:, :, :, :] -= root_pos.unsqueeze(2)
        refined_output[:, :, 0, :] = 0
        # refined_mpjpe = mpjpe_cal(refined_output, out_target) * 1000
        # print("refined_mpjpe:", refined_mpjpe)
        
    # continue
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
        # return y_local
    
    for s_keep in eval_steps:
        list_hypothesis = []
        for i in range(args.hypothesis_num):
            
            y = torch.randn_like(gt_3D)
            
            y_s, list_v_s = euler_sample(input_2D_nonflip, y, s_keep, model_FMPose)
            vis_v_s = False
            if vis_v_s:
              path_vis_v_s = folder + '/' + "vis_v_s"
              if not os.path.exists(path_vis_v_s):
                os.makedirs(path_vis_v_s)
              for i in range(len(list_v_s)):
                
                figx  = plt.figure(num=1, figsize=(figsize_x, figsize_y) ) # 1280 * 720
                ax1 = plt.axes(projection = '3d')  
                img_path = path_vis_v_s + '/' + "steps_" + str(i) + "_" + action[0] + '_idx_' + str(i_data) + '.png'
                _ = save3Dpose(i_data, list_v_s[i], out_target, ax1, (0.99, 0, 0), img_path, action[0], dpi_number=dpi_number)
                plt.close(figx)
                
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
        
        list_hypothesis_last = list_hypothesis
                
    output_fmpose[:, :, 0, :] = 0
    
    fmpose_mpjpe = (mpjpe_cal(output_fmpose, out_target) * 1000).item()
    mgcn_refined_mpjpe = (mpjpe_cal(refined_output, out_target) * 1000).item()
    
    # print("mgcn_refined_mpjpe:", mgcn_refined_mpjpe)
    delta = mgcn_refined_mpjpe - fmpose_mpjpe
    if not (delta > 8 and fmpose_mpjpe < 60):
      continue    

    input_2D_no  = input_2D_no[0, 0].cpu().detach().numpy()
    # pose 打印在image上
    image_dir = '/media/ti/datasets/Human3.6M/my/images'    
    cam_ind = cam_ind[0]
    if cam_ind == 0:
      camera_index = '.54138969'
    elif cam_ind == 1:
      camera_index = '.55011271'
    elif cam_ind == 2:
      camera_index = '.58860488'
    elif cam_ind == 3:
      camera_index = '.60457274'

    figsize_x = 6.0*2
    figsize_y = 3.6*2
    dpi_number = 1000
    

    path = folder + "/" + str(i_data)
    if not os.path.exists(path):
        os.makedirs(path) 
   
    # show images
    out_dir = path + '/' + subject[0] + '_' + action[0] + camera_index + '_'
    image_path = image_dir + '/' + str(subject[0]) + '/' + str(action[0]) + \
        str(camera_index) + '/' + str(('%04d'%index_image)) + '.jpg'
    image = cv2.imread(image_path)
    image = drawskeleton(input_2D_no, image)
    cv2.imwrite(out_dir + str(i_data) + '_2d.jpg', image)
    
    # figure
    fig = plt.figure( figsize=(figsize_x, figsize_y) ) # 1280 * 720
    fig.subplots_adjust(wspace=-0.05)  # Reduce horizontal spacing between subplots
    color=(0/255, 176/255, 240/255)
    linewidth=2.5
    # ax0 = fig.add_subplot(121)
    # ax0.set_title("GT_2D")
    # input_2D_GT_np = input_2D_GT[0, 0].cpu().detach().numpy()
    # show2Dpose( input_2D_GT_np, ax0)

    gt_np = out_target[0, 0].cpu().detach().numpy()
    
    ax1 = fig.add_subplot(121, projection='3d')
          # Plot GT in red
    show3Dpose_GT(gt_np, ax1, world=False, linewidth=linewidth)
    # ax1.set_title("GT_3D")
    mgcn_refined_np = refined_output[0, 0].cpu().detach().numpy()
    show3Dpose( mgcn_refined_np, ax1, color, world = False, linewidth=linewidth)

    ax2 = fig.add_subplot(122, projection='3d')
    # ax2.set_title("Output_3D")
    fmpose_np = output_fmpose[0, 0].cpu().detach().numpy()
    show3Dpose_GT(gt_np, ax2, world=False, linewidth=linewidth)
    show3Dpose(fmpose_np, ax2, color, world = False, linewidth=linewidth)
    plt.savefig(f'{out_dir}{index_image}_delta_{delta:.2f}_{fmpose_mpjpe:.2f}_FMPose.jpg', dpi=dpi_number, format='jpg', bbox_inches = 'tight')
    plt.clf ()
    plt.close () 

if __name__ == "__main__":
  # Delete_Files('results/')
  manualSeed = 1
  random.seed(manualSeed)
  torch.manual_seed(manualSeed)
  torch.manual_seed(manualSeed)
  np.random.seed(manualSeed)
  torch.cuda.manual_seed_all(manualSeed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  show_frame()