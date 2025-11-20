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
    
    # ============ 调试开关 ============
    DEBUG_WEIGHTS = False  # 👈 设为 False 关闭调试输出
    if args.debug:
        DEBUG_WEIGHTS = True
    if DEBUG_WEIGHTS:
        # ============ 详细调试输出 ============
        print(f"\n{'='*60}")
        print(f"DEBUG: Weight Calculation Details")
        print(f"{'='*60}")
        print(f"tau = {tau}")
        print(f"k = {k}, H (total hypotheses) = {H}")
        
        # 检查 topk_vals 的统计信息
        print(f"\ntopk_vals statistics:")
        print(f"  mean: {topk_vals.mean().item():.6f}")
        # print(f"  std: {topk_vals.std().item():.6f}")
        print(f"  min: {topk_vals.min().item():.6f}")
        print(f"  max: {topk_vals.max().item():.6f}")
        
        # 查看具体的一个样本的一个关节
        b_sample, j_sample = 0, 5  # batch 0, joint 5
        if B > 0 and J > j_sample:
            print(f"\n示例: Batch {b_sample}, Joint {j_sample}")
            print(f"  topk_vals[{b_sample},:,{j_sample}] = {topk_vals[b_sample, :, j_sample].detach().cpu().numpy()}")
            print(f"  topk_idx[{b_sample},:,{j_sample}] = {topk_idx[b_sample, :, j_sample].detach().cpu().numpy()}")
            
            # 计算差异
            if k >= 2:
                diff_vals = topk_vals[b_sample, 1, j_sample] - topk_vals[b_sample, 0, j_sample]
                print(f"  差异 (第2小 - 第1小): {diff_vals.item():.6f}")
                print(f"  相对差异: {(diff_vals / (topk_vals[b_sample, 0, j_sample] + 1e-8)).item():.2%}")
        
        # 计算 softmax 输入
        softmax_input = -topk_vals / max(tau, 1e-6)
        print(f"\nsoftmax 输入 (-topk_vals / tau):")
        print(f"  mean: {softmax_input.mean().item():.6f}")
        print(f"  std: {softmax_input.std().item():.6f}")
        print(f"  range: [{softmax_input.min().item():.6f}, {softmax_input.max().item():.6f}]")
        
        if B > 0 and J > j_sample:
            print(f"  示例 softmax_input[{b_sample},:,{j_sample}] = {softmax_input[b_sample, :, j_sample].detach().cpu().numpy()}")
    
    # ========== 选择权重计算方法 ==========
    # 方法选择: 'softmax' | 'inverse' | 'hard' | 'exp'
    weight_method = 'exp'  # 👈 使用 inverse 方法（已添加 NaN 保护）

    if DEBUG_WEIGHTS:
        print(f"\n使用的权重计算方法: {weight_method}")
        b_sample, j_sample = 0, 5  # 用于调试输出的样本索引
    
    if weight_method == 'softmax':
        # 原始 softmax 方法
        softmax_input = -topk_vals / max(tau, 1e-6)
        topk_weights = torch.softmax(softmax_input, dim=1)
    elif weight_method == 'inverse':
        # 反比例权重 - 推荐！误差越小权重越大
        eps = 1e-6
        inv_weights = 1.0 / (topk_vals + eps)
        topk_weights = inv_weights / inv_weights.sum(dim=1, keepdim=True)
        if DEBUG_WEIGHTS:
            print(f"  inverse 方法计算详情:")
            if B > 0 and J > j_sample:
                print(f"    topk_vals[{b_sample},:,{j_sample}] = {topk_vals[b_sample, :, j_sample].detach().cpu().numpy()}")
                print(f"    inv_weights[{b_sample},:,{j_sample}] (归一化前: 1/topk_vals) = {inv_weights[b_sample, :, j_sample].detach().cpu().numpy()}")
                print(f"    inv_weights sum = {inv_weights[b_sample, :, j_sample].sum().item():.6f}")
                print(f"    topk_weights[{b_sample},:,{j_sample}] (归一化后) = {topk_weights[b_sample, :, j_sample].detach().cpu().numpy()}")
                print(f"    topk_weights sum = {topk_weights[b_sample, :, j_sample].sum().item():.6f} (应该 = 1.0)")
    elif weight_method == 'exp':
        # 指数权重 - 更激进，使用更小的温度参数
        # 温度越小，差异越大
        temp = args.exp_temp  # 👈 减小这个值会让差异更大
        
        # 防止数值下溢：clip topk_vals，避免 exp(-very_large/temp) -> 0
        # 如果 topk_val > temp * 20，exp(-topk_val/temp) < 2e-9，实际上权重为0
        max_safe_val = temp * 20  # 对应 exp(-20) ≈ 2e-9
        topk_vals_clipped = torch.clamp(topk_vals, max=max_safe_val)
        
        exp_vals = torch.exp(-topk_vals_clipped / temp)
        exp_sum = exp_vals.sum(dim=1, keepdim=True)
        
        # 避免除以零
        topk_weights = exp_vals / torch.clamp(exp_sum, min=1e-10)
        
        # 检查 NaN 并回退到均匀权重
        nan_mask = torch.isnan(topk_weights).any(dim=1, keepdim=True)  # (B,1,J)
        uniform_weights = torch.ones_like(topk_weights) / k
        topk_weights = torch.where(nan_mask.expand_as(topk_weights), uniform_weights, topk_weights)
        
        if DEBUG_WEIGHTS:
            print(f"  exp 方法计算详情 (temp={temp}):")
            if B > 0 and J > j_sample:
                print(f"    topk_vals[{b_sample},:,{j_sample}] = {topk_vals[b_sample, :, j_sample].detach().cpu().numpy()}")
                print(f"    topk_vals_clipped[{b_sample},:,{j_sample}] = {topk_vals_clipped[b_sample, :, j_sample].detach().cpu().numpy()}")
                print(f"    exp(-topk_vals_clipped/{temp})[{b_sample},:,{j_sample}] = {exp_vals[b_sample, :, j_sample].detach().cpu().numpy()}")
                if nan_mask[b_sample, 0, j_sample]:
                    print(f"    ⚠️  检测到 NaN，已回退到均匀权重")
    else:
        softmax_input = -topk_vals / max(tau, 1e-6)
        topk_weights = torch.softmax(softmax_input, dim=1)
    
    if DEBUG_WEIGHTS:
        # 检查权重分布
        print(f"\n最终 topk_weights (使用 {weight_method} 方法):")
        print(f"  mean: {topk_weights.mean().item():.6f}")
        print(f"  std: {topk_weights.std().item():.6f}")
        print(f"  理论均匀值 (1/k): {1.0/k:.6f}")
        
        if B > 0 and J > j_sample:
            print(f"  示例 topk_weights[{b_sample},:,{j_sample}] = {topk_weights[b_sample, :, j_sample].detach().cpu().numpy()}")
            
        # 检查有多少关节的权重接近均匀
        weight_diff = (topk_weights.max(dim=1)[0] - topk_weights.min(dim=1)[0])  # (B,J)
        near_uniform = (weight_diff < 0.1).float().mean()
        print(f"\n权重接近均匀分布的关节比例 (diff < 0.1): {near_uniform.item():.2%}")
        
        # 查看所有假设的原始距离（不只是 top-k）
        print(f"\n完整 dist 张量统计 (所有假设的2D误差):")
        print(f"  dist shape: {dist.shape}")
        print(f"  dist mean: {dist.mean().item():.6f}")
        print(f"  dist std: {dist.std().item():.6f}")
        
        if B > 0 and J > j_sample:
            print(f"  示例 dist[{b_sample},:,{j_sample}] (所有{H}个假设): {dist[b_sample, :, j_sample].detach().cpu().numpy()}")
        
        print(f"{'='*60}\n")

    # scatter back to full H with zeros elsewhere
    weights = torch.zeros_like(dist)  # (B,H,J)
    # weights.scatter_(1, topk_idx, topk_weights)
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
    shutil.copyfile(src="vis_intermediate_states_diff_S.sh", dst = os.path.join(folder, args.create_time + "_vis_intermediate_states_diff_S.sh"))
    # shutil.copyfile(src="vis_FMPose_hypothesis.py", dst = os.path.join(folder, args.create_time + "_vis_FMPose_hypothesis.py"))
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
    
    # Simple Euler sampler for CFM at test time (independent runs per step if eval_multi_steps)
    def euler_sample(x2d, y_local, steps, model_3d, save_intermediate=False):
        list_v_s = []
        list_y_s = []  # Store intermediate position states
        dt = 1.0 / steps
        for s in range(steps):
            t_s = torch.full((gt_3D.size(0), 1, 1, 1), s * dt, device=gt_3D.device, dtype=gt_3D.dtype)
            v_s = model_3d(x2d, y_local, t_s)
            if save_intermediate:
                # Save current position state before update
                list_y_s.append(y_local.clone())
            y_local = y_local + dt * v_s
            list_v_s.append(v_s)
        if save_intermediate:
            # Save final state
            list_y_s.append(y_local.clone())
        return y_local, list_v_s, list_y_s
    
    # Store intermediate RPEA states for each S
    intermediate_states_by_s = {}
    
    for s_keep in eval_steps:
        list_results = []
        list_intermediate_states = []  # Store intermediate states for this S

        y = torch.randn_like(gt_3D)
        # Save intermediate states for visualization
        y_s, list_v_s, list_y_s = euler_sample(input_2D_nonflip, y, s_keep, model_FMPose, save_intermediate=True)
        list_intermediate_states.append(list_y_s)
        # per-step metrics only; do not store per-sample outputs
        output_3D_s = y_s[:, args.pad].unsqueeze(1)
        output_3D_s[:, :, 0, :] = 0
        
        list_results.append(output_3D_s)
        
        # Store intermediate states for this S
        intermediate_states_by_s[s_keep] = list_intermediate_states
        
    # loss_RPEA = mpjpe_cal(output_3D_RPEA, out_target)*1000  
    # if loss_RPEA > 65:
    #   continue
    # print(f"loss_RPEA: {loss_RPEA.item():.2f}")
    
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

    figsize_x = 6.4*2
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
    fig2  = plt.figure(num=2, figsize=(figsize_x, figsize_y) ) # 1280 * 720
    ax1 = plt.axes(projection = '3d')  

    gt_vis = gt_3D[:, args.pad].unsqueeze(1).clone()
    gt_vis[:, :, 0, :] = 0
    gt_np = gt_vis[0, 0].cpu().detach().numpy()
    
    # Save intermediate RPEA states for each S
    for s_keep in eval_steps:
        if s_keep not in intermediate_states_by_s:
            continue
            
        # Create folder for this S
        path_intermediate_s = os.path.join(path, f'Step_{s_keep}')
        if not os.path.exists(path_intermediate_s):
            os.makedirs(path_intermediate_s)
        
        list_results_states = intermediate_states_by_s[s_keep]
        
        # Extract the list of intermediate states (only one hypothesis in this case)
        list_y_s = list_results_states[0]
        
        # Save each intermediate step
        for step_idx, y_step in enumerate(list_y_s):
            # Extract pose at pad position
            pose_step = y_step[:, args.pad].unsqueeze(1)
            pose_step[:, :, 0, :] = 0
            pose_step_np = pose_step[0, 0].cpu().detach().numpy()
            
            # Create figure
            fig_step = plt.figure(figsize=(figsize_x, figsize_y))
            ax_step = plt.axes(projection='3d')
            
            # Plot GT in red
            show3Dpose_GT(gt_np, ax_step, world=False, linewidth=1.5)
            
            # Plot intermediate state
            color = (0/255, 176/255, 240/255) # blue
            show3Dpose(pose_step_np, ax_step, color=color, world=False, linewidth=1.5)
            
            # Save figure
            step_path = os.path.join(path_intermediate_s, f'step_{step_idx:03d}.jpg')
            plt.savefig(step_path, dpi=dpi_number, format='jpg', bbox_inches='tight', transparent=False)
            plt.close(fig_step)

    plt.clf()  # Clear current figure and all axes
    plt.close()  # Close the figure window completely
 
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