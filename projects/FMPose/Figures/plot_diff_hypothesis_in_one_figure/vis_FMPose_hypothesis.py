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
    shutil.copyfile(src="vis_FM_multi_hypothesis.sh", dst = os.path.join(folder, args.create_time + "_vis_FM_multi_hypothesis.sh"))
    shutil.copyfile(src="vis_FMPose_hypothesis.py", dst = os.path.join(folder, args.create_time + "_vis_FMPose_hypothesis.py"))
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
        
        output_3D_s = aggregate_hypothesis(list_hypothesis)
        list_hypothesis_last = list_hypothesis
        output_3D_s_last = output_3D_s
                

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
    # path_nonflip_PU_svg = path + '/' + "nonflip_PU_svg"
    # path_nonflip_P = path + '/' + "nonflip_P"
    # path_nonflip_PU = path + '/' + "nonflip_PU"
    # # path_mix_z = path + '/' + "mix_z"
    # path_list = [path_nonflip_P, path_nonflip_PU]
    # for path1 in path_list:
    #     if not os.path.exists(path1):
    #         os.makedirs(path1)
   
    # show images
    out_dir = path + '/' + subject[0] + '_' + action[0] + camera_index + '_'
    image_path = image_dir + '/' + str(subject[0]) + '/' + str(action[0]) + \
        str(camera_index) + '/' + str(('%04d'%index_image)) + '.jpg'
    image = cv2.imread(image_path)
    image = drawskeleton(input_2D_no, image)
    cv2.imwrite(out_dir + str(i_data) + '_2d.png', image)


    # figure
    fig2  = plt.figure(num=2, figsize=(figsize_x, figsize_y) ) # 1280 * 720
    ax1 = plt.axes(projection = '3d')  

    gt_vis = gt_3D[:, args.pad].unsqueeze(1).clone()
    gt_vis[:, :, 0, :] = 0
    gt_np = gt_vis[0, 0].cpu().detach().numpy()
    show3Dpose_GT(gt_np, ax1, world=False, linewidth=1.0)


    # Overlay: all hypotheses (gray), aggregated (blue), GT (red)
    if 'list_hypothesis_last' in locals() and 'output_3D_s_last' in locals():
      print("list_hypothesis_last:", len(list_hypothesis_last))
      num_h = len(list_hypothesis_last)
      for idx, hypo in enumerate(list_hypothesis_last):
        hypo_vis = hypo.clone()
        hypo_vis[:, :, 0, :] = 0
        pose_np = hypo_vis[0, 0].cpu().detach().numpy()
        if num_h > 1:
          shade = 0.35 + 0.45 * (idx / (num_h - 1))
        else:
          shade = 0.6
        show3Dpose(pose_np, ax1, color=(shade, shade, shade), world=False, linewidth=1.0)

      agg_vis = output_3D_s_last.clone()
      agg_vis[:, :, 0, :] = 0
      agg_np = agg_vis[0, 0].cpu().detach().numpy()
      show3Dpose(agg_np, ax1, color=(0/255, 176/255, 240/255), world=False, linewidth=1.0)

      overlay_path = os.path.join(path, action[0] + '_idx_' + str(i_data) + '_overlay.png')
      plt.savefig(overlay_path, dpi=dpi_number, format='png', bbox_inches='tight', transparent=False)
    
    # _ = save3Dpose(i_data, gt_3D.clone(), out_target, ax1, (0.99, 0, 0), path_nonflip_P, action[0], dpi_number=dpi_number)
    

    # # create_gif(path_nonflip_P + '/' + action[0] +"_idx" + str(i_data)+ "_iter" + str(iter_num) + '.gif', folder_path=path_nonflip_P, duration=0.3)
    # # create_gif(path_mix_z + '/' + action[0] +"_idx" + str(i)+ "_iter" + str(iter_num) + '.gif', folder_path=path_mix_z, duration=0.25)
    # create_gif(path_nonflip_P + '/' + action[0] +"_idx" + str(index_image)+ "_iter" + str(iter_num) + '.gif', folder_path=path_nonflip_PU, duration=0.3)
        
    plt.clf () #清除当前图形及其所有轴，但保持窗口打开，以便可以将其重新用于其他绘图。
    plt.close () #完全关闭图形窗口
 
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