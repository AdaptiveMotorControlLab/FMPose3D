import sys
sys.path.append("..")
import random
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
dataloader = torch.utils.data.DataLoader(test_data, batch_size=2, shuffle=True, num_workers=16)

# count actions utility
def count_and_save_actions(loader, out_path='Vis/actions/test_actions_stats.json'):
    from collections import Counter
    counts = Counter()
    with torch.no_grad():
        for _, _, _, _, _, action, _, _, _ in tqdm(loader, 0):
            # Follow mpjpe_by_action_p1 pattern using num = len(action)
            action_list = list(action)
            num = len(action_list)
            if num == 1:
                full_name = action_list[0]
                end_index = full_name.find(' ')
                if end_index != -1:
                    base_name = full_name[:end_index]
                else:
                    base_name = full_name
                counts[base_name] += 1
            else:
                for i in range(num):
                    full_name = action_list[i]
                    end_index = full_name.find(' ')
                    if end_index != -1:
                        base_name = full_name[:end_index]
                    else:
                        base_name = full_name
                    counts[base_name] += 1

    import json
    import os
    import csv
    dir_path = os.path.dirname(out_path) or '.'
    os.makedirs(dir_path, exist_ok=True)

    stats = {
        'num_unique_actions': len(counts),
        'counts': dict(counts)
    }
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Saved actions stats to {out_path}. Unique actions: {len(counts)}")

    # also save CSV next to JSON
    csv_path = os.path.splitext(out_path)[0] + '.csv'
    with open(csv_path, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['action', 'count'])
        for k, v in counts.items():
            writer.writerow([k, v])
    print(f"Saved actions stats CSV to {csv_path}")

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

def show3Dpose_GT(channels, ax, world = True): # blue, orange
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

    ax.plot(x, y, z, lw=2.5, color = colors[LR[i]-1])
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

  # white = (1.0, 1.0, 1.0, 0.0)
  # ax.xaxis.set_pane_color(white) #不显示背景
  # ax.yaxis.set_pane_color(white)
  # ax.zaxis.set_pane_color(white)

  # ax.w_xaxis.line.set_color(white) #不限制边缘线
  # ax.w_yaxis.line.set_color(white)
  # ax.w_zaxis.line.set_color(white)
  
  ax.tick_params('x', labelbottom = False) # 不显示坐标轴文本
  ax.tick_params('y', labelleft = False)
  ax.tick_params('z', labelleft = False)

  if not world: 
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.invert_zaxis()

def show3Dpose(channels, ax, color, world = True): # blue, orange
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

    ax.plot(x, y, z, lw=2.5, color = color)

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

def show2Dpose(channels, ax, color): # blue, orange
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

    ax.plot(x, y, lw=4, color=color) # lw=2
    ax.scatter(x, y, s=5, color=color) # s 默认是20
    ax.set_aspect('equal') # 正常的人体比例

  # ax.invert_xaxis()
  ax.invert_yaxis()

  ax.set_xticks([]) # 不显示坐标
  ax.set_yticks([]) 
  white = (1.0, 1.0, 1.0, 0.0)
  plt.axis('off')


def show3Dpose_no_bg(channels, ax, color, world = True): # blue, orange
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

    ax.plot(x, y, z, lw=2.5, color = color)

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
  
  # Fully disable background for 3D pose
  ax.set_axis_off()
  ax.set_facecolor('none')
    
def save3Dpose(index, pose3D, out_target, ax, color, save_path, action, dpi_number):

    pose3D[:, :, 0] = 0
    # p1 = mpjpe_cal(pose3D, out_target) * 1000
    pose3D = pose3D[0, 0].cpu().detach().numpy()
    plt.sca(ax)
    show3Dpose_no_bg(pose3D, ax, color= color, world= False)
    # Remove the background
    # ax.set_axis_off()
    # Set the background to transparent
    # ax.patch.set_alpha(0)
    # plt.savefig(save_path + '/' + action + '_idx_'+ str(index) + '.png', dpi=dpi_number, format='png', bbox_inches = 'tight', transparent=True)
    ax.set_axis_off()
    ax.set_facecolor('none')
    ax.get_figure().patch.set_alpha(0.0)
    plt.savefig(save_path, dpi=dpi_number, format='png', bbox_inches = 'tight', transparent=True)
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

def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

def aggregate_hypothesis(list_hypothesis):
    return torch.mean(torch.stack(list_hypothesis), dim=0)


def show_frame():
  
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
    shutil.copyfile(src="run_FM_vis.sh", dst = os.path.join(folder, args.create_time + "_run_FM_vis.sh"))

  figsize_x = 6.4*2
  figsize_y = 3.6*2
  dpi_number = 1000
  
  
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

    
    [input_2D, input_2D_GT, input_2D_no, gt_3D, batch_cam] = get_varialbe('test', [input_2D, input_2D_GT, input_2D_no, gt_3D, batch_cam])
    input_2D_GT = input_2D_GT[:, 0, args.pad].unsqueeze(1) # 1,1,17,2
    input_2D_no = input_2D_no[:, 0, args.pad].unsqueeze(1)

    input_2D_nonflip = input_2D[:, 0]
    input_2D_flip = input_2D[:, 1]
    out_target = gt_3D.clone() # B F J 3
    out_target[:, :, args.root_joint] = 0
     

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


    # save 2D pose with transparent background
    fig_2d, ax_2d = plt.subplots(figsize=(figsize_x, figsize_y))
    color = (0/255, 176/255, 240/255)
    show2Dpose(input_2D_no, ax_2d, color)
    ax_2d.set_axis_off()
    fig_2d.patch.set_alpha(0.0)
    ax_2d.set_facecolor('none')
    plt.savefig(out_dir + str(i_data) + '_2d_transparent.png', dpi=dpi_number, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig_2d)

    plt.clf () #清除当前图形及其所有轴，但保持窗口打开，以便可以将其重新用于其他绘图。
    plt.close () #完全关闭图形窗口
    break
 

def plot_samples_per_action(loader, samples_root='samples', per_action=4, dpi=300):
  os.makedirs(samples_root, exist_ok=True)
  target_actions = define_actions('All')
  remaining = {a: per_action for a in target_actions}

  with torch.no_grad():
    for data in tqdm(loader, 0):
      batch_cam, gt_3D, input_2D, input_2D_GT, input_2D_no, action, subject, cam_ind, index = data
      [input_2D, input_2D_GT, input_2D_no, gt_3D, batch_cam] = get_varialbe('test', [input_2D, input_2D_GT, input_2D_no, gt_3D, batch_cam])

      print(batch_cam.shape)
      # break
    
      B = gt_3D.size(0)
      for b in range(B):
        full_action = action[b]
        if not isinstance(full_action, str):
          full_action = str(full_action)
        end_index = full_action.find(' ')
        base_action = full_action[:end_index] if end_index != -1 else full_action

        if base_action not in remaining or remaining[base_action] <= 0:
          continue

        subj = subject[b] if isinstance(subject[b], str) else str(subject[b])
        cam_id = int(cam_ind[b].item()) if hasattr(cam_ind[b], 'item') else int(cam_ind[b])
        if cam_id == 0:
          camera_index = '.54138969'
        elif cam_id == 1:
          camera_index = '.55011271'
        elif cam_id == 2:
          camera_index = '.58860488'
        else:
          camera_index = '.60457274'

        idx_image = int(index[b].item()) if hasattr(index[b], 'item') else int(index[b])
        idx_image = idx_image + args.pad + 1

        image_dir = '/media/ti/datasets/Human3.6M/my/images'
        image_path = os.path.join(image_dir, str(subj), full_action + camera_index, f"{idx_image:04d}.jpg")
        if not os.path.exists(image_path):
          continue
        img = cv2.imread(image_path)
        if img is None:
          continue

        # 2D pose for this sample/frame
        # input_2D_no: (B, 2, F, J, 2) -> [b, 0, pad] -> (J,2)
        try:
          kps2d = input_2D_no[b, 0, args.pad].detach().cpu().numpy()
        except Exception:
          continue
        img_overlay = drawskeleton(kps2d, img.copy())
        img_overlay_rgb = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)

        # 3D GT pose for the same sample/frame
        pose3d = gt_3D[b, args.pad].detach().cpu().numpy()  # (J,3)
        # set root joint to origin for visualization
        pose3d[args.root_joint] = 0

        # Create figure with two subplots (second is 3D)
        fig = plt.figure(figsize=(10, 4))
        ax_img = fig.add_subplot(1, 2, 1)
        ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
        ax_img.imshow(img_overlay_rgb)
        ax_img.axis('off')
        color = (0/255, 176/255, 240/255)
        show3Dpose(pose3d, ax_3d, color=color, world=False)
        fig.tight_layout()

        out_dir = os.path.join(samples_root, base_action)
        os.makedirs(out_dir, exist_ok=True)
        example_idx = per_action - remaining[base_action] + 1
        out_name = f"{base_action}_{example_idx:02d}.png"
        out_path = os.path.join(out_dir, out_name)
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        remaining[base_action] -= 1

      # stop early if all collected
      if all(v <= 0 for v in remaining.values()):
        break

  # also return a small report
  return {k: per_action - max(0, v) for k, v in remaining.items()}


if __name__ == "__main__":
#   # Delete_Files('results/')
  manualSeed = 1
  random.seed(manualSeed)
  torch.manual_seed(manualSeed)
  torch.manual_seed(manualSeed)
  np.random.seed(manualSeed)
  torch.cuda.manual_seed_all(manualSeed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  # Always compute and save action stats to Vis/actions before visualization
  # count_and_save_actions(dataloader)
  # show_frame()
  
  # Run the sampler to save figures under samples/
  plot_samples_per_action(dataloader, samples_root='./Vis/samples', per_action=4, dpi=300)