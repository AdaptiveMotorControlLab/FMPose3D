import sys
sys.path.append("..")
import random
import torch
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from IPython import embed
import os
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
from common.load_data_hm36 import Fusion
from common.h36m_dataset import Human36mDataset
from common.utils import *
from model.model_GUMLP import Model as Gaussian
from common.arguments import opts
args = opts().parse()

import matplotlib
import matplotlib.pyplot as plot
plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

## dataset
dataset_path = args.root_path + 'data_3d_' + args.dataset + '.npz'
dataset = Human36mDataset(dataset_path, args)
test_data = Fusion(opt=args, train=False, dataset=dataset, root_path =args.root_path)
dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=16)

## model
if args.reload:
  model_gaussian = Gaussian(args).cuda()
  # Reloadmodel
  stgcn_dict = model_gaussian.state_dict()
  no_refine_path = "./data/230221_0853_48/Model_Gaussian_mu_p1_32_4920.pth"
  pre_dict_stgcn = torch.load(no_refine_path)
  for name, key in stgcn_dict.items():
      stgcn_dict[name] = pre_dict_stgcn[name]
  model_gaussian.load_state_dict(stgcn_dict)

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

def save3Dpose(index, pose3D, out_target, ax, color, save_path, action, iter_num, iter_opt, dpi_number):

    pose3D[:, :, 0] = 0
    p1 = mpjpe_cal(pose3D, out_target) * 1000
    pose3D = pose3D[0, 0].cpu().detach().numpy()
    plt.sca(ax)
    show3Dpose(pose3D, ax, color= color, world= False)
    # Remove the background
    # ax.set_axis_off()
    # Set the background to transparent
    # ax.patch.set_alpha(0)
    plt.savefig(save_path + '/' + action + '_idx_'+ str(index)+ '_iter_'+str(iter_num) + '_error_'+ str('%.2f' % p1.item()) + '.png', dpi=dpi_number, format='png', bbox_inches = 'tight', transparent=False)
    return p1
  
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

def show_frame():
    model_gaussian.eval()
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
        shutil.copyfile(src="run.sh", dst = os.path.join(folder, args.create_time + "_run.sh"))


    mean_3D_pose_tensor = torch.load(os.path.join('./dataset', 'mean_3D_pose.pth'))
    mean_3D_pose_tensor = mean_3D_pose_tensor.cuda()

    for i_data, data in enumerate(tqdm(dataloader, 0)):
        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, cam_ind = data
        
        index_image = index + args.pad + 1
        index_image = index_image.item()
            
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

        iter_opt = args.iter_opt_list[0]
        input_2D_nonflip = input_2D[:, 0]
        input_2D_flip = input_2D[:, 1]
        out_target = gt_3D.clone() # B F J 3
        out_target[:, :, args.root_joint] = 0

        steps = getattr(args, 'sample_steps', args.sample_steps)
        dt = 1.0 / steps
        y = mean_3D_pose_tensor.clone().to(device=gt_3D.device, dtype=gt_3D.dtype)
        y = y.expand_as(gt_3D)
        y[:, :, 0, :] = 0
        y_noise = torch.randn_like(y)
        y = y + y_noise


        # test only projection loss
        z_nonflip = nn.Parameter(input_2D_nonflip.clone(), requires_grad=True)
        opt_z = torch.optim.Adam([z_nonflip], lr = 0.001)
        for iter_num in range(iter_opt):
            output_3D_nonflip, _ = model_gaussian(z_nonflip)
            # cal p1
            if iter_num==0:
                output_3D_nonflip[:, :, args.root_joint] = 0
                error_0_P = mpjpe_cal(output_3D_nonflip, out_target) * 1000
            if iter_num==iter_opt-1: 
                output_3D_nonflip[:, :, args.root_joint] = 0
                error_last_P = mpjpe_cal(output_3D_nonflip, out_target) * 1000
                
            # loss proj 2d nonflip
            proj_gt_3D = gt_3D.clone()
            output_3D_nonflip[:,:,1:] += proj_gt_3D[:,:,:1] # 还原为绝对坐标
            output_3D_nonflip[:,:,:1] = proj_gt_3D[:,:,:1]
            proj_nonflip_2d = project_to_2d(output_3D_nonflip, batch_cam)
            loss_nonflip_proj = mpjpe_cal(proj_nonflip_2d, input_2D_nonflip)
            # nonflip 
            loss =  (loss_nonflip_proj)*1.0
            opt_z.zero_grad()
            loss.backward()
            opt_z.step()

        z_nonflip = nn.Parameter(input_2D_nonflip.clone(), requires_grad=True)
        opt_z = torch.optim.Adam([z_nonflip], lr = 0.001)
        for iter_num in range(iter_opt):
            output_3D_nonflip, _ = model_gaussian(z_nonflip)
            # cal p1
            if iter_num==0: 
                output_3D_nonflip[:, :, args.root_joint] = 0
                error_0_PU = mpjpe_cal(output_3D_nonflip, out_target) * 1000
            if iter_num==iter_opt-1: 
                output_3D_nonflip[:, :, args.root_joint] = 0
                error_last_PU = mpjpe_cal(output_3D_nonflip, out_target) * 1000
                
            # loss gaussian nonflip 
            output_3D_nonflip_gaussian = output_3D_nonflip.clone()
            # output_3D_nonflip_gaussian[:, :, 0] = 0 
            loss_nonflip_gaussian = gaussian_loss_opt(mu_nonflip_gt.detach(), s_nonflip_gt.detach(), output_3D_nonflip_gaussian)
            
            # loss proj 2d nonflip
            proj_gt_3D = gt_3D.clone()
            output_3D_nonflip[:,:,1:] += proj_gt_3D[:,:,:1] # 还原为绝对坐标
            output_3D_nonflip[:,:,:1] = proj_gt_3D[:,:,:1]
            proj_nonflip_2d = project_to_2d(output_3D_nonflip, batch_cam)
            loss_nonflip_proj = mpjpe_cal(proj_nonflip_2d, input_2D_nonflip)
            # nonflip 
            loss =  (loss_nonflip_proj)*1.0  + (loss_nonflip_gaussian) *0.005 # proj 0.0083  g:-2.94
            opt_z.zero_grad()
            loss.backward()
            opt_z.step()
            
        # delta_P = int(((error_0_P - error_last_P)).item())
        # delta_PU = int(((error_0_PU - error_last_PU)).item())
        delta_P = ((error_0_P - error_last_P)).item()
        delta_PU = ((error_0_PU - error_last_PU)).item()
        

        input_2D_no  = input_2D_no[0, 0].cpu().detach().numpy()
        # pose 打印在image上
        image_dir = '/data3/xiu/datasets/Human3.6M/my/images'    
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
        fig  = plt.figure(num=1, figsize=(figsize_x, figsize_y) ) # 1280 * 720
        path = folder + "/" + str(i_data)
        if not os.path.exists(path):
            os.makedirs(path) 
        # path_nonflip_PU_svg = path + '/' + "nonflip_PU_svg"
        path_nonflip_P = path + '/' + "nonflip_P"
        path_nonflip_PU = path + '/' + "nonflip_PU"
        # path_mix_z = path + '/' + "mix_z"
        path_list = [path_nonflip_P, path_nonflip_PU]
        for path1 in path_list:
            if not os.path.exists(path1):
                os.makedirs(path1)
      
        # show images
        out_dir = path + '/' + subject[0] + '_' + action[0] + camera_index + '_'
        image_path = image_dir + '/' + str(subject[0]) + '/' + str(action[0]) + \
            str(camera_index) + '/' + str(('%04d'%index_image)) + '.jpg'
        image = cv2.imread(image_path)
        image = drawskeleton(input_2D_no, image)
        cv2.imwrite(out_dir + str(i_data) + '_Error_Delta_' + str(delta_PU) + '_2d.png', image)
      
        color_list = [    (0.5019607843137255, 0.7294117647058824, 0.8784313725490196, 1.0),
                    (0.3764705882352941, 0.596078431372549, 0.8941176470588236, 1.0), 
                        (0.20915032679738566, 0.41830065359477125, 0.9150326797385621, 1.0),
                    (0.0, 0.19607843137254902, 0.9411764705882353, 1.0),
              (0.0, 0.06535947712418302, 0.5751633986928104, 1.0)
        ]
      #6
        # colors = [      (0.5019607843137255, 0.7294117647058824, 0.8784313725490196, 1.0),
        # (0.4310957324106113, 0.6264359861591696, 0.8097808535178778, 1.0),
        # (0.3582622068435217, 0.5205997693194926, 0.7392233756247597, 1.0),
        # (0.2873971549404075, 0.41762399077277973, 0.6705728565936179, 1.0),
        # (0.21456362937331797, 0.3117877739331027, 0.6000153787004998, 1.0),   # 5
        #             (0.0, 0.0, 0.39215686274509803, 1.0)
        # ]

        # figure~ show P
        fig2  = plt.figure(num=2, figsize=(figsize_x, figsize_y) ) # 1280 * 720
        ax1 = plt.axes(projection = '3d')  
        _ = save3Dpose(i_data, gt_3D.clone(), out_target, ax1, (0.99, 0, 0), path_nonflip_P, action[0], -1, iter_opt, dpi_number=dpi_number)
        
        z_nonflip = nn.Parameter(input_2D_nonflip.clone(), requires_grad=True)
        opt_z = torch.optim.Adam([z_nonflip], lr = 0.001)
        for iter_num in range(iter_opt):
            output_3D_nonflip, _ = model_gaussian(z_nonflip)
            # cal p1
            p1_nonflip_z = save3Dpose(i_data, output_3D_nonflip.clone(), out_target, ax1, color_list[iter_num], path_nonflip_P, action[0], iter_num, iter_opt, dpi_number=dpi_number)
            # loss proj 2d nonflip
            proj_gt_3D = gt_3D.clone()
            output_3D_nonflip[:,:,1:] += proj_gt_3D[:,:,:1] # 还原为绝对坐标
            output_3D_nonflip[:,:,:1] = proj_gt_3D[:,:,:1]
            proj_nonflip_2d = project_to_2d(output_3D_nonflip, batch_cam)
            loss_nonflip_proj = mpjpe_cal(proj_nonflip_2d, input_2D_nonflip)
            # nonflip and flip
            loss =  (loss_nonflip_proj)*1.0
            opt_z.zero_grad()
            loss.backward()
            opt_z.step()
            
        # figure~ show P+U
        fig3  = plt.figure(num=3, figsize=(figsize_x, figsize_y) ) # 1280 * 720
        ax2 = plt.axes(projection = '3d')  
        _ = save3Dpose(i_data, gt_3D.clone(), out_target, ax2, (0.99, 0, 0), path_nonflip_PU, action[0], -1, iter_opt, dpi_number=dpi_number)
        
        z_nonflip = nn.Parameter(input_2D_nonflip.clone(), requires_grad=True)
        opt_z = torch.optim.Adam([z_nonflip], lr = 0.001)
        for iter_num in range(iter_opt):
            output_3D_nonflip, _ = model_gaussian(z_nonflip)
            # cal p1
            p1_nonflip_z = save3Dpose(i_data, output_3D_nonflip.clone(), out_target, ax2, color_list[iter_num], path_nonflip_PU, action[0], iter_num, iter_opt, dpi_number=dpi_number)
            
            # loss gaussian nonflip 
            output_3D_nonflip_gaussian = output_3D_nonflip.clone() 
            loss_nonflip_gaussian = gaussian_loss_opt(mu_nonflip_gt.detach(), s_nonflip_gt.detach(), output_3D_nonflip_gaussian) 
            
            # loss proj 2d nonflip
            proj_gt_3D = gt_3D.clone()
            output_3D_nonflip[:,:,1:] += proj_gt_3D[:,:,:1] # 还原为绝对坐标
            output_3D_nonflip[:,:,:1] = proj_gt_3D[:,:,:1]
            proj_nonflip_2d = project_to_2d(output_3D_nonflip, batch_cam)
            loss_nonflip_proj = mpjpe_cal(proj_nonflip_2d, input_2D_nonflip)
            # nonflip and flip
            loss =  (loss_nonflip_proj)*1.0  + (loss_nonflip_gaussian) *0.005 # proj 0.0083  g:-2.94
            opt_z.zero_grad()
            loss.backward()
            opt_z.step()

        # create_gif(path_nonflip_P + '/' + action[0] +"_idx" + str(i_data)+ "_iter" + str(iter_num) + '.gif', folder_path=path_nonflip_P, duration=0.3)
        # create_gif(path_mix_z + '/' + action[0] +"_idx" + str(i)+ "_iter" + str(iter_num) + '.gif', folder_path=path_mix_z, duration=0.25)
        create_gif(path_nonflip_P + '/' + action[0] +"_idx" + str(index_image)+ "_iter" + str(iter_num) + '.gif', folder_path=path_nonflip_PU, duration=0.3)
        create_gif(path_nonflip_PU + '/' + action[0] +"_idx" + str(index_image)+ "_iter" + str(iter_num) + '.gif', folder_path=path_nonflip_PU, duration=0.3)
            
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