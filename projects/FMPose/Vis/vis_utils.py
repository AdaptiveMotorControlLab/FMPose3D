import numpy as np
import matplotlib.pyplot as plt
from IPython import embed
from common.utils import *

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
    #   x2, z2, y2 = [np.array( [vals[I[i], j]+np.random.random_sample()*0.01, vals[J[i], j]+np.random.random_sample()*0.01] ) for j in range(3)]

    ax.plot(x, y, z, lw=2.5, color = color)

  # draw dark markers at each joint for emphasis (respect axis ordering)
  if world:
    ax.scatter(vals[:,0], vals[:,1], vals[:,2], c='k', s=14, depthshade=False)
  else:
    ax.scatter(vals[:,0], vals[:,2], vals[:,1], c='k', s=14, depthshade=False)

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