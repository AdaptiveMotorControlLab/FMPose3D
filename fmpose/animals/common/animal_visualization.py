import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

def save_3Dpose_colored(pre_pose, gt_pose, figure_name):
    fig = plt.figure()
    ax1 = fig.add_subplot(211, projection = '3d')
    ax1.scatter(pre_pose[:, 0], pre_pose[:, 1], pre_pose[:,2], c=list(range(pre_pose.shape[0])), cmap='jet')
    # plt.axis('off')
    ax2 = fig.add_subplot(212, projection = '3d')
    ax2.scatter(gt_pose[:, 0], gt_pose[:, 1], gt_pose[:,2], c=list(range(gt_pose.shape[0])), cmap='jet')
    # plt.axis('off')
    plt.show()
    plt.savefig(figure_name, dpi=400.0)
    plt.close()

def save_absolute_3Dpose_image(image, pre_pose, gt_pose, vid_3D, skeleton, figure_name):
    fig = plt.figure(figsize=(20, 9))
    ax1 = fig.add_subplot(131, projection = '3d')
    ax1.scatter(pre_pose[:, 0], pre_pose[:, 2], -pre_pose[:,1], c=list(range(pre_pose.shape[0])), cmap='jet')
    for i in range(skeleton.shape[0]):
        ax1.plot([pre_pose[skeleton[i,0], 0], pre_pose[skeleton[i, 1], 0]], 
                 [pre_pose[skeleton[i,0], 2], pre_pose[skeleton[i, 1], 2]], 
                 [-pre_pose[skeleton[i,0], 1], -pre_pose[skeleton[i, 1], 1]], 
                 c = 'black')
    ax1.set_xlim([-3,3])
    ax1.set_zlim([-1.5,3])
    ax1.set_ylim([12,20])
    ax1.title.set_text('Prediction')

    # plt.axis('off')
    ax2 = fig.add_subplot(132, projection = '3d')
    visiable_gt = gt_pose[np.where(vid_3D)[0],:]
    ax2.scatter(visiable_gt[:, 0], visiable_gt[:, 2], -visiable_gt[:,1], c=list(np.array(range(gt_pose.shape[0]))[np.where(vid_3D)]), cmap='jet')
    for i in range(skeleton.shape[0]):
        if vid_3D[skeleton[i,0]] > 0 and vid_3D[skeleton[i,1]] > 0:
            ax2.plot([gt_pose[skeleton[i,0], 0], gt_pose[skeleton[i, 1], 0]], 
                    [gt_pose[skeleton[i,0], 2], gt_pose[skeleton[i, 1], 2]], 
                    [-gt_pose[skeleton[i,0], 1], -gt_pose[skeleton[i, 1], 1]], 
                    c = 'black')
    ax2.set_xlim([-3,3])
    ax2.set_zlim([-1.5,3])
    ax2.set_ylim([12,20])
    ax2.title.set_text('GT')
    # plt.axis('off')
    ax3 = fig.add_subplot(133)
    ax3.imshow(image)
    ax3.title.set_text('Camera1 view')
    plt.show()
    plt.savefig(figure_name, dpi=200.0)
    plt.close()

def save_absolute_3Dpose(pre_pose, skeleton, figure_name):
    fig = plt.figure(figsize=(20, 9))
    ax1 = fig.add_subplot(111, projection = '3d')
    ax1.scatter(pre_pose[:, 0], pre_pose[:, 2], -pre_pose[:,1], c=list(range(pre_pose.shape[0])), cmap='jet')
    for i in range(skeleton.shape[0]):
        ax1.plot([pre_pose[skeleton[i,0], 0], pre_pose[skeleton[i, 1], 0]], 
                 [pre_pose[skeleton[i,0], 2], pre_pose[skeleton[i, 1], 2]], 
                 [-pre_pose[skeleton[i,0], 1], -pre_pose[skeleton[i, 1], 1]], 
                 c = 'black')
    ax1.set_xlim([-1,1])
    ax1.set_zlim([-1,1])
    ax1.set_ylim([-1,1])
    ax1.title.set_text('gt')
    plt.show()
    plt.savefig(figure_name, dpi=200.0)
    plt.close()