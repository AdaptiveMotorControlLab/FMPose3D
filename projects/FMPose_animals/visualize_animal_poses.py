import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import pathlib
import sys

# Add paths if needed
sys.path.append("..")

from common.arguments import opts as parse_args
from common.utils import *
from common.animal3d_dataset_ti import TrainDataset

args = parse_args().parse()
args.n_joints = 26  # Set to match the dataset keypoints
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Support loading the model class from a specific file path if provided
CFM = None
if getattr(args, 'model_path', ''):
    import importlib.util
    model_abspath = os.path.abspath(args.model_path)
    module_name = pathlib.Path(model_abspath).stem
    spec = importlib.util.spec_from_file_location(module_name, model_abspath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    CFM = getattr(module, 'Model')

# Matplotlib backend
plt.switch_backend('agg')

# Functions for visualization
# def drawskeleton(kps, img, thickness=3):
#     colors = [(240, 176, 0), (240, 176, 0), (255/255, 127/255, 127/255)]
#     connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
#                    [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
#                    [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
#     LR = [2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2]
#     for j, c in enumerate(connections):
#         start = tuple(map(int, kps[c[0]]))
#         end = tuple(map(int, kps[c[1]]))
#         cv2.line(img, start, end, colors[LR[j]-1], thickness)
#         cv2.circle(img, start, thickness=-1, color=colors[LR[j]-1], radius=3)
#         cv2.circle(img, end, thickness=-1, color=colors[LR[j]-1], radius=3)
#     return img

def show3Dpose(channels, ax, color, world=True, linewidth=2.5):
    vals = np.reshape(channels, (26, 3))
    I = np.array([24, 24, 1, 0, 24, 2, 2, 24, 18, 18, 12, 13, 8, 9, 14, 15, 18, 7, 7, 10, 11, 16, 17, 7, 25])
    J = np.array([0, 1, 21, 21, 2, 22, 23, 18, 12, 13, 8, 9, 14, 15, 3, 4, 7, 10, 11, 16, 17, 5, 6, 25, 19])
    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=linewidth, color=color)
    RADIUS = 0.55
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)
    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)
    if not world:
        ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
        ax.invert_zaxis()

def show3Dpose_GT(channels, ax, world=True, linewidth=2.5):
    vals = np.reshape(channels, (26, 3))
    I = np.array([24, 24, 1, 0, 24, 2, 2, 24, 18, 18, 12, 13, 8, 9, 14, 15, 18, 7, 7, 10, 11, 16, 17, 7, 25])
    J = np.array([0, 1, 21, 21, 2, 22, 23, 18, 12, 13, 8, 9, 14, 15, 3, 4, 7, 10, 11, 16, 17, 5, 6, 25, 19])
    LR = [1, 2, 2, 1, 0, 1, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2, 0, 1, 2, 1, 2, 1, 2, 0, 0]
    colors = [(255/255, 0/255, 0/255), (255/255, 0/255, 0/255), (255/255, 0/255, 0/255)]
    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=linewidth, color=colors[LR[i]-1])
    RADIUS = 0.55
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)
    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)
    if not world:
        ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
        ax.invert_zaxis()

def show2Dpose(channels, ax):
    vals = np.reshape(channels, (26, 2))
    I = np.array([24, 24, 1, 0, 24, 2, 2, 24, 18, 18, 12, 13, 8, 9, 14, 15, 18, 7, 7, 10, 11, 16, 17, 7, 25])
    J = np.array([0, 1, 21, 21, 2, 22, 23, 18, 12, 13, 8, 9, 14, 15, 3, 4, 7, 10, 11, 16, 17, 5, 6, 25, 19])
    for i in np.arange(len(I)):
        x, y = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(2)]
        ax.plot(x, y, lw=1)
        ax.scatter(x, y, s=5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')

def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

def euler_sample(x2d, y_local, steps_local, model_3d):
    dt = 1.0 / steps_local
    for s in range(steps_local):
        t_s = torch.full((y_local.size(0), 1, 1, 1), s * dt, device=y_local.device, dtype=y_local.dtype)
        v_s = model_3d(x2d, y_local, t_s)
        y_local = y_local + dt * v_s
    return y_local

if __name__ == '__main__':
    manualSeed = 1
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load model
    model = {}
    model['CFM'] = CFM(args).cuda()

    if args.reload:
        model_dict = model['CFM'].state_dict()
        model_path = args.saved_model_path
        print(f"Loading model from {model_path}")
        pre_dict = torch.load(model_path)
        for name, key in model_dict.items():
            model_dict[name] = pre_dict[name]
        model['CFM'].load_state_dict(model_dict)
        print("Model loaded successfully!")

    # Load dataset
    test_paths = args.test_dataset_path if isinstance(args.test_dataset_path, list) else [args.test_dataset_path]
    test_datasets = [TrainDataset(is_train=False, json_file=p, root_joint=args.root_joint) for p in test_paths]
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=int(args.workers), pin_memory=True)

    # Create output folder
    import time
    logtime = time.strftime('%y%m%d_%H%M_%S')
    output_folder = f'./visualizations_{logtime}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model['CFM'].eval()

    eval_steps = 3  # Default steps, can be adjusted

    with torch.no_grad():
        for i_data, data in enumerate(tqdm(test_dataloader, desc="Visualizing")):
            input_2D = data['keypoints_2d']
            gt_3D = data['keypoints_3d']
            
            # Convert to numpy if tensor and squeeze batch dim
            if isinstance(input_2D, torch.Tensor):
                input_2D = input_2D.numpy()
                gt_3D = gt_3D.numpy()
            
            # Squeeze batch dim since batch_size=1
            input_2D = input_2D.squeeze(0)
            gt_3D = gt_3D.squeeze(0)
            
            gt_3D = gt_3D[:, :3]  # Only x,y,z

            # Pad to 26 joints if necessary
            J = input_2D.shape[0]
            target_J = 26
            if J < target_J:
                pad_size = target_J - J
                input_2D = np.pad(input_2D, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
                gt_3D = np.pad(gt_3D, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)

            input_2D = torch.from_numpy(input_2D).float()
            gt_3D = torch.from_numpy(gt_3D).float()

            input_2D = input_2D.unsqueeze(0).unsqueeze(0)  # (1,1,J,2)
            gt_3D = gt_3D.unsqueeze(0).unsqueeze(0)  # (1,1,J,3)

            device = next(model['CFM'].parameters()).device
            dtype = next(model['CFM'].parameters()).dtype
            input_2D = input_2D.to(device=device, dtype=dtype)
            gt_3D = gt_3D.to(device=device, dtype=dtype)

            out_target = gt_3D.clone()
            out_target[:, :, args.root_joint] = 0

            # Generate prediction
            y = torch.randn_like(gt_3D)
            y_s = euler_sample(input_2D, y, eval_steps, model['CFM'])
            output_3D = y_s
            output_3D[:, :, args.root_joint, :] = 0

            # Convert to numpy
            input_2d_np = input_2D[0, 0].cpu().numpy()
            gt_3d_np = out_target[0, 0].cpu().numpy()
            pred_3d_np = output_3D[0, 0].cpu().numpy()

            # Calculate error
            error = mpjpe_cal(output_3D, out_target).item() * 1000

            # Visualization
            fig = plt.figure(figsize=(15, 5))

            # 2D Pose
            ax1 = fig.add_subplot(131)
            ax1.set_title("2D Pose")
            show2Dpose(input_2d_np, ax1)

            # 3D GT
            ax2 = fig.add_subplot(132, projection='3d')
            ax2.set_title("3D GT Pose")
            show3Dpose_GT(gt_3d_np, ax2, world=False)

            # 3D Predicted
            ax3 = fig.add_subplot(133, projection='3d')
            ax3.set_title(f"3D Predicted Pose\nMPJPE: {error:.2f} mm")
            show3Dpose_GT(gt_3d_np, ax3, world=False)
            show3Dpose(pred_3d_np, ax3, color=(0/255, 176/255, 240/255), world=False)

            plt.tight_layout()
            plt.savefig(f'{output_folder}/frame_{i_data:04d}.png', dpi=300, bbox_inches='tight')
            plt.close()

            if i_data >= 99:  # Limit to first 100 for demo
                break

    print(f"Visualizations saved to {output_folder}")