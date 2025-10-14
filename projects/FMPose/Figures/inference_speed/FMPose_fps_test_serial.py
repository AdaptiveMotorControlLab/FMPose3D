import sys
sys.path.append("..")
import random
import torch
import numpy as np
import os
import time
from tqdm import tqdm
from common.load_data_hm36_vis import Fusion
from common.h36m_dataset import Human36mDataset
from common.utils import *
from common.arguments import opts as parse_args

args = parse_args().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Load model
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

# Load dataset
dataset_path = args.root_path + 'data_3d_' + args.dataset + '.npz'
dataset = Human36mDataset(dataset_path, args)
test_data = Fusion(opt=args, train=False, dataset=dataset, root_path=args.root_path)
dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=8)

# Initialize model
model_FMPose = CFM(args).cuda()

if args.reload:
    model_dict = model_FMPose.state_dict()
    model_path = args.saved_model_path
    print(f"Loading model from: {model_path}")
    pre_dict = torch.load(model_path)
    for name, key in model_dict.items():
        model_dict[name] = pre_dict[name]
    model_FMPose.load_state_dict(model_dict)
    print("Model loaded successfully!")

model_FMPose.eval()

def aggregate_hypothesis(list_hypothesis):
    return torch.mean(torch.stack(list_hypothesis), dim=0)

def test_fps():
    """
    Test inference FPS by measuring the time from receiving 2D input 
    to generating final 3D pose prediction.
    """
    print("\n" + "="*60)
    print("FMPose Inference Speed Test")
    print("="*60)
    print(f"Batch size: {args.batch_size}")
    print(f"Hypothesis number: {args.hypothesis_num}")
    print(f"Test augmentation: {args.test_augmentation}")
    print(f"Sampling steps: {args.eval_sample_steps}")
    print("="*60 + "\n")
    
    # Get evaluation steps
    eval_steps = sorted({int(s) for s in getattr(args, 'eval_sample_steps', '3').split(',') if str(s).strip()})
    
    # Warm-up runs
    print("Warming up GPU...")
    warmup_iterations = 10
    with torch.no_grad():
        for i_data, data in enumerate(dataloader):
            if i_data >= warmup_iterations:
                break
            
            batch_cam, gt_3D, input_2D, input_2D_GT, input_2D_no, action, subject, cam_ind, index = data
            [input_2D, input_2D_GT, input_2D_no, gt_3D, batch_cam] = get_varialbe('test', [input_2D, input_2D_GT, input_2D_no, gt_3D, batch_cam])
            
            input_2D_nonflip = input_2D[:, 0]
            input_2D_flip = input_2D[:, 1]
            
            # Simple Euler sampler for CFM
            def euler_sample(x2d, y_local, steps, model_3d):
                dt = 1.0 / steps
                for s in range(steps):
                    t_s = torch.full((gt_3D.size(0), 1, 1, 1), s * dt, device=gt_3D.device, dtype=gt_3D.dtype)
                    v_s = model_3d(x2d, y_local, t_s)
                    y_local = y_local + dt * v_s
                return y_local
            
            for s_keep in eval_steps:
                list_hypothesis = []
                for i in range(args.hypothesis_num):
                    y = torch.randn_like(gt_3D)
                    y_s = euler_sample(input_2D_nonflip, y, s_keep, model_FMPose)
                    
                    output_3D_s = y_s[:, args.pad].unsqueeze(1)
                    output_3D_s[:, :, 0, :] = 0
                    list_hypothesis.append(output_3D_s)
                
                output_3D_s = aggregate_hypothesis(list_hypothesis)
    
    print("Warm-up complete!\n")
    
    # Actual FPS testing
    print("Starting FPS measurement...")
    num_test_samples = 200  # Test on 200 samples for reliable statistics
    inference_times = []
    
    with torch.no_grad():
        for i_data, data in enumerate(tqdm(dataloader, desc="Testing FPS", total=num_test_samples)):
            if i_data >= num_test_samples:
                break
            
            batch_cam, gt_3D, input_2D, input_2D_GT, input_2D_no, action, subject, cam_ind, index = data
            
            # Start timing after data is prepared
            [input_2D, input_2D_GT, input_2D_no, gt_3D, batch_cam] = get_varialbe('test', [input_2D, input_2D_GT, input_2D_no, gt_3D, batch_cam])
            
            input_2D_nonflip = input_2D[:, 0]
            input_2D_flip = input_2D[:, 1]
            
            # Synchronize GPU before starting timer
            torch.cuda.synchronize()
            start_time = time.time()
            
            # ============ INFERENCE STARTS HERE ============
            
            # Simple Euler sampler for CFM
            def euler_sample(x2d, y_local, steps, model_3d):
                dt = 1.0 / steps
                for s in range(steps):
                    t_s = torch.full((gt_3D.size(0), 1, 1, 1), s * dt, device=gt_3D.device, dtype=gt_3D.dtype)
                    v_s = model_3d(x2d, y_local, t_s)
                    y_local = y_local + dt * v_s
                return y_local
            
            for s_keep in eval_steps:
                list_hypothesis = []
                for i in range(args.hypothesis_num):
                    y = torch.randn_like(gt_3D)
                    y_s = euler_sample(input_2D_nonflip, y, s_keep, model_FMPose)
                    
                    output_3D_s = y_s[:, args.pad].unsqueeze(1)
                    output_3D_s[:, :, 0, :] = 0
                    list_hypothesis.append(output_3D_s)
                
                output_3D_s = aggregate_hypothesis(list_hypothesis)
            
            # ============ INFERENCE ENDS HERE ============
            
            # Synchronize GPU before stopping timer
            torch.cuda.synchronize()
            end_time = time.time()
            
            inference_time = end_time - start_time
            inference_times.append(inference_time)
    
    # Calculate statistics
    inference_times = np.array(inference_times)
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    median_time = np.median(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    mean_fps = 1.0 / mean_time * args.batch_size
    median_fps = 1.0 / median_time * args.batch_size
    
    print("\n" + "="*60)
    print("FPS Test Results")
    print("="*60)
    print(f"Total samples tested: {len(inference_times)}")
    print(f"Batch size: {args.batch_size}")
    print(f"\nInference Time Statistics (seconds):")
    print(f"  Mean:   {mean_time:.4f} ± {std_time:.4f}")
    print(f"  Median: {median_time:.4f}")
    print(f"  Min:    {min_time:.4f}")
    print(f"  Max:    {max_time:.4f}")
    print(f"\nFPS Statistics:")
    print(f"  Mean FPS:   {mean_fps:.2f}")
    print(f"  Median FPS: {median_fps:.2f}")
    print("="*60)
    
    # Save results to file
    results_file = f"fps_results_{args.sh_file.replace('.sh', '')}.txt"
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FMPose Inference Speed Test Results\n")
        f.write("="*60 + "\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model path: {args.model_path}\n")
        f.write(f"Saved model path: {args.saved_model_path}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Hypothesis number: {args.hypothesis_num}\n")
        f.write(f"Test augmentation: {args.test_augmentation}\n")
        f.write(f"Sampling steps: {args.eval_sample_steps}\n")
        f.write(f"GPU: {args.gpu}\n")
        f.write(f"Total samples tested: {len(inference_times)}\n")
        f.write("\n" + "-"*60 + "\n")
        f.write("Inference Time Statistics (seconds):\n")
        f.write(f"  Mean:   {mean_time:.4f} ± {std_time:.4f}\n")
        f.write(f"  Median: {median_time:.4f}\n")
        f.write(f"  Min:    {min_time:.4f}\n")
        f.write(f"  Max:    {max_time:.4f}\n")
        f.write("\nFPS Statistics:\n")
        f.write(f"  Mean FPS:   {mean_fps:.2f}\n")
        f.write(f"  Median FPS: {median_fps:.2f}\n")
        f.write("="*60 + "\n")
    
    print(f"\nResults saved to: {results_file}")
    
    return mean_fps, median_fps

if __name__ == "__main__":
    # Set random seeds for reproducibility
    manualSeed = 1
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.benchmark = True  # Enable for faster inference
    torch.backends.cudnn.deterministic = False  # Can disable for faster inference
    
    mean_fps, median_fps = test_fps()

