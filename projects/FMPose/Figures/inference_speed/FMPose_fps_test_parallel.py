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
    # Optimize: directly stack and squeeze instead of multiple operations
    stack = torch.stack(list_hypothesis, dim=0)  # (H,B,1,J,3)
    X_bhj3 = stack.squeeze(2).transpose(0, 1)    # (B,H,J,3) - removed contiguous() call
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
    pad_idx = getattr(args, 'pad', 0)
    target_2d = input_2D[:, pad_idx]  # (B,J,2) - removed contiguous()

    # Convert hypotheses from root-relative to absolute camera coordinates using GT root
    # Root at frame args.pad: (B,3)
    gt_root = gt_3D[:, pad_idx, 0, :]  # (B,3) - removed contiguous()
    # Optimize: avoid clone(), create new tensor directly
    gt_root_expand = gt_root.unsqueeze(1).unsqueeze(1)  # (B,1,1,3)
    X_abs = X_bhj3 + gt_root_expand  # Broadcast addition
    X_abs[:, :, 0, :] = gt_root.unsqueeze(1)  # Set root joint

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
        # Softmax method
        softmax_input = -topk_vals / max(tau, 1e-6)
        topk_weights = torch.softmax(softmax_input, dim=1)
    elif weight_method == 'inverse':
        # Inverse proportional weights (slower but interpretable)
        inv_weights = 1.0 / (topk_vals + 1e-6)
        topk_weights = inv_weights / inv_weights.sum(dim=1, keepdim=True)
    elif weight_method == 'exp':
        # Exponential weights with temperature
        temp = args.exp_temp
        max_safe_val = temp * 20
        topk_vals_clipped = torch.clamp(topk_vals, max=max_safe_val)
        exp_vals = torch.exp(-topk_vals_clipped / temp)
        exp_sum = exp_vals.sum(dim=1, keepdim=True)
        topk_weights = exp_vals / torch.clamp(exp_sum, min=1e-10)
        
        # Handle NaN by falling back to uniform weights
        nan_mask = torch.isnan(topk_weights).any(dim=1, keepdim=True)
        uniform_weights = torch.ones_like(topk_weights) / k
        topk_weights = torch.where(nan_mask.expand_as(topk_weights), uniform_weights, topk_weights)
    else:
        # Default to softmax
        topk_weights = torch.softmax(-topk_vals / max(tau, 1e-6), dim=1)

    # Scatter back to full H with zeros elsewhere
    weights = torch.zeros_like(dist)  # (B,H,J)
    weights.scatter_(1, topk_idx, topk_weights)

    # Weighted sum of root-relative 3D hypotheses per joint
    weights_exp = weights.unsqueeze(-1)                     # (B,H,J,1)
    weighted_bj3 = torch.sum(X_bhj3 * weights_exp, dim=1)   # (B,J,3)

    # Assemble output (B,1,J,3) and enforce root at origin
    agg = weighted_bj3.unsqueeze(1).to(dtype=dtype)
    agg[:, :, 0, :] = 0
    return agg


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
            batch_size = gt_3D.size(0)
            # Pre-expand for warmup
            input_2D_expanded_warmup = input_2D_nonflip.repeat(args.hypothesis_num, 1, 1, 1)
            
            # Parallel Euler sampler for warmup
            def euler_sample_parallel_warmup(x2d_expanded, y_local_batch, steps, model_3d):
                dt = 1.0 / steps
                total_batch = y_local_batch.size(0)
                for s in range(steps):
                    t_s = torch.full((total_batch, 1, 1, 1), s * dt, device=y_local_batch.device, dtype=y_local_batch.dtype)
                    v_s = model_3d(x2d_expanded, y_local_batch, t_s)
                    y_local_batch = y_local_batch + dt * v_s
                return y_local_batch
            
            for s_keep in eval_steps:
                y_batch = torch.randn(args.hypothesis_num * batch_size, *gt_3D.shape[1:], 
                                     device=gt_3D.device, dtype=gt_3D.dtype)
                y_s_batch = euler_sample_parallel_warmup(input_2D_expanded_warmup, y_batch, s_keep, model_FMPose)
                y_s_batch = y_s_batch.view(args.hypothesis_num, batch_size, *gt_3D.shape[1:])
                
                # Optimized extraction
                y_extract = y_s_batch[:, :, args.pad].unsqueeze(2)
                y_extract[:, :, :, 0, :] = 0
                list_hypothesis = [y_extract[i] for i in range(args.hypothesis_num)]
                
                output_3D_s = aggregate_hypothesis_camera_weight(list_hypothesis, batch_cam, input_2D, gt_3D)
    
    print("Warm-up complete!\n")
    
    # Actual FPS testing
    print("Starting FPS measurement...")
    num_test_samples = 500  # Test on 500 samples for reliable statistics
    inference_times = []
    
    with torch.no_grad():
        for i_data, data in enumerate(tqdm(dataloader, desc="Testing FPS", total=num_test_samples)):
            if i_data >= num_test_samples:
                break
            
            batch_cam, gt_3D, input_2D, input_2D_GT, input_2D_no, action, subject, cam_ind, index = data
            
            # Start timing after data is prepared
            [input_2D, input_2D_GT, input_2D_no, gt_3D, batch_cam] = get_varialbe('test', [input_2D, input_2D_GT, input_2D_no, gt_3D, batch_cam])
            
            input_2D_nonflip = input_2D[:, 0]
            
            # Parallel Euler sampler for CFM - processes all hypotheses in parallel
            batch_size = gt_3D.size(0)
            # Pre-expand x2d once to avoid repeated operations
            
            def euler_sample_parallel(x2d_expanded, y_local_batch, steps, model_3d):
                """
                x2d_expanded: [N*B, F, J, 2] - pre-expanded input 2D poses
                y_local_batch: [N*B, F, J, 3] - N hypotheses batched together
                """
                dt = 1.0 / steps
                total_batch = y_local_batch.size(0)
                for s in range(steps):
                    t_s = torch.full((total_batch, 1, 1, 1), s * dt, device=y_local_batch.device, dtype=y_local_batch.dtype)
                    v_s = model_3d(x2d_expanded, y_local_batch, t_s)
                    y_local_batch = y_local_batch + dt * v_s
                return y_local_batch
            
            # Synchronize GPU before starting timer
            torch.cuda.synchronize()
            start_time = time.time()
            
            input_2D_expanded = input_2D_nonflip.repeat(args.hypothesis_num, 1, 1, 1)
            
            # ============ INFERENCE STARTS HERE ============
            
            for s_keep in eval_steps:
                # Generate all random noise samples at once [N*B, F, J, 3]
                y_batch = torch.randn(args.hypothesis_num * batch_size, *gt_3D.shape[1:], 
                                     device=gt_3D.device, dtype=gt_3D.dtype)
                
                # Process all hypotheses in parallel (using pre-expanded input)
                y_s_batch = euler_sample_parallel(input_2D_expanded, y_batch, s_keep, model_FMPose)
                
                # Reshape back to [N, B, F, J, 3]
                y_s_batch = y_s_batch.view(args.hypothesis_num, batch_size, *gt_3D.shape[1:])
                
                # Extract predictions and set root to zero
                # Optimize: extract all at once then split
                y_extract = y_s_batch[:, :, args.pad].unsqueeze(2)  # (N,B,1,J,3)
                y_extract[:, :, :, 0, :] = 0  # Set root joint to zero
                list_hypothesis = [y_extract[i] for i in range(args.hypothesis_num)]
                
                # Use camera-weighted aggregation
                # output_3D_s = aggregate_hypothesis_camera_weight(list_hypothesis, batch_cam, input_2D, gt_3D)
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

