import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2_video_predictor
import os
from datetime import datetime

def visualize_masks(frame, masks, alpha=0.5):
    """Visualize masks on frame"""
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy array")
    if not isinstance(masks, np.ndarray):
        raise TypeError("masks must be a numpy array")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
        
    vis_frame = frame.copy()
    for mask in masks:
        vis_frame = vis_frame * (1 - alpha * mask[..., None])
    return vis_frame.astype(np.uint8)

def save_results(save_dir, frame_idx, frame, masks):
    """Save both visualization and raw masks"""
    # Create directories if they don't exist
    os.makedirs(os.path.join(save_dir, 'vis'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'masks'), exist_ok=True)
    
    # Save visualization
    vis_frame = visualize_masks(frame, masks)
    vis_path = os.path.join(save_dir, 'vis', f'frame_{frame_idx:05d}.png')
    cv2.imwrite(vis_path, cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
    
    # Save raw masks
    mask_path = os.path.join(save_dir, 'masks', f'frame_{frame_idx:05d}.npy')
    np.save(mask_path, masks)

# Load model
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# Set paths
video_path = "/home/ti_wang/Ti_workspace/projects/sam2/demo/data/gallery/01_dog.mp4"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"results/video_seg_{timestamp}"
os.makedirs(save_dir, exist_ok=True)

# Check if video file exists
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

# Check if checkpoint file exists
if not os.path.exists(checkpoint):
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    # Initialize state
    state = predictor.init_state(video_path)
    
    # Get first frame dimensions
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to read the first frame")
    if ret:
        # Use center point of frame
        h, w = frame.shape[:2]
        points = np.array([[w//2, h//2]])  # Center point
        labels = np.array([1])
        
        # Save the point data
        point_data = {
            'points': points,
            'labels': labels
        }
        np.save(os.path.join(save_dir, 'point_data.npy'), point_data)
        print(f"Using center point: {points[0]}")
    cap.release()
    
    # Add prompt using the correct method - pass points directly to the method
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(
        state,
        points=points,
        point_labels=labels
    )
    
    # Process and save frames
    cap = cv2.VideoCapture(video_path)
    
    # Create video writer
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_writer = cv2.VideoWriter(
        os.path.join(save_dir, 'visualization.mp4'),
        cv2.VideoWriter_fourcc(*'avc1'),
        fps,
        (width, height)
    )
    if not video_writer.isOpened():
        raise RuntimeError("Failed to create video writer")
    
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        # Clear CUDA cache periodically
        if frame_idx % 100 == 0:
            torch.cuda.empty_cache()
        
        # Convert masks to numpy immediately and clear from GPU
        masks_np = masks.cpu().numpy()
        del masks
        
        print(f"Processing frame {frame_idx}, tracking {len(object_ids)} objects")
        
        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save results
            save_results(save_dir, frame_idx, frame_rgb, masks_np)
            
            # Add to video
            vis_frame = visualize_masks(frame_rgb, masks_np)
            vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            video_writer.write(vis_frame_bgr)
    
    try:
        cap.release()
        video_writer.release()
    except Exception as e:
        print(f"Error releasing resources: {e}")

    # Save metadata
    metadata = {
        'video_path': video_path,
        'timestamp': timestamp,
        'num_frames': frame_idx + 1,
        'object_ids': object_ids.cpu().numpy()
    }
    np.save(os.path.join(save_dir, 'metadata.npy'), metadata)

print(f"Results saved to: {save_dir}")
