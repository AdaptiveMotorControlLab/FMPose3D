import cv2
import os
import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from contextlib import contextmanager

sys.path.append('scripts')
sys.path.append("./sam2")  # Add sam2 to path as done in demo.py
from demo import build_sam2_video_predictor, determine_model_cfg

class VideoProcessingError(Exception):
    pass

@contextmanager
def video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise VideoProcessingError(f"Failed to open video file: {video_path}")
        yield cap
    finally:
        cap.release()

def read_bbox(bbox_path):
    try:
        with open(bbox_path, 'r') as f:
            return [float(x) for x in f.read().strip().split(',')]
    except (FileNotFoundError, ValueError) as e:
        raise VideoProcessingError(f"Failed to read bbox file: {e}")

def print_state_structure(state, level=0, prefix=''):
    """Recursively print the structure of nested dictionaries, lists, tensors, etc."""
    indent = '  ' * level
    
    if isinstance(state, dict):
        print(f"{indent}{prefix}Dict with keys:")
        for key, value in state.items():
            print(f"{indent}  {key}:")
            print_state_structure(value, level + 2)
    elif isinstance(state, (list, tuple)):
        print(f"{indent}{prefix}List/Tuple of length {len(state)}:")
        for i, item in enumerate(state):
            print(f"{indent}  Item {i}:")
            print_state_structure(item, level + 2)
    elif isinstance(state, torch.Tensor):
        print(f"{indent}{prefix}Tensor shape: {state.shape}, dtype: {state.dtype}, device: {state.device}")
    else:
        print(f"{indent}{prefix}Type: {type(state)}, Value: {state}")

def process_video_with_samurai(video_path, bbox_path, output_dir):
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read first frame bbox
    first_bbox = read_bbox(bbox_path)
    
    # Initialize SAMURAI model
    try:
        model_path = "sam2/checkpoints/sam2.1_hiera_base_plus.pt"
        model_cfg = determine_model_cfg(model_path)
        # predictor = build_sam2_video_predictor(model_cfg, model_path, device="cuda:0")
        predictor = build_sam2_video_predictor(model_cfg, model_path, device="cpu")
    except Exception as e:
        raise VideoProcessingError(f"Failed to load model: {e}")
    
    # Load all frames first (following demo.py approach)
    loaded_frames = []
    with video_capture(video_path) as cap:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            loaded_frames.append(frame)
            
    if len(loaded_frames) == 0:
        raise VideoProcessingError("No frames were loaded from the video")
        
    height, width = loaded_frames[0].shape[:2]
    bbox_file = output_dir / "all_bboxes.txt"
    
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        # Initialize state
        state = predictor.init_state(video_path, offload_video_to_cpu=True)
        print("\nState structure:")
        print_state_structure(state)
        
        x, y, w, h = first_bbox
        initial_bbox = (int(x), int(y), int(x+w), int(y+h))
        _, _, masks = predictor.add_new_points_or_box(state, box=initial_bbox, frame_idx=0, obj_id=0)
        
        print("\nState structure after add_new_points_or_box:")
        # print_state_structure(state)
        
        print("\nMasks structure:")
        print_state_structure(masks)
        
        # Process frames
        # for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        #     try:
        #         # Get mask and bbox
        #         mask = masks[0][0].cpu().numpy() > 0.0
        #         non_zero_indices = np.argwhere(mask)
                
        #         if len(non_zero_indices) == 0:
        #             bbox = [0, 0, 0, 0]
        #         else:
        #             y_min, x_min = non_zero_indices.min(axis=0).tolist()
        #             y_max, x_max = non_zero_indices.max(axis=0).tolist()
        #             bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                
        #         # Get corresponding frame from loaded frames
        #         frame = loaded_frames[frame_idx]
                
        #         # Draw bbox and mask
        #         frame_with_vis = frame.copy()
        #         mask_img = np.zeros((height, width, 3), np.uint8)
        #         mask_img[mask] = [255, 0, 0]  # Blue color for mask
        #         frame_with_vis = cv2.addWeighted(frame_with_vis, 1, mask_img, 0.2, 0)
                
        #         x, y, w, h = [int(v) for v in bbox]
        #         cv2.rectangle(frame_with_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
        #         # Save frame
        #         output_path = output_dir / f"frame_{frame_idx:06d}.jpg"
        #         cv2.imwrite(str(output_path), frame_with_vis)
                
        #         # Save bbox data
        #         with open(bbox_file, 'a') as f:
        #             f.write(f"frame_{frame_idx:06d}: {','.join(map(str, bbox))}\n")
                
        #         if frame_idx % 10 == 0:
        #             print(f"Processed {frame_idx} frames")
                    
        #     except Exception as e:
        #         raise VideoProcessingError(f"Error processing frame {frame_idx}: {e}")
    
    # Cleanup
    del predictor, state
    torch.cuda.empty_cache()
    
    print(f"Done! Processed {len(loaded_frames)} frames")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description='Process video with SAMURAI')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--bbox', required=True, help='Path to bbox file')
    parser.add_argument('--output', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    try:
        process_video_with_samurai(args.video, args.bbox, args.output)
    except VideoProcessingError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1) 