import cv2
import os
import sys
from pathlib import Path
import torch
import numpy as np
from contextlib import contextmanager
import logging
import subprocess

# Add necessary paths
# sys.path.append('scripts')
# sys.path.append("./sam2")
import sys
# print("sys.path:", sys.path)
from sam2.build_sam import build_sam2_video_predictor
# from sam2.utils.config import Config

# Import pose estimation components
from deeplabcut.pose_estimation_pytorch.apis.analyze_videos import VideoIterator
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.task import Task

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# Define skeleton for visualization
PFM_SKELETON = [
    [3, 5], [4, 5], [6, 3], [7, 4],
    [5, 12], [13, 12], [14, 12], [2, 17],
    [19, 13], [20, 14], [21, 19], [22, 20],
    [23, 21], [24, 22], [25, 12], [26, 12],
    [25, 27], [26, 27], [25, 28], [26, 29],
    [27, 28], [27, 29], [28, 30], [29, 31],
    [30, 32], [31, 33], [27, 34], [34, 35],
    [35, 36], [36, 37]
]

def read_bbox(bbox_path):
    """Read initial bbox from file"""
    with open(bbox_path, 'r') as f:
        x, y, w, h = map(float, f.read().strip().split(','))
        return [int(x), int(y), int(w), int(h)]

def initialize_models(pose_config_path, pose_snapshot_path, detector_path, device="cuda"):
    """Initialize SAM2, pose estimation and detector models"""
    # Initialize SAM2
    # sam2_checkpoint = "/home/ti_wang/Ti_workspace/projects/sam2/checkpoints/sam2.1_hiera_large.pt"
    # model_cfg = "/home/ti_wang/Ti_workspace/projects/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    
    checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

    # # Verify config file exists
    # if not os.path.exists(model_cfg):
    #     raise FileNotFoundError(f"Config file not found at: {model_cfg}")
    
    # print(f"Using config file at: {model_cfg}")
    
    # # Initialize SAM2 model
    # try:
    #     predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    # except Exception as e:
    #     raise RuntimeError(f"Failed to initialize SAM2 model: {e}")
    
    print("SAM model initialized")
    
    # Initialize pose estimation model
    model_cfg = read_config_as_dict(pose_config_path)
    model_cfg["device"] = device
    
    pose_task = Task(model_cfg["method"])
    pose_runner, detector_runner = get_inference_runners(
        model_config=model_cfg,
        snapshot_path=pose_snapshot_path,
        max_individuals=10,
        num_bodyparts=len(model_cfg["metadata"]["bodyparts"]),
        num_unique_bodyparts=len(model_cfg["metadata"]["unique_bodyparts"]),
        with_identity=model_cfg["metadata"].get("with_identity", False),
        detector_path=detector_path,
    )
    print("Pose estimation model initialized")
    
    return predictor, pose_runner, detector_runner

def calculate_point_size(frame_width, frame_height):
    """Calculate appropriate point size based on image resolution"""
    # 基于图像对角线长度计算点的大小
    diagonal = np.sqrt(frame_width**2 + frame_height**2)
    
    # 设置点的大小为对角线长度的千分之一到千分之二之间
    point_size = max(3, int(diagonal * 0.0015))  # 最小为3像素
    
    return point_size

def draw_predictions(frame, mask, bbox, keypoints, confidences):
    """Draw all predictions on frame"""
    frame_vis = frame.copy()
    height, width = frame.shape[:2]
    
    # Calculate appropriate point size and line thickness based on image resolution
    point_size = calculate_point_size(width, height)
    line_thickness = max(1, int(point_size / 2))
    
    # Draw mask - handle tensor masks properly
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Ensure mask is binary and correct shape
    if mask is not None:
        if len(mask.shape) > 2:  # If mask has extra dimensions
            mask = mask.squeeze()  # Remove single dimensions
        
        # Convert to binary mask
        mask = (mask > 0).astype(np.uint8)
        
        # Create 3-channel mask for visualization
        mask_3d = np.stack([
            mask * 255,  # Blue channel
            mask * 0,    # Green channel
            mask * 0     # Red channel
        ], axis=2).astype(np.uint8)
        
        # Apply mask overlay
        frame_vis = cv2.addWeighted(frame_vis, 1, mask_3d, 0.2, 0)

    # Draw bbox
    if isinstance(bbox, np.ndarray):
        bbox = bbox.flatten()
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)
    
    # Draw keypoints and skeleton
    if isinstance(keypoints, np.ndarray):
        # Handle multiple instances if present
        if len(keypoints.shape) == 3:  # Shape: (N, 37, 2)
            for instance_idx in range(len(keypoints)):
                instance_keypoints = keypoints[instance_idx]
                instance_confidences = confidences[instance_idx]
                
                # Draw skeleton connections
                for connection in PFM_SKELETON:
                    idx1, idx2 = connection[0]-1, connection[1]-1
                    if (instance_confidences[idx1] > 0 and 
                        instance_confidences[idx2] > 0):
                        pt1 = tuple(map(int, instance_keypoints[idx1]))
                        pt2 = tuple(map(int, instance_keypoints[idx2]))
                        cv2.line(frame_vis, pt1, pt2, (0, 255, 255), line_thickness)
                
                # Draw keypoints
                for kp_idx, (kp, conf) in enumerate(zip(instance_keypoints, instance_confidences)):
                    if conf > 0:
                        x, y = int(kp[0]), int(kp[1])
                        cv2.circle(frame_vis, (x, y), point_size, (0, 0, 255), -1)
        else:  # Single instance case
            instance_keypoints = keypoints
            instance_confidences = confidences
            
            # Draw skeleton connections
            for connection in PFM_SKELETON:
                idx1, idx2 = connection[0]-1, connection[1]-1
                if (instance_confidences[idx1] > 0 and 
                    instance_confidences[idx2] > 0):
                    pt1 = tuple(map(int, instance_keypoints[idx1]))
                    pt2 = tuple(map(int, instance_keypoints[idx2]))
                    cv2.line(frame_vis, pt1, pt2, (0, 255, 255), line_thickness)
            
            # Draw keypoints
            for kp_idx, (kp, conf) in enumerate(zip(instance_keypoints, instance_confidences)):
                if conf > 0:
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(frame_vis, (x, y), point_size, (0, 0, 255), -1)
    
    return frame_vis

def process_frame_batch(frames, pose_runner):
    """Process a batch of frames through pose estimation"""
    try:
        predictions = pose_runner.inference(frames)
        return predictions
    except Exception as e:
        print(f"Error in pose estimation: {e}")
        return None

def convert_bbox_to_coco(bbox):
    """Convert bbox from [x_min, y_min, x_max, y_max] to COCO format [x, y, w, h]"""
    x_min, y_min, x_max, y_max = bbox
    return np.array([[
        x_min,
        y_min,
        x_max - x_min,  # width
        y_max - y_min   # height
    ]])

def process_video_with_tracking(video_path, frames_dir, bbox_path, output_path, pose_config, pose_snapshot, detector_path, device="cuda"):
    """Main processing function for video tracking and pose estimation"""
    # Step 1: Initialize models
    sam2_predictor, pose_runner, detector_runner = initialize_models(pose_config, pose_snapshot, detector_path, device)
     
    # Create output directories
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create frames and processed_frames directories
    frames_dir = output_dir / "frames"
    processed_frames_dir = output_dir / "processed_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    processed_frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get first frame for detection
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")
        
    # Use detector to get all initial bboxes
    with torch.inference_mode():
        detections = detector_runner.inference([first_frame])
        if not detections or len(detections[0]['bboxes']) == 0:
            raise ValueError("No detection in first frame")
       
        all_bboxes = detections[0]['bboxes']
        first_bbox = all_bboxes[0]
        print(f"Detected {len(all_bboxes)} objects in first frame")
        
        # Initialize tracking with first bbox
        x, y, w, h = first_bbox
        input_box = np.array([x, y, x+w, y+h])
        
        # Initialize SAM2 with frames directory
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Initialize state with video path
            state = sam2_predictor.init_state(video_path=video_path)
            # print(f"State: {state}")
            # Add box prompt
            frame_idx, object_ids, masks = sam2_predictor.add_new_points_or_box(
                state, 
                box=input_box[None, :], 
                frame_idx=0,
                obj_id=0
            )
            previous_mask = masks[0].cpu().numpy()  # Add .cpu() here too
    
    frame_paths = []
    frame_idx = 0
         
    # Process video frames
    with torch.inference_mode():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            print(f"Processing frame {frame_idx}/{total_frames}")
            
            # Propagate masks using SAM2
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # this needs to be fixed; refers to inference_pose_detector_SAM2_aptv2.py
                sam_frame_idx, object_ids, masks = next(sam2_predictor.propagate_in_video(state))
                # Move mask to CPU before converting to numpy
                current_mask = masks[0].cpu().numpy()  # Add .cpu() here
                current_mask = current_mask.squeeze(0)
                print(f"Squeezed mask shape: {current_mask.shape}")
                print(f"Mask value range: [{current_mask.min():.2f}, {current_mask.max():.2f}]")
            
            # # Convert logits to binary mask using sigmoid and threshold
            # current_mask = 1 / (1 + np.exp(-current_mask))  # sigmoid
            # current_mask = (current_mask > 0.5).astype(np.uint8)  # threshold at 0.5
            # print(f"Binary mask unique values: {np.unique(current_mask)}")
            
            current_mask = (current_mask > 0).astype(np.uint8)
            
            # Get bbox from mask
            non_zero_indices = np.argwhere(current_mask)
            if len(non_zero_indices) > 0:
                y_min, x_min = non_zero_indices.min(axis=0)
                y_max, x_max = non_zero_indices.max(axis=0)
                bbox = np.array([[
                    int(x_min),
                    int(y_min),
                    int(x_max - x_min),  # width
                    int(y_max - y_min)   # height
                ]])
                
                # Update input_box for next frame
                input_box = np.array([x_min, y_min, x_max, y_max])
            else:
                print(f"Warning: Empty mask in frame {frame_idx}")
                # Save original frame when mask detection fails
                frame_path = processed_frames_dir / f"frame_{frame_idx:06d}.jpg"
                print(f"Saving original frame to {frame_path}")
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(str(frame_path))
                frame_idx += 1
                continue
            
            # Run pose estimation
            context = {"bboxes": bbox}
            frame_with_context = (frame, context)
            predictions = pose_runner.inference([frame_with_context])
            
            if predictions and len(predictions) > 0:
                keypoints = predictions[0]["bodyparts"][..., :2]
                confidences = predictions[0]["bodyparts"][..., 2]
                
                print(f"bbox: {[x_min, y_min, x_max, y_max]}")
                frame_vis = draw_predictions(
                    frame=frame,
                    mask=current_mask,
                    bbox=[x_min, y_min, x_max, y_max],
                    keypoints=keypoints,
                    confidences=confidences
                )
                
                # Save frame
                frame_path = processed_frames_dir / f"frame_{frame_idx:06d}.jpg"
                print(f"Saving frame to {frame_path}")
                cv2.imwrite(str(frame_path), frame_vis)
                frame_paths.append(str(frame_path))
            
            frame_idx += 1
            if frame_idx >= total_frames:
                break

    # Cleanup (moved outside the with block)
    cap.release()
    del sam2_predictor, pose_runner
    torch.cuda.empty_cache()
    
    # Get video name and create output paths
    video_name = Path(video_path).stem
    output_dir = Path(output_path)
    output_video = output_dir / f"{video_name}_tracked.mp4"
    
    # Use ffmpeg to create video from frames
    if frame_paths:
        try:
            print(f"Creating video at {output_video}")
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', str(processed_frames_dir / 'frame_%06d.jpg'),
                '-vcodec', 'mpeg4',
                '-q:v', '1',
                '-pix_fmt', 'yuv420p',
                str(output_video)
            ]
            
            result = subprocess.run(ffmpeg_cmd, 
                                 check=True, 
                                 capture_output=True, 
                                 text=True)
            
            # If mpeg4 fails, try with alternative encoder
            if not (output_video.exists() and output_video.stat().st_size > 0):
                print("MPEG4 encoding failed, trying alternative encoder...")
                alternative_video = output_dir / f"{video_name}_tracked.avi"
                alternative_cmd = [
                    'ffmpeg', '-y',
                    '-framerate', str(fps),
                    '-i', str(processed_frames_dir / 'frame_%06d.jpg'),
                    '-c:v', 'mjpeg',
                    '-q:v', '2',
                    '-pix_fmt', 'yuv420p',
                    str(alternative_video)
                ]
                subprocess.run(alternative_cmd, check=True, capture_output=True, text=True)
                print(f"Video saved as {alternative_video}")
                return str(alternative_video)
            
            print(f"Video successfully saved to {output_video}")
            return str(output_video)
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to create video: {e.stderr}")
            # If video creation fails, return the directory containing individual frames
            print(f"Frames saved in directory: {processed_frames_dir}")
            return str(processed_frames_dir)

    print(f"Processing complete! Results saved to {output_dir}")
    return str(output_video)

if __name__ == "__main__":
    # Paths
    # VIDEO_PATH = "/home/ti_wang/Ti_workspace/projects/samurai/results/monkey_data/multi_monkey_uhd_3840_2160_25fps.mp4"
    VIDEO_PATH = "/home/ti_wang/Ti_workspace/projects/sam2/results2/monkey_data/a_monkey_10frames.mp4"
    BBOX_PATH = "/home/ti_wang/Ti_workspace/projects/samurai/bbox.txt"
    
    # Create output directory based on video name
    video_name = Path(VIDEO_PATH).stem
    OUTPUT_DIR = Path("/home/ti_wang/Ti_workspace/projects/sam2/results2/output") / video_name
    frame_dir = OUTPUT_DIR / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    
    # Split video into images using ffmpeg
    ffmpeg_extract_cmd = [
        'ffmpeg', '-i', str(VIDEO_PATH),
        '-q:v', '2',
        '-start_number', '0',
        str(frame_dir / '%05d.jpg')
    ]
    
    try:
        subprocess.run(ffmpeg_extract_cmd, check=True, capture_output=True, text=True)
        print(f"Video split into images in: {frame_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to split video: {e.stderr}")
        sys.exit(1)
    
    # Pose estimation paths
    POSE_CONFIG = "/home/ti_wang/Ti_workspace/projects/samurai/pre_trained_models/pytorch_config.yaml"
    POSE_SNAPSHOT = "/home/ti_wang/Ti_workspace/projects/samurai/pre_trained_models/snapshot-best-056.pt"
    DETECTOR_PATH = "/home/ti_wang/Ti_workspace/projects/samurai/pre_trained_models/snapshot-detector-best-171.pt"
    
    # try:
    output_path = process_video_with_tracking(
        video_path=VIDEO_PATH,
        frames_dir=frame_dir,
        bbox_path=BBOX_PATH,
        output_path=OUTPUT_DIR,
        pose_config=POSE_CONFIG,
        pose_snapshot=POSE_SNAPSHOT,
        detector_path=DETECTOR_PATH,
        device="cuda"
    )
    print(f"Processing complete! Output saved to: {output_path}")
    # except Exception as e:
    #     print(f"Processing failed: {e}")
    #     sys.exit(1)
    # except KeyboardInterrupt:
    #     print("Processing interrupted by user")
    #     sys.exit(0)