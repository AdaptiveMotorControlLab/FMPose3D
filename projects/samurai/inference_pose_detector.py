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
sys.path.append('scripts')
sys.path.append("./sam2")
from demo import build_sam2_video_predictor, determine_model_cfg

# Import pose estimation components
from deeplabcut.pose_estimation_pytorch.apis.analyze_videos import VideoIterator
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.task import Task

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    """Initialize SAMURAI, pose estimation and detector models"""
    # Initialize SAMURAI
    model_path = "sam2/checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg = determine_model_cfg(model_path)
    samurai_predictor = build_sam2_video_predictor(model_cfg, model_path, device=device)
    logger.info("SAMURAI model initialized")

    # Initialize pose estimation model
    model_cfg = read_config_as_dict(pose_config_path)
    model_cfg["device"] = device
    
    pose_task = Task(model_cfg["method"])
    pose_runner, detector_runner = get_inference_runners(
        model_config=model_cfg,
        snapshot_path=pose_snapshot_path,
        max_individuals=1,
        num_bodyparts=len(model_cfg["metadata"]["bodyparts"]),
        num_unique_bodyparts=len(model_cfg["metadata"]["unique_bodyparts"]),
        with_identity=model_cfg["metadata"].get("with_identity", False),
        detector_path=detector_path,  # No detector needed
    )
    logger.info("Pose estimation model initialized")
    
    return samurai_predictor, pose_runner, detector_runner

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
     
    # Draw mask
    mask_img = np.zeros_like(frame)
    mask_img[mask] = [255, 0, 0]  # Blue for mask   [blue, green, red]
    frame_vis = cv2.addWeighted(frame_vis, 1, mask_img, 0.2, 0)
    
    # Draw bbox
    if isinstance(bbox, np.ndarray):
        bbox = bbox.flatten()
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), line_thickness) # green bbox
    
    # Draw keypoints and skeleton
    if isinstance(keypoints, np.ndarray):
        instance_keypoints = keypoints[0]  # shape: (37, 2)
        instance_confidences = confidences[0]  # shape: (37,)
        
        pcutoff=0
        # First draw skeleton connections
        for connection in PFM_SKELETON:
            idx1, idx2 = connection[0]-1, connection[1]-1  # Indices start from 0
            if (instance_confidences[idx1] > pcutoff and 
                instance_confidences[idx2] > pcutoff):  # Only connect high confidence points
                pt1 = tuple(map(int, instance_keypoints[idx1]))
                pt2 = tuple(map(int, instance_keypoints[idx2]))
                cv2.line(frame_vis, pt1, pt2, (0, 255, 255), line_thickness)  # Yellow lines
        
        # Then draw keypoints on top
        for kp_idx, (kp, conf) in enumerate(zip(instance_keypoints, instance_confidences)):
            if conf > pcutoff:  # Confidence threshold
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame_vis, (x, y), point_size, (0, 0, 255), -1)  # Red points
                
                # Optional: Display keypoint index
                # cv2.putText(frame_vis, str(kp_idx+1), (x+5, y+5), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.3*point_size/3, (0,0,255), 1)
    
    return frame_vis

def process_frame_batch(frames, pose_runner):
    """Process a batch of frames through pose estimation"""
    try:
        predictions = pose_runner.inference(frames)
        return predictions
    except Exception as e:
        logger.error(f"Error in pose estimation: {e}")
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

def process_video_with_tracking(video_path, bbox_path, output_path, pose_config, pose_snapshot, detector_path, device="cuda"):
    """Main processing function for video tracking and pose estimation"""
    # Step 1: Initialize models
    samurai_predictor, pose_runner, detector_runner = initialize_models(pose_config, pose_snapshot, detector_path, device)
     
    # Initialize video capture and writer
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output directories
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    # Read initial bbox
    # first_bbox = read_bbox(bbox_path)

    # Get first frame for detection
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Could not read first frame")
    # Use detector to get initial bbox
    with torch.inference_mode():
        detections = detector_runner.inference([first_frame])
        if not detections or len(detections[0]['bboxes']) == 0:
            raise ValueError("No detection in first frame")
        
        first_bbox = detections[0]['bboxes'][0]  # 使用第一个检测框
        logger.info(f"Initial detection bbox: {first_bbox}")
        
    # We'll save frames instead of directly writing to video
    frame_paths = []
         
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            # Initialize SAMURAI tracking
            state = samurai_predictor.init_state(video_path, offload_video_to_cpu=True)
            x, y, w, h = first_bbox
            initial_bbox = (int(x), int(y), int(x+w), int(y+h))
            _, _, masks = samurai_predictor.add_new_points_or_box(state, box=initial_bbox, frame_idx=0, obj_id=0)
            # print("masks:", type(masks))
            # print("masks:", masks.shape)
            # Process frames one by one
            for frame_idx, object_ids, masks in samurai_predictor.propagate_in_video(state):
                # Print progress
                if frame_idx % 10 == 0:
                    print(f"Processing frame {frame_idx}/{total_frames}")
                
                # Get mask and bbox from SAMURAI
                mask = masks[0][0].cpu().numpy() > 0.0
                non_zero_indices = np.argwhere(mask)
                
                if len(non_zero_indices) == 0:
                    x, y, w, h = first_bbox
                    bbox = np.array([[x, y, w, h]])  # Already in COCO format
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    # Convert to COCO format
                    bbox = np.array([[
                        x_min,
                        y_min,
                        x_max - x_min,  # width
                        y_max - y_min   # height
                    ]])
                
                # Read current frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Create context for current frame - correct format
                context = {"bboxes": bbox}  # Changed this line
                frame_with_context = (frame, context)
                
                # Run pose estimation on single frame
                predictions = pose_runner.inference([frame_with_context])
                
                # if predictions:
                #     print("First prediction type:", type(predictions[0]))
                #     print("First prediction keys:", predictions[0].keys() if isinstance(predictions[0], dict) else "Not a dict") # dict_keys(['bodyparts', 'bboxes'])
                #     print("First prediction content:", predictions[0])
                
                if predictions and len(predictions) > 0:
                    try:
                        # 每10帧打印一次
                        if frame_idx % 10 == 0:
                            num_instances = predictions[0]["bodyparts"].shape[0]
                            print(f"Frame {frame_idx}: Detected {num_instances} instances")
                        
                        keypoints = predictions[0]["bodyparts"][..., :2]
                        confidences = predictions[0]["bodyparts"][..., 2]
                        
                        x, y, w, h = bbox[0]
                        vis_bbox = [x, y, x+w, y+h]
                        
                        frame_vis = draw_predictions(
                            frame=frame,
                            mask=mask,
                            bbox=vis_bbox,
                            keypoints=keypoints,
                            confidences=confidences
                        )
                        
                        # Save frame
                        frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
                        cv2.imwrite(str(frame_path), frame_vis)
                        frame_paths.append(str(frame_path))
                        
                    except Exception as e:
                        print(f"Error processing predictions: {e}")
                        print("Current bbox:", bbox)
                        print("Current frame shape:", frame.shape)
    
    finally:
        # Cleanup
        cap.release()
        del samurai_predictor, state, pose_runner
        torch.cuda.empty_cache()
    
    # Use ffmpeg to create video from frames
    if frame_paths:
        output_video = str(output_dir / "output.mp4")
        # Try with mpeg4 encoder instead of libx264
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', str(frames_dir / 'frame_%06d.jpg'),
            '-vcodec', 'mpeg4',  # Changed from libx264 to mpeg4
            '-q:v', '1',        # Quality setting for mpeg4
            '-pix_fmt', 'yuv420p',
            output_video
        ]
        
        try:
            logger.info(f"Creating video with command: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(ffmpeg_cmd, 
                                 check=True, 
                                 capture_output=True, 
                                 text=True)
            
            # If mpeg4 fails, try with other available encoders
            if not (os.path.exists(output_video) and os.path.getsize(output_video) > 0):
                logger.info("MPEG4 encoding failed, trying alternative encoder...")
                alternative_cmd = [
                    'ffmpeg', '-y',
                    '-framerate', str(fps),
                    '-i', str(frames_dir / 'frame_%06d.jpg'),
                    '-c:v', 'mjpeg',  # Try mjpeg encoder
                    '-q:v', '2',
                    '-pix_fmt', 'yuv420p',
                    str(output_dir / "output.avi")  # Use .avi for mjpeg
                ]
                subprocess.run(alternative_cmd, check=True, capture_output=True, text=True)
                return str(output_dir / "output.avi")
            
            logger.info("Successfully created video")
            return output_video
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create video: {e.stderr}")
            # If video creation fails, return the directory containing individual frames
            return str(frames_dir)

    logger.info(f"Processing complete! Results saved to {output_dir}")
    return str(output_dir / "output.mp4")

if __name__ == "__main__":
    # Paths
    VIDEO_PATH = "/home/ti_wang/Ti_workspace/projects/samurai/results/monkey_data/a_monkey_10frames.mp4"
    BBOX_PATH = "/home/ti_wang/Ti_workspace/projects/samurai/bbox.txt"
    OUTPUT_DIR = "/home/ti_wang/Ti_workspace/projects/samurai/results/output"  # Changed this
    
    # Pose estimation paths
    POSE_CONFIG = "/home/ti_wang/Ti_workspace/projects/samurai/pre_trained_models/pytorch_config.yaml"
    POSE_SNAPSHOT = "/home/ti_wang/Ti_workspace/projects/samurai/pre_trained_models/snapshot-best-056.pt"
    DETECTOR_PATH = "/home/ti_wang/Ti_workspace/projects/samurai/pre_trained_models/snapshot-detector-best-171.pt"
    try:
        process_video_with_tracking(
            video_path=VIDEO_PATH,
            bbox_path=BBOX_PATH,
            output_path=OUTPUT_DIR,  # Now passing directory path
            pose_config=POSE_CONFIG,
            pose_snapshot=POSE_SNAPSHOT,
            detector_path=DETECTOR_PATH,
            device="cuda"
        )
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(0) 