import cv2
import os
import sys
from pathlib import Path
import torch
import numpy as np
from contextlib import contextmanager
import logging
import copy

# Add necessary paths
sys.path.append('scripts')
sys.path.append("./sam2")
from demo import build_sam2_video_predictor, determine_model_cfg

# Import pose estimation components
from deeplabcut.pose_estimation_pytorch.apis.analyze_videos import (
    create_df_from_prediction,
    video_inference,
    VideoIterator,
)
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.task import Task

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProcessingError(Exception):
    pass

@contextmanager
def video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ProcessingError(f"Failed to open video file: {video_path}")
        yield cap
    finally:
        cap.release()

def read_bbox(bbox_path):
    try:
        with open(bbox_path, 'r') as f:
            return [float(x) for x in f.read().strip().split(',')]
    except (FileNotFoundError, ValueError) as e:
        raise ProcessingError(f"Failed to read bbox file: {e}")

def initialize_samurai(model_path="sam2/checkpoints/sam2.1_hiera_base_plus.pt", device="cuda:0"):
    """Initialize SAMURAI model"""
    try:
        model_cfg = determine_model_cfg(model_path)
        predictor = build_sam2_video_predictor(model_cfg, model_path, device=device)
        logger.info("SAMURAI model initialized successfully")
        return predictor
    except Exception as e:
        raise ProcessingError(f"Failed to initialize SAMURAI: {e}")

def initialize_pose_model(model_config, snapshot_path, detector_path, num_animals=1, device="cuda"):
    """Initialize pose estimation model"""
    try:
        model_cfg = read_config_as_dict(model_config)
        model_cfg["device"] = device
        model_cfg["max_individuals"] = num_animals

        pose_task = Task(model_cfg["method"])
        pose_runner, detector_runner = get_inference_runners(
            model_config=model_cfg,
            snapshot_path=snapshot_path,
            max_individuals=num_animals,
            num_bodyparts=len(model_cfg["metadata"]["bodyparts"]),
            num_unique_bodyparts=len(model_cfg["metadata"]["unique_bodyparts"]),
            with_identity=model_cfg["metadata"].get("with_identity", False),
            detector_path=detector_path,
            detector_transform=None,
        )
        logger.info("Pose estimation model initialized successfully")
        return pose_task, pose_runner, detector_runner, model_cfg
    except Exception as e:
        raise ProcessingError(f"Failed to initialize pose model: {e}")

def process_video(
    video_path,
    bbox_path,
    output_dir,
    pose_config_path,
    pose_snapshot_path,
    detector_path,
    num_animals=1,
    device="cuda"
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Initialize models
    samurai_predictor = initialize_samurai(device=device)
    pose_task, pose_runner, detector_runner, model_cfg = initialize_pose_model(
        pose_config_path, pose_snapshot_path, detector_path, num_animals, device
    )
    
    # Step 2: Read initial bbox
    first_bbox = read_bbox(bbox_path)
    
    # Initialize video writer
    with video_capture(video_path) as cap:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_dir / "output.mp4"),
        fourcc,
        fps,
        (width, height)
    )
    
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            # Initialize SAMURAI tracking
            state = samurai_predictor.init_state(video_path, offload_video_to_cpu=True)
            x, y, w, h = first_bbox
            initial_bbox = (int(x), int(y), int(x+w), int(y+h))
            _, _, masks = samurai_predictor.add_new_points_or_box(state, box=initial_bbox, frame_idx=0, obj_id=0)
            
            # Process frames
            for frame_idx, object_ids, masks in samurai_predictor.propagate_in_video(state):
                # Get mask and bbox from SAMURAI
                mask = masks[0][0].cpu().numpy() > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                
                # Get frame
                with video_capture(video_path) as cap:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        break
                
                # Get pose estimation
                predictions = video_inference(
                    video=frame,  # Pass single frame
                    task=pose_task,
                    pose_runner=pose_runner,
                    detector_runner=detector_runner,
                    cropping=None,
                )
                
                # Draw visualizations
                frame_vis = frame.copy()
                
                # Draw mask
                mask_img = np.zeros_like(frame)
                mask_img[mask] = [255, 0, 0]  # Blue for mask
                frame_vis = cv2.addWeighted(frame_vis, 1, mask_img, 0.2, 0)
                
                # Draw bbox
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame_vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw keypoints
                if predictions:
                    keypoints = predictions[0]["bodyparts"][..., :2]  # Get x,y coordinates
                    confidences = predictions[0]["bodyparts"][..., 2]  # Get confidence scores
                    for kp_idx, (kp, conf) in enumerate(zip(keypoints, confidences)):
                        if conf > 0.6:  # Confidence threshold
                            x, y = map(int, kp)
                            cv2.circle(frame_vis, (x, y), 3, (0, 0, 255), -1)
                
                # Write frame
                out.write(frame_vis)
                
                if frame_idx % 10 == 0:
                    logger.info(f"Processed frame {frame_idx}")
    
    finally:
        # Cleanup
        out.release()
        del samurai_predictor, state, pose_runner, detector_runner
        torch.cuda.empty_cache()
    
    logger.info("Processing complete!")

if __name__ == "__main__":
    # Paths
    VIDEO_PATH = "/home/ti_wang/Ti_workspace/projects/samurai/demo.mp4"
    BBOX_PATH = "/home/ti_wang/Ti_workspace/projects/samurai/bbox.txt"
    OUTPUT_DIR = "/home/ti_wang/Ti_workspace/projects/samurai/results"
    
    # Pose estimation paths
    POSE_CONFIG = "/home/ti_wang/Ti_workspace/PrimatePose/project/pfm_pose_hrnet_train/train/pytorch_config.yaml"
    POSE_SNAPSHOT = "/home/ti_wang/Ti_workspace/PrimatePose/project/pfm_pose_hrnet_train/train/snapshot-best-056.pt"
    DETECTOR_PATH = "/home/ti_wang/Ti_workspace/PrimatePose/project/pfm_merged_checked_detector_fasterrcnn_train/train/snapshot-detector-best-083.pt"
    
    try:
        process_video(
            video_path=VIDEO_PATH,
            bbox_path=BBOX_PATH,
            output_dir=OUTPUT_DIR,
            pose_config_path=POSE_CONFIG,
            pose_snapshot_path=POSE_SNAPSHOT,
            detector_path=DETECTOR_PATH,
            num_animals=1,
            device="cuda"
        )
    except ProcessingError as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(0)
