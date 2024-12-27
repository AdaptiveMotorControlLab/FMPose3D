import cv2
import os
import sys
from pathlib import Path
import torch
import numpy as np
from contextlib import contextmanager
import logging
import subprocess
import json
from typing import Dict, List, Any
import os.path as osp
import shutil
import sys
from sam2.build_sam import build_sam2_video_predictor
from collections import defaultdict
from utils.metrics import MOTATracker
from utils.visualization import draw_predictions
from utils.utils import load_json_data, get_video_data, get_frame_path, save_video_data_to_json, save_processed_data_to_json

# Import pose estimation components
from deeplabcut.pose_estimation_pytorch.apis.analyze_videos import VideoIterator
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.pose_estimation_pytorch.apis.evaluate import plot_gt_and_predictions

# Reserved colors (BGR format)
GT_COLOR = (0, 0, 255)       # Red for ground truth
PRED_COLOR = (0, 255, 255)   # Yellow for predictions

def initialize_models(pose_config_path, pose_snapshot_path, detector_path, device="cuda"):
    """Initialize SAM2, pose estimation and detector models"""
    # Initialize SAM2
    # sam2_checkpoint = "/home/ti_wang/Ti_workspace/projects/sam2/checkpoints/sam2.1_hiera_large.pt"
    # model_cfg = "/home/ti_wang/Ti_workspace/projects/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    
    checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

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

def process_video_from_json(
    json_path: str,
    base_image_path: str,
    video_id: int,
    base_output_path: str,
    pose_config: str,
    pose_snapshot: str,
    detector_path: str,
    prompt_gt_bbox: bool = False,
    plot_prediction_bbox: bool = False,
    device: str = "cuda"
) -> str:
    # Load JSON data directly
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Get video data directly
    video_images = [img for img in json_data['images'] if img['video_id'] == video_id]
    video_annotations = [ann for ann in json_data['annotations'] if ann['video_id'] == video_id]
    video_images.sort(key=lambda x: int(x['file_name'].split('/')[-1].split('.')[0]))
   
    if not video_images:
        raise ValueError(f"No images found for video_id {video_id}")
    
    # Create directories
    output_dir = Path(base_output_path) / f"videoID_{video_id}"
    frames_dir = output_dir / "frames"
    processed_frames_dir = output_dir / "processed_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    processed_frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original video data to JSON
    original_json_path = save_video_data_to_json(json_data, video_id, output_dir)
    print(f"Original video data saved to: {original_json_path}")
    
    # Copy frames directly
    for image_data in video_images:
        src_path = osp.join(base_image_path, image_data['file_name'])
        original_filename = Path(image_data['file_name']).name
        dst_path = frames_dir / original_filename
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)
    
    # Create original video from frames
    original_video = output_dir / f"videoID_{video_id}_original.mp4"
    create_video_from_frames(frames_dir, original_video)
    
    # Initialize models
    sam2_predictor, pose_runner, detector_runner = initialize_models(
        pose_config, pose_snapshot, detector_path, device
    )
    
    # Get first frame for detection
    first_image_name = Path(video_images[0]['file_name']).name
    first_frame_path = frames_dir / first_image_name
    first_frame = cv2.imread(str(first_frame_path))
    if first_frame is None:
        raise ValueError(f"Could not read first frame: {first_frame_path}")
    
    total_frames = len(video_images)
    
    frame_paths = []
    
    # Initialize lists to collect processed data
    all_predictions = [[] for _ in range(total_frames)]
    all_masks = [[] for _ in range(total_frames)]
    all_bboxes = [[] for _ in range(total_frames)]
    
    # Process first frame
    with torch.inference_mode():
        if prompt_gt_bbox:
            # Get all ground truth bboxes from first annotation
            first_annotations = [ann for ann in video_annotations 
                              if ann['image_id'] == video_images[0]['id']]
            first_bboxes = [np.array(ann['bbox']) for ann in first_annotations]  # List of [x_min, y_min, w, h]
            
            # Convert all [x_min, y_min, w, h] to [x1, y1, x2, y2]
            input_boxes = np.array([
                [
                    bbox[0],              # x1 = x_min
                    bbox[1],              # y1 = y_min
                    bbox[0] + bbox[2],    # x2 = x_min + w
                    bbox[1] + bbox[3]     # y2 = y_min + h
                ] for bbox in first_bboxes
            ])
            print(f"Found {len(first_bboxes)} objects in ground truth")
            print(f"GT bboxes [x_min, y_min, w, h]: {first_bboxes}")
            print(f"Input boxes [x1, y1, x2, y2]: {input_boxes}")
        else:
            # Use detector to get all bboxes
            detections = detector_runner.inference([first_frame])
            if not detections or len(detections[0]['bboxes']) == 0:
                raise ValueError("No detection in first frame")
            
            first_bboxes = detections[0]['bboxes']  # List of [x, y, w, h]
            input_boxes = np.array([
                [x, y, x+w, y+h] for x, y, w, h in first_bboxes
            ])
            print(f"Detected {len(first_bboxes)} objects")

        # Initialize SAM2 with frames directory
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = sam2_predictor.init_state(video_path=str(frames_dir))
            
            # Initialize tracking for all objects at once
            print("Input boxes shape:", input_boxes.shape)
            # Initialize each object separately since SAM2 expects single object IDs
            initial_masks = []
            for obj_id in range(len(input_boxes)):
                frame_idx, object_ids, masks = sam2_predictor.add_new_points_or_box(
                    state, 
                    box=input_boxes[obj_id:obj_id+1],  # Pass one box at a time
                    frame_idx=0,
                    obj_id=obj_id  # Pass single integer as obj_id
                )
                initial_masks.extend([mask.cpu().numpy() for mask in masks])

    # Initialize lists to store results for each object
    num_objects = len(input_boxes)
    all_predictions = [[] for _ in range(num_objects)]
    all_masks = [[] for _ in range(num_objects)]
    all_bboxes = [[] for _ in range(num_objects)]

    # Process all frames
    with torch.inference_mode():
        frame_idx = 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for out_frame_idx, out_object_ids, out_masks in sam2_predictor.propagate_in_video(state):
                print(f"Processing frame {frame_idx}/{total_frames}")
                print("out_object_ids:", out_object_ids)
                # Get the current frame's image
                image_data = video_images[out_frame_idx]
                frame_path = get_frame_path(image_data, base_image_path)
                frame = cv2.imread(frame_path)
                
                frame_name = os.path.basename(frame_path)
                
                if frame is None:
                    print(f"Could not read frame: {frame_path}")
                    continue
                
                frame_predictions = []
                frame_masks = []
                frame_bboxes = []
                
                # Process the masks of the current frame
                current_masks = [mask.cpu().numpy().squeeze(0) for mask in out_masks]
                current_masks = [(mask > 0).astype(np.uint8) for mask in current_masks]
                
                # Process each object's mask
                for obj_id in range(len(out_object_ids)):
                    print(f"Processing object {obj_id}")
                    current_mask = current_masks[obj_id]
                    
                    # Get bbox from mask
                    non_zero_indices = np.nonzero(current_mask)
                    print(f"non_zero_indices: {non_zero_indices}")
                    if len(non_zero_indices[0]) > 0:
                        y_min, x_min = non_zero_indices[0].min(), non_zero_indices[1].min()
                        y_max, x_max = non_zero_indices[0].max(), non_zero_indices[1].max()
                        bbox = np.array([[
                            int(x_min),
                            int(y_min),
                            int(x_max - x_min),  # width
                            int(y_max - y_min)   # height
                        ]])
                        
                        frame_masks.append(current_mask)
                        frame_bboxes.append([x_min, y_min, x_max-x_min, y_max-y_min])
                        
                        if plot_prediction_bbox:
                            # ToDo
                            # Get detections from detector for this frame
                            detections = detector_runner.inference([frame])
                            print("len(detections[0]['bboxes']):", len(detections[0]['bboxes']))
                            if detections and len(detections[0]['bboxes']) > 0:
                                # Use detector bbox for pose estimation
                                bbox = np.array([detections[0]['bboxes'][obj_id]])  # [x, y, w, h]
                         
                        # Run pose estimation for this object
                        context = {"bboxes": bbox}
                        frame_with_context = (frame, context)
                        predictions = pose_runner.inference([frame_with_context])
                        
                        if predictions and len(predictions) > 0:
                            frame_predictions.append(predictions[0])
                        else:
                            frame_predictions.append(None)
                    else:
                        frame_masks.append(None)
                        frame_bboxes.append(None)
                        frame_predictions.append(None)
                
                # Initialize MOTA tracker if this is the first frame
                if frame_idx == 0:
                    mota_tracker = MOTATracker()
                
                # Get ground truth bboxes for current frame
                current_gt_bboxes = []
                for ann in video_annotations:
                    if ann['image_id'] == image_data['id']:
                        current_gt_bboxes.append(ann['bbox'])  # [x, y, w, h] format
                
                # Update MOTA tracker
                print("current_gt_bboxes:", [[int(x) for x in bbox] for bbox in current_gt_bboxes])
                print("frame_bboxes:", [[x for x in bbox] if bbox is not None else None for bbox in frame_bboxes])
                
                mota_tracker.update(current_gt_bboxes, frame_bboxes)
                
                # Print current metrics every 10 frames
                if frame_idx % 10 == 0:
                    summary = mota_tracker.get_metrics()
                    print(f"\nFrame {frame_idx} Tracking Metrics:")
                    print(summary)
                
                # Save final metrics at the end
                if frame_idx == total_frames - 1:
                    metrics_path = output_dir / "tracking_metrics.json"
                    summary = mota_tracker.save_metrics(metrics_path)
                    print("\nFinal Tracking Metrics:")
                    print(summary)
                    print(f"Tracking metrics saved to: {metrics_path}")
                
                # Visualize all objects in one frame
                frame_vis = frame.copy()
                
                # for obj_id in range(len(out_object_ids)):
                #     if frame_predictions[obj_id] is not None:
                        
                #         color = COLOR_PALETTE[obj_id % len(COLOR_PALETTE)]
                        
                #         frame_vis = draw_predictions(
                #             frame=frame_vis,
                #             mask=frame_masks[obj_id],
                #             bbox=frame_bboxes[obj_id],
                #             keypoints=frame_predictions[obj_id]["bodyparts"][..., :2],
                #             confidences=frame_predictions[obj_id]["bodyparts"][..., 2],
                #             color=color
                #         )
    
                # Save frame
                frame_path = processed_frames_dir / f"frame_{frame_idx:04d}.jpg"
                # cv2.imwrite(str(frame_path), frame_vis)
                # Test dlc plotting ToDo:
                # plot_gt_and_predictions(
                #     image_path = xxx,
                #     output_dir = xxx,
                #     gt_bodyparts= ,
                #     pred_bodyparts= ,
                #     bounding_boxes=frame_bboxes,
                # )
        
                frame_paths.append(str(frame_path))
                
                # Store processed data for each object
                for obj_id in range(len(input_boxes)):
                    all_masks[obj_id].append(frame_masks[obj_id])
                    all_bboxes[obj_id].append(frame_bboxes[obj_id])
                    all_predictions[obj_id].append(frame_predictions[obj_id])
                
                frame_idx += 1
                if frame_idx >= total_frames:
                    break

    # Save processed data to JSON
    json_output_path = save_processed_data_to_json(
        output_dir=output_dir,
        video_id=video_id,
        video_images=video_images,
        video_annotations=video_annotations,
        predictions=all_predictions,
        masks=all_masks,
        bboxes=all_bboxes
    )
    print(f"Processed data saved to: {json_output_path}")
    
    # Cleanup
    del sam2_predictor, pose_runner, detector_runner
    torch.cuda.empty_cache()
    
    # Create output video
    video_name = f"videoID_{video_id}"
    output_video = output_dir / f"{video_name}_tracked.mp4"
    
    fps = 15  # Default FPS, adjust if known
    
    # Use ffmpeg to create video from processed frames
    if frame_paths:
        print(f"Creating video at {output_video}")
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', str(processed_frames_dir / 'frame_%04d.jpg'),
            '-vcodec', 'mpeg4',
            '-q:v', '1',
            '-pix_fmt', 'yuv420p',
            str(output_video)
        ]
        
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        print(f"Video successfully saved to {output_video}")
        return str(output_video)

    return str(output_video)

if __name__ == "__main__":
    # Update paths
    # JSON_PATH = "/home/ti_wang/Ti_workspace/projects/sam2/primate_data/datasets/aptv2/processed_dataset/processed/test_annotations_easy_gorilla.json"
    JSON_PATH = "/home/ti_wang/Ti_workspace/projects/sam2/primate_data/datasets/aptv2/processed_dataset/processed/test_annotations_hard_gorilla.json"
    
    BASE_IMAGE_PATH = "/home/ti_wang/Ti_workspace/projects/sam2/primate_data/datasets/aptv2/processed_dataset/images"
    # VIDEO_ID = 1000013  # Example video_id
    VIDEO_ID = 1000008
    # VIDEO_ID = 1000012
    # VIDEO_ID = 1000027
    # VIDEO_ID = 16
    BASE_OUTPUT_DIR = Path("/home/ti_wang/Ti_workspace/projects/sam2/results2/aptv2")
    
    # Pose estimation paths
    POSE_CONFIG = "/home/ti_wang/Ti_workspace/projects/samurai/pre_trained_models/pytorch_config.yaml"
    POSE_SNAPSHOT = "/home/ti_wang/Ti_workspace/projects/samurai/pre_trained_models/snapshot-best-056.pt"
    DETECTOR_PATH = "/home/ti_wang/Ti_workspace/projects/samurai/pre_trained_models/snapshot-detector-best-171.pt"
    
    output_path = process_video_from_json(
        json_path=JSON_PATH,
        base_image_path=BASE_IMAGE_PATH,
        video_id=VIDEO_ID,
        base_output_path=BASE_OUTPUT_DIR,
        pose_config=POSE_CONFIG,
        pose_snapshot=POSE_SNAPSHOT,
        detector_path=DETECTOR_PATH,
        prompt_gt_bbox=True,
        plot_prediction_bbox=False,
        device="cuda"
    )
    print(f"Processing complete! Output saved to: {output_path}")