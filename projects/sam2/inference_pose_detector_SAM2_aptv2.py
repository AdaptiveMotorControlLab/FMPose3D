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
import motmetrics as mm

# Import pose estimation components
from deeplabcut.pose_estimation_pytorch.apis.analyze_videos import VideoIterator
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.task import Task

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

# Define color palette for different objects (BGR format)
# Avoid Blue, Red, Yellow as they are reserved for special purposes
COLOR_PALETTE = [
    (128, 0, 128),   # Purple
    (0, 165, 255),   # Orange
    (255, 0, 255),   # Magenta
    (0, 128, 128),   # Brown
    (255, 191, 0),   # Deep Sky Blue  
    (180, 105, 255), # Pink
    (128, 128, 0),   # Teal
    (147, 20, 255),  # Deep Pink
    (127, 255, 212), # Aquamarine
]

# Reserved colors (BGR format)
GT_COLOR = (0, 0, 255)       # Red for ground truth
PRED_COLOR = (0, 255, 255)   # Yellow for predictions

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

def draw_predictions(frame, mask, bbox, keypoints, confidences, color):
    """Draw all predictions on frame with specified color"""
    frame_vis = frame.copy()
    height, width = frame.shape[:2]
    
    # Calculate skeleton color (slightly darker)
    skeleton_color = tuple(int(c * 0.8) for c in color)
    
    # Calculate appropriate point size and line thickness based on image resolution
    point_size = calculate_point_size(width, height)
    line_thickness = max(1, int(point_size / 2*1.5))
    
    # Draw mask
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    if mask is not None:
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        
        mask = (mask > 0).astype(np.uint8)
        
        # Create 3-channel mask with specified color
        mask_3d = np.stack([
            mask * color[0],  # Blue channel
            mask * color[1],  # Green channel
            mask * color[2]   # Red channel
        ], axis=2).astype(np.uint8)
        
        # Apply mask overlay
        frame_vis = cv2.addWeighted(frame_vis, 1, mask_3d, 0.2, 0)

    # Draw bbox with specified color
    if isinstance(bbox, np.ndarray):
        bbox = bbox.flatten()
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    cv2.rectangle(frame_vis, (x1, y1), (x1+x2, y1+y2), color, line_thickness)
    
    # Draw keypoints and skeleton
    if isinstance(keypoints, np.ndarray):
        if len(keypoints.shape) == 3:  # Shape: (N, 37, 2)
            for instance_idx in range(len(keypoints)):
                instance_keypoints = keypoints[instance_idx]
                instance_confidences = confidences[instance_idx]
                
                # Draw skeleton connections with skeleton color
                for connection in PFM_SKELETON:
                    idx1, idx2 = connection[0]-1, connection[1]-1
                    if (instance_confidences[idx1] > 0 and 
                        instance_confidences[idx2] > 0):
                        pt1 = tuple(map(int, instance_keypoints[idx1]))
                        pt2 = tuple(map(int, instance_keypoints[idx2]))
                        cv2.line(frame_vis, pt1, pt2, skeleton_color, line_thickness)
                
                # Draw keypoints with specified color
                for kp_idx, (kp, conf) in enumerate(zip(instance_keypoints, instance_confidences)):
                    if conf > 0:
                        x, y = int(kp[0]), int(kp[1])
                        cv2.circle(frame_vis, (x, y), point_size, color, -1)
    
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

def load_json_data(json_path: str) -> Dict[str, Any]:
    """Load and return the JSON data"""
    with open(json_path, 'r') as f:
        return json.load(f)

def get_video_data(json_data: Dict[str, Any], video_id: int) -> tuple[List[Dict], List[Dict]]:
    """
    Extract images and annotations for a specific video_id
    Returns:
        tuple: (video_images, video_annotations)
    """
    # Filter images and annotations by video_id
    video_images = [img for img in json_data['images'] if img['video_id'] == video_id]
    video_annotations = [ann for ann in json_data['annotations'] if ann['video_id'] == video_id]
    
    # Sort images by frame number (extracted from file_name)
    video_images.sort(key=lambda x: int(x['file_name'].split('/')[-1].split('.')[0]))
    
    return video_images, video_annotations

def get_frame_path(image_data: Dict, base_image_path: str) -> str:
    """Construct full frame path from image data"""
    return osp.join(base_image_path, image_data['file_name'])

def save_processed_data_to_json(
    output_dir: Path,
    video_id: int,
    video_images: List[Dict],
    video_annotations: List[Dict],
    predictions: List[List[Dict]],  # List of predictions for each object
    masks: List[List[np.ndarray]],  # List of masks for each object
    bboxes: List[List[np.ndarray]]  # List of bboxes for each object
) -> str:
    processed_data = {
        "video_id": int(video_id),
        "images": [],
        "annotations": [],
        "predictions": []
    }
    
    num_objects = len(predictions)
    
    for frame_idx, (image_data, annotation) in enumerate(zip(video_images, video_annotations)):
        # Save image info
        processed_data["images"].append({
            "id": int(image_data["id"]),
            "file_name": image_data["file_name"],
            "width": int(image_data["width"]),
            "height": int(image_data["height"]),
            "frame_idx": int(frame_idx)
        })
        
        # Save original annotation
        processed_data["annotations"].append({
            "image_id": int(annotation["image_id"]),
            "bbox": [float(x) for x in annotation["bbox"]],
            "keypoints": [float(x) for x in annotation["keypoints"].ravel()] if isinstance(annotation["keypoints"], np.ndarray) else annotation["keypoints"],
            "frame_idx": int(frame_idx)
        })
        
        # Save predictions for each object
        frame_predictions = []
        for obj_id in range(num_objects):
            pred = predictions[obj_id][frame_idx]
            if pred:
                bodyparts = pred["bodyparts"]
                if bodyparts is not None:
                    frame_predictions.append({
                        "object_id": obj_id,
                        "frame_idx": int(frame_idx),
                        "bbox": [float(x) for x in bboxes[obj_id][frame_idx]] if bboxes[obj_id][frame_idx] is not None else None,
                        "keypoints": bodyparts[..., :2].tolist(),
                        "confidences": bodyparts[..., 2].tolist()
                    })
        
        processed_data["predictions"].extend(frame_predictions)
    
    json_path = output_dir / f"videoID_{video_id}_processed.json"
    with open(json_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    return str(json_path)

def save_video_data_to_json(json_data: Dict, video_id: int, output_dir: Path) -> str:
    """Extract and save video-specific data to a new JSON file"""
    video_data = {
        "info": json_data.get("info", {}),
        "licenses": json_data.get("licenses", []),
        "categories": json_data.get("categories", []),
        "images": [],
        "annotations": []
    }
    
    # Get all images for this video
    video_images = [img for img in json_data['images'] if img['video_id'] == video_id]
    video_data['images'] = video_images
    
    # Get all annotations for this video
    image_ids = {img['id'] for img in video_images}
    video_annotations = [ann for ann in json_data['annotations'] 
                        if ann['image_id'] in image_ids]
    video_data['annotations'] = video_annotations
    
    # Save to JSON file
    json_path = output_dir / f"videoID_{video_id}_original.json"
    with open(json_path, 'w') as f:
        json.dump(video_data, f, indent=2)
    
    return str(json_path)

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in format [x1, y1, x2, y2]"""
    # Convert [x_min, y_min, x_max, y_max] format if needed
    if len(box1) == 4 and len(box2) == 4:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0
    return 0

def convert_to_x1y1x2y2(bbox):
    """Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]"""
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

# Add this class after the helper functions and before process_video_from_json
class MOTATracker:
    def __init__(self, iou_threshold=0.5):
        """Initialize MOTA tracker
        Args:
            iou_threshold: IoU threshold for matching predictions with ground truth
        """
        self.acc = mm.MOTAccumulator(auto_id=True)
        self.iou_threshold = iou_threshold
        
    def update(self, gt_bboxes, pred_bboxes):
        """Update metrics with new frame data
        Args:
            gt_bboxes: List of ground truth bboxes [x, y, w, h]
            pred_bboxes: List of predicted bboxes [x1, y1, x2, y2]
        """
        if not gt_bboxes and not pred_bboxes:
            return
            
        # Convert ground truth boxes to x1y1x2y2 format
        gt_bboxes = [convert_to_x1y1x2y2(bbox) for bbox in gt_bboxes]
        
        # pred_bboxes already in [x1, y1, x2, y2] format
        # valid_pred_bboxes = [bbox for bbox in pred_bboxes if bbox is not None]
        valid_pred_bboxes = [convert_to_x1y1x2y2(bbox) for bbox in pred_bboxes if bbox is not None]
        
        # Calculate distances (cost matrix)
        distances = []
        for gt_box in gt_bboxes:
            frame_distances = []
            for pred_box in valid_pred_bboxes:
                iou = calculate_iou(gt_box, pred_box)
                # Convert IoU to distance (1 - IoU)
                distance = 1 - iou if iou > self.iou_threshold else np.nan
                frame_distances.append(distance)
            distances.append(frame_distances)
        
        # Update accumulator
        self.acc.update(
            [i for i in range(len(gt_bboxes))],      # Ground truth objects
            [i for i in range(len(valid_pred_bboxes))], # Predicted objects
            distances if distances else np.empty((0, 0))  # Distance matrix
        )
    
    def get_metrics(self):
        """Calculate current metrics"""
        mh = mm.metrics.create()
        return mh.compute(
            self.acc,
            metrics=['mota', 'motp', 'num_switches', 'num_false_positives', 'num_misses'],
            name='acc'
        )
    
    def save_metrics(self, output_path):
        """Save current metrics to JSON file"""
        summary = self.get_metrics()
        with open(output_path, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)
        return summary

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
                
                # 获取当前帧的图像
                image_data = video_images[out_frame_idx]
                frame_path = get_frame_path(image_data, base_image_path)
                frame = cv2.imread(frame_path)
                
                if frame is None:
                    print(f"Could not read frame: {frame_path}")
                    continue
                
                frame_predictions = []
                frame_masks = []
                frame_bboxes = []
                
                # 处理当前帧的masks
                current_masks = [mask.cpu().numpy().squeeze(0) for mask in out_masks]
                current_masks = [(mask > 0).astype(np.uint8) for mask in current_masks]
                
                # Process each object's mask
                for obj_id in range(len(input_boxes)):
                    current_mask = current_masks[obj_id]
                    
                    # Get bbox from mask
                    non_zero_indices = np.nonzero(current_mask)
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
                    mota_tracker = MOTATracker(iou_threshold=0.5)
                
                # Get ground truth bboxes for current frame
                current_gt_bboxes = []
                for ann in video_annotations:
                    if ann['image_id'] == image_data['id']:
                        current_gt_bboxes.append(ann['bbox'])  # [x, y, w, h] format
                
                # Update MOTA tracker
                print("current_gt_bboxes:", current_gt_bboxes)
                print("frame_bboxes:", frame_bboxes)
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
                for obj_id in range(len(input_boxes)):
                    if frame_predictions[obj_id] is not None:


                        color = COLOR_PALETTE[obj_id % len(COLOR_PALETTE)]
                        
                        frame_vis = draw_predictions(
                            frame=frame_vis,
                            mask=frame_masks[obj_id],
                            bbox=frame_bboxes[obj_id],
                            keypoints=frame_predictions[obj_id]["bodyparts"][..., :2],
                            confidences=frame_predictions[obj_id]["bodyparts"][..., 2],
                            color=color
                        )
                
                # Save frame
                frame_path = processed_frames_dir / f"frame_{frame_idx:04d}.jpg"
                cv2.imwrite(str(frame_path), frame_vis)
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
    
    fps = 25  # Default FPS, adjust if known
    
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

def create_video_from_frames(frames_dir: Path, output_video_path: Path, fps: int = 25):
    """Create a video from original frames
    Args:
        frames_dir: Directory containing the frame images
        output_video_path: Path to save the output video
        fps: Frames per second for the output video
    """
    print(f"Creating original video at {output_video_path}")
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', str(frames_dir / '%04d.jpg'),  # '%04d.jpg'
        '-vcodec', 'mpeg4',
        '-q:v', '1',  # High quality
        '-pix_fmt', 'yuv420p',
        str(output_video_path)
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        print(f"Original video successfully saved to {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e.stderr}")

if __name__ == "__main__":
    # Update paths
    # JSON_PATH = "/home/ti_wang/Ti_workspace/projects/sam2/primate_data/datasets/aptv2/processed_dataset/processed/test_annotations_easy_gorilla.json"
    JSON_PATH = "/home/ti_wang/Ti_workspace/projects/sam2/primate_data/datasets/aptv2/processed_dataset/processed/test_annotations_hard_gorilla.json"
    
    BASE_IMAGE_PATH = "/home/ti_wang/Ti_workspace/projects/sam2/primate_data/datasets/aptv2/processed_dataset/images"
    # VIDEO_ID = 1000013  # Example video_id
    # VIDEO_ID = 1000008
    # VIDEO_ID = 1000012
    VIDEO_ID = 1000027
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