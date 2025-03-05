import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import functools
import types

# Import DeepLabCut components
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.task import Task

from deeplabcut.pose_estimation_pytorch.apis.evaluate import plot_gt_and_predictions_PFM

# from utils import plot_gt_and_predictions_PFM


keypoint_name_simplified = [
    "forehead",
    "head",
    "L_E",
    "R_E",
    "nose",
    "L_ear",
    "R_ear",
    "mouth_front_top",
    "mouth_front_bottom",
    "mouth_B_L",
    "mouth_B_R",
    "neck",
    "L_S",
    "R_S",
    "upper_B",
    "torso_M_B",
    "body_C",
    "lower_B",
    "L_E",
    "R_E",
    "L_W",
    "R_W",
    "L_H",
    "R_H",
    "L_hip",
    "R_hip",
    "C_hip",
    "L_K",
    "R_K",
    "L_A",
    "R_A",
    "L_foot",
    "R_foot",
    "root_tail",
    "M_tail",
    "M_end_tail",
    "end_tail"
]
# Define skeleton for visualization - same as in the reference file
# PFM_SKELETON = [
#     [3, 5], [4, 5], [6, 3], [7, 4],
#     [5, 12], [13, 12], [14, 12], [2, 17],
#     [19, 13], [20, 14], [21, 19], [22, 20],
#     [23, 21], [24, 22], [25, 12], [26, 12],
#     [25, 27], [26, 27], [25, 28], [26, 29],
#     [27, 28], [27, 29], [28, 30], [29, 31],
#     [30, 32], [31, 33], [27, 34], [34, 35],
#     [35, 36], [36, 37]
# ]
PFM_SKELETON = [
    [1, 11], # [head, neck]
    [2, 4], [3, 4], [5, 2], [6, 3],
    # [4, 11],  # [nose, neck]
    [12, 11], [13, 11], 
    # [1, 16],
    [18, 12], [19, 13], [20, 18], [21, 19],
    [22, 20], [23, 21],
    # [24, 11], [25, 11],
    [26, 11],
    [24, 26], [25, 26], [24, 27], [25, 28],
    # [26, 27], [26, 28], 
    [27, 29], [28, 30],
    [29, 31], [30, 32], [26, 33], [33, 34],
    [34, 35], [35, 36]
]
PFM_SKELETON_NAME = [
    ["head", "neck"], # [1, 11]
    ["L_E", "nose"],      # [2, 4]
    ["R_E", "nose"],      # [3, 4]
    ["L_ear", "L_E"],     # [5, 2]
    ["R_ear", "R_E"],     # [6, 3]
    # ["nose", "neck"],     # [4, 11]
    ["L_S", "neck"],      # [12, 11]
    ["R_S", "neck"],      # [13, 11]
    # ["head", "lower_B"],  # [1, 16]
    ["L_E", "L_S"],       # [18, 12]
    ["R_E", "R_S"],       # [19, 13]
    ["L_W", "L_E"],       # [20, 18]
    ["R_W", "R_E"],       # [21, 19]
    ["L_H", "L_W"],       # [22, 20]
    ["R_H", "R_W"],       # [23, 21]
    # ["L_hip", "neck"],    # [24, 11]
    # # ["R_hip", "neck"],    # [25, 11]
    ["C_hip", "neck"],  # [26, 11]
    ["L_hip", "C_hip"],   # [24, 26]
    ["R_hip", "C_hip"],   # [25, 26]
    ["L_hip", "L_K"],     # [24, 27]
    ["R_hip", "R_K"],     # [25, 28]
    # ["C_hip", "L_K"],     # [26, 27]
    # ["C_hip", "R_K"],     # [26, 28]
    ["L_K", "L_A"],       # [27, 29]
    ["R_K", "R_A"],       # [28, 30]
    ["L_A", "L_foot"],    # [29, 31]
    ["R_A", "R_foot"],    # [30, 32]
    ["C_hip", "root_tail"],# [26, 33]
    ["root_tail", "M_tail"], # [33, 34]
    ["M_tail", "M_end_tail"], # [34, 35]
    ["M_end_tail", "end_tail"] # [35, 36]
]

# idx                 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36          
keypoint_vis_mask = [ 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1]  # Original
keypoint_vis_mask = [ 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1]  # Version 2
keypoint_vis_mask = [ 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]  # Version 3
# mask: forehead:0; upper_back: 14; torso_mid_back: 15; body_center:16; lower_back:17;
keypoint_vis_mask = [ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  

def initialize_model(pose_config_path, pose_snapshot_path, detector_path, device="cuda"):
    """
    Initialize pose estimation model
    
    Args:
        pose_config_path: Path to pose estimation configuration file
        pose_snapshot_path: Path to pose estimation model checkpoint
        detector_path: Path to detector model
        device: Device to run model on (cuda or cpu)
        
    Returns:
        pose_runner: Pose estimation model
        detector_runner: Detector model for bounding box detection
        model_cfg: Model configuration dictionary
    """
    # Fix for PyTorch 2.6+ model loading by patching torch.load
    original_torch_load = torch.load
    
    @functools.wraps(original_torch_load)
    def patched_torch_load(*args, **kwargs):
        # Set weights_only=False explicitly for backward compatibility
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    
    # Replace torch.load with our patched version temporarily
    torch.load = patched_torch_load
    
    try:
        # Initialize pose estimation model
        model_cfg = read_config_as_dict(pose_config_path)
        model_cfg["device"] = device
        
        pose_runner, detector_runner = get_inference_runners(
            model_config=model_cfg,
            snapshot_path=pose_snapshot_path,
            max_individuals=2, # 
            num_bodyparts=len(model_cfg["metadata"]["bodyparts"]),
            num_unique_bodyparts=len(model_cfg["metadata"]["unique_bodyparts"]),
            with_identity=model_cfg["metadata"].get("with_identity", False),
            detector_path=detector_path,
        )
        print("Model initialized successfully")
    finally:
        # Restore original torch.load
        torch.load = original_torch_load
    
    return pose_runner, detector_runner, model_cfg

def calculate_point_size(frame_width, frame_height):
    """Calculate appropriate point size based on image resolution"""
    base_size = 5
    scale_factor = min(frame_width, frame_height) / 500
    return max(3, int(base_size * scale_factor))

def draw_predictions(frame, keypoints, confidences, threshold=0.5, bbox=None, instance_idx=None):
    """
    Draw pose estimation predictions on frame
    
    Args:
        frame: Input image frame
        keypoints: Detected keypoints coordinates
        confidences: Confidence scores for keypoints
        threshold: Confidence threshold for displaying keypoints
        bbox: Bounding box to draw (optional)
        instance_idx: Instance index for color variation (optional)
    Returns:
        frame_vis: Visualization frame with drawn predictions
    """
    frame_vis = frame.copy()
    height, width = frame.shape[:2]
    
    # Calculate appropriate point size and line thickness
    point_size = calculate_point_size(width, height)
    line_thickness = max(1, int(point_size / 2))
    
    # Generate colors based on instance index
    colors = [
        (0, 255, 0),   # Green
        (0, 0, 255),   # Red
        (255, 0, 0),   # Blue
        (255, 255, 0), # Cyan
        (255, 0, 255), # Magenta
        (0, 255, 255), # Yellow
        (128, 0, 128), # Purple
        (0, 128, 128), # Teal
        (128, 128, 0), # Olive
        (128, 0, 0)    # Maroon
    ]
    
    # Default to first color if no instance index
    instance_color = colors[instance_idx % len(colors)] if instance_idx is not None else colors[0]
    keypoint_color = (0, 0, 255) if instance_idx is None else colors[(instance_idx + 5) % len(colors)]
    
    # Draw bounding box if provided
    if bbox is not None:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame_vis, (x, y), (x + w, y + h), instance_color, 2)
    
    # Draw skeleton lines
    for start_idx, end_idx in PFM_SKELETON:
        if (confidences[start_idx] > threshold and 
            confidences[end_idx] > threshold and
            start_idx < len(keypoints) and 
            end_idx < len(keypoints)):
            
            start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
            end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
            
            cv2.line(frame_vis, start_point, end_point, instance_color, line_thickness)
    
    # Draw keypoints
    for i, (x, y) in enumerate(keypoints):
        if confidences[i] > threshold:
            cv2.circle(frame_vis, (int(x), int(y)), point_size, keypoint_color, -1)
    
    return frame_vis

def process_image(image_path, pose_runner, detector_runner=None, output_path=None):
    """
    Process a single image for pose estimation
    
    Args:
        image_path: Path to input image
        pose_runner: Pose estimation model
        detector_runner: Detector model for bounding box detection (optional)
        output_path: Path to save output image
        
    Returns:
        result_image: Processed image with pose estimation
        keypoints_list: List of detected keypoints for each instance
        confidences_list: List of confidence scores for keypoints for each instance
        bboxes: List of detected bounding boxes
    """
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image at: {image_path}")
    
    # Get bounding boxes using detector if provided
    bboxes = []
    if detector_runner is not None:
        with torch.inference_mode():
            detections_bboxes = detector_runner.inference([frame])
            # print(f"detections_bboxes: {detections_bboxes}")
            if detections_bboxes and len(detections_bboxes[0]['bboxes']) > 0:
                bboxes = detections_bboxes[0]['bboxes']
                
                print(f"Detected {len(bboxes)} bounding boxes")
                
    # Result image starts as the original frame
    result_image = frame.copy()
    pred_keypoints_list = []
    pred_confidences_list = []
    
    # Run pose estimation with context (bounding boxes)
    with torch.inference_mode():
        # print(f"bboxes: {bboxes}")
        # if len(bboxes) > 0:
        #     # Process each bounding box separately
        #     for i, bbox in enumerate(bboxes):
        #         # Create frame with context for pose runner
        #         context = {"bboxes": np.array([bbox])}
        #         frame_with_context = (frame, context)
        #         # Run inference with context
        #         predictions = pose_runner.inference([frame_with_context])
        #         print(f"predictions len: {len(predictions)}")
        #         print(f"predictions: {predictions}")
                
        #         # Extract keypoints and confidences
        #         keypoints = predictions[0]["bodyparts"][..., :2]
        #         print(f"keypoints shape: {keypoints.shape}")
        #         confidences = predictions[0]["bodyparts"][..., 2]
        #         # Store results
        #         pred_keypoints_list.append(keypoints)
        #  
        # print("bboxes0:", bboxes[0])
        # context = {"bboxes": np.array(bboxes[0])}
        # print(f"context: {context}")
        # frame_with_context = (frame, context)
        # print(f"frame_with_context: {frame_with_context}")
        # predictions = pose_runner.inference([frame_with_context])
        
        # the context containt the bboxes and confidences;
        
        # context: bboxes, bbox_scores
        context = detections_bboxes[0] 
        # print(f"context: {context}")
        predictions = pose_runner.inference([(frame, context)])

        # why we need to use [0] here?
        pred_keypoints = predictions[0]["bodyparts"] 
        pred_bboxs = predictions[0]["bboxes"]
        pred_scores = predictions[0]["bbox_scores"]
        pred_bboxes_scores = (pred_bboxs, pred_scores)
        print("pred_scores:", pred_scores)
        # print(f"pred_bboxes_scores: {pred_bboxes_scores}")
        
        plot_gt_and_predictions_PFM(
        image_path=image_path,
        output_dir=output_path,
        pred_bodyparts=pred_keypoints,
        bounding_boxes=pred_bboxes_scores,
        skeleton=PFM_SKELETON,
        keypoint_names=keypoint_name_simplified,
        p_cutoff=0.6,
        keypoint_vis_mask=keypoint_vis_mask, # Pass the mask to plotting function
        )
    
    return result_image, pred_keypoints_list, pred_confidences_list, bboxes

def process_video(video_path, pose_runner, detector_runner=None, output_path=None):
    """
    Process video for pose estimation
    
    Args:
        video_path: Path to input video
        pose_runner: Pose estimation model
        detector_runner: Detector model for bounding box detection (optional)
        output_path: Path to save output video
        
    Returns:
        None
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output video writer if path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    detection_interval = 10  # Update detection every N frames
    max_bboxes = 5  # Maximum number of bounding boxes to track
    bboxes = []
    
    with torch.inference_mode():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get bounding boxes using detector
            if frame_count % detection_interval == 0 and detector_runner is not None:
                detections = detector_runner.inference([frame])
                if (detections and isinstance(detections, list) and len(detections) > 0 and 
                    'bboxes' in detections[0] and 
                    isinstance(detections[0]['bboxes'], np.ndarray) and 
                    detections[0]['bboxes'].size > 0):
                    
                    # Get all bounding boxes, limited to max_bboxes
                    new_bboxes = detections[0]['bboxes'].tolist() if isinstance(detections[0]['bboxes'], np.ndarray) else detections[0]['bboxes']
                    new_bboxes = new_bboxes[:max_bboxes]
                    bboxes = new_bboxes
                    print(f"\rFrame {frame_count}: Detected {len(bboxes)} instances", end="")
            
            # Create copy of frame for drawing
            result_frame = frame.copy()
            
            # Process each bounding box
            if len(bboxes) > 0:
                for i, bbox in enumerate(bboxes):
                    # Create frame with context for pose runner
                    context = {"bboxes": np.array([bbox])}
                    frame_with_context = (frame, context)
                    
                    # Run inference with context
                    predictions = pose_runner.inference([frame_with_context])
                    
                    if (predictions and isinstance(predictions, list) and len(predictions) > 0 and 
                        "bodyparts" in predictions[0] and 
                        isinstance(predictions[0]["bodyparts"], np.ndarray) and 
                        predictions[0]["bodyparts"].shape[0] > 0):
                        
                        # Extract keypoints and confidences
                        keypoints = predictions[0]["bodyparts"][..., :2]
                        confidences = predictions[0]["bodyparts"][..., 2]
                        
                        # Draw predictions for this instance
                        result_frame = draw_predictions(
                            result_frame, 
                            keypoints, 
                            confidences, 
                            bbox=bbox, 
                            instance_idx=i
                        )
            else:
                # No detection, run without bounding box
                predictions = pose_runner.inference([frame])
                
                if (predictions and isinstance(predictions, list) and len(predictions) > 0 and
                    "keypoints" in predictions[0] and len(predictions[0]["keypoints"]) > 0):
                    
                    # Extract keypoints and confidences
                    keypoints = predictions[0]["keypoints"][0]  # Assuming single animal
                    confidences = predictions[0]["confidences"][0]
                    
                    # Draw predictions
                    result_frame = draw_predictions(result_frame, keypoints, confidences)
            
            # Write frame to output video
            if output_path:
                out.write(result_frame)
            
            # Display progress
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"\rProcessing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)", end="")
                
    # Clean up
    cap.release()
    if output_path:
        out.release()
        print(f"\nOutput video saved to: {output_path}")
        
    # Free GPU memory
    torch.cuda.empty_cache()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Estimate primate pose in images or videos using DLC")
    parser.add_argument("--input", required=True, help="Path to input image or video")
    parser.add_argument("--output", help="Path to output image or video")
    parser.add_argument("--pose_config", required=True, help="Path to pose estimation config file")
    parser.add_argument("--pose_snapshot", required=True, help="Path to pose estimation model snapshot")
    parser.add_argument("--detector", required=True, help="Path to detector model")
    parser.add_argument("--device", default="cuda", help="Device to run model on (cuda or cpu)")
    args = parser.parse_args()
    
    # Check if input exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    # create the output directory if it does not exist
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize model
    pose_runner, detector_runner, model_cfg = initialize_model(
        args.pose_config, 
        args.pose_snapshot, 
        args.detector, 
        args.device
    )
     
    # Determine output path if not specified
    if not args.output:
        input_path = Path(args.input)
        output_dir = input_path.parent / "outputs"
        os.makedirs(output_dir, exist_ok=True)
        args.output = str(output_dir / f"{input_path.stem}_pose{input_path.suffix}")
    
    # Process input based on type
    input_lower = args.input.lower()
    if input_lower.endswith(('.png', '.jpg', '.jpeg')):
        print(f"Processing image: {args.input}")
        process_image(args.input, pose_runner, detector_runner, args.output)
    elif input_lower.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print(f"Processing video: {args.input}")
        # process_video(args.input, pose_runner, detector_runner, args.output)
    else:
        raise ValueError(f"Unsupported file format: {args.input}")

if __name__ == "__main__":
    main()