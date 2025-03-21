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

from deeplabcut.pose_estimation_pytorch.apis.evaluation import plot_gt_and_predictions_PFM

# from utils import plot_gt_and_predictions_PFM

keypoint_name_simplified_V1 = [
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
    "L_Elbow",
    "R_Elbow",
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

keypoint_name_simplified_V2 = [
    "forehead",
    "head",
    "L_E",
    "R_E",
    "nose",
    "L_ear",
    "R_ear",
    "M_T",
    "M_B",
    "M_L",
    "M_R",
    "neck",
    "L_shoulder",
    "R_shoulder",
    "upper_B",
    "torso_M_B",
    "body_C",
    "lower_B",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
    "L_hip",
    "R_hip",
    "C_hip",
    "L_Knee",
    "R_Knee",
    "L_Ankle",
    "R_Ankle",
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
    ["L_Elbow", "L_S"],       # [18, 12]
    ["R_Elbow", "R_S"],       # [19, 13]
    ["L_W", "L_Elbow"],       # [20, 18]
    ["R_W", "R_Elbow"],       # [21, 19]
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

def initialize_model(pose_config_path, pose_snapshot_path, detector_path, max_individuals=8, device="cuda"):
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
            max_individuals=max_individuals, # 
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
                
                # print(f"Detected {len(bboxes)} bounding boxes")
                
    # Result image starts as the original frame
    result_image = frame.copy()
    pred_keypoints_list = []
    pred_confidences_list = []
    
    # Run pose estimation with context (bounding boxes)
    with torch.inference_mode():
        
        # the context containt the bboxes and confidences;  context: {bboxes, bbox_scores}
        context = detections_bboxes[0] 
        predictions = pose_runner.inference([(frame, context)])

        # why we need to use [0] here?
        pred_keypoints = predictions[0]["bodyparts"] 
        pred_bboxs = predictions[0]["bboxes"]
        pred_scores = predictions[0]["bbox_scores"]
        pred_bboxes_scores = (pred_bboxs, pred_scores) # if we use GT bboxes for pose_runner.inference, here pred_bboxes will be the GT bboxes
        # print("pred_scores:", pred_scores)
        
        plot_gt_and_predictions_PFM(
        image_path=image_path,
        output_dir=output_path,
        pred_bodyparts=pred_keypoints,
        bounding_boxes=pred_bboxes_scores,
        skeleton=PFM_SKELETON,
        # keypoint_names=keypoint_name_simplified_V2,
        p_cutoff=0.65,
        keypoint_vis_mask=keypoint_vis_mask, # Pass the mask to plotting function
        )
        
    return result_image, pred_keypoints_list, pred_confidences_list, bboxes

def process_video_pose(video_path, pose_runner, detector_runner=None, output_path=None):
    """
    Process video for pose estimation by extracting frames, processing each frame,
    and reconstructing the video.
    
    Args:
        video_path: Path to input video
        pose_runner: Pose estimation model
        detector_runner: Detector model for bounding box detection (optional)
        output_path: Path to save output video
        
    Returns:
        None
    """
    # Create directory structure
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    target_folder = os.path.dirname(output_path)
    
    # Create a dedicated folder for this video
    video_folder = os.path.join(target_folder, video_name)
    ori_frames_dir = os.path.join(video_folder, "ori_frames")
    results_dir = os.path.join(video_folder, "results")
    
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(ori_frames_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract frames from video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video at: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Extracting frames from video...")
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame
        frame_path = os.path.join(ori_frames_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"\rExtracted {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)", end="")
    
    cap.release()
    print("\nFrame extraction complete.")
    
    # Process each frame
    print("\nProcessing frames...")
    frame_files = sorted(os.listdir(ori_frames_dir))
    # for i, frame_file in enumerate(frame_files):
    #     frame_path = os.path.join(ori_frames_dir, frame_file)
        
    #     # Process frame using process_image
    #     process_image(frame_path, pose_runner, detector_runner, results_dir)
        
    #     if i % 10 == 0:
    #         print(f"\rProcessed {i+1}/{len(frame_files)} frames ({(i+1)/len(frame_files)*100:.1f}%)", end="")
    
    print("\nFrame processing complete.")
    
    # Combine frames into video
    print("\nCreating output video...")
    first_frame = cv2.imread(os.path.join(ori_frames_dir, frame_files[0]))
    height, width = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(video_folder, "1.mp4")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    result_files = sorted(os.listdir(results_dir))
    for i, result_file in enumerate(result_files):
        frame_path = os.path.join(results_dir, result_file)
        frame = cv2.imread(frame_path)
        out.write(frame)
        
        if i % 10 == 0:
            print(f"\rWriting frame {i+1}/{len(result_files)} ({(i+1)/len(result_files)*100:.1f}%)", end="")
    
    out.release()
    print(f"\nOutput video saved to: {video_path}")

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
        pose_config_path=args.pose_config, 
        pose_snapshot_path=args.pose_snapshot, 
        detector_path=args.detector, 
        max_individuals=1,
        device=args.device,
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
        process_video_pose(args.input, pose_runner, detector_runner, args.output)
    else:
        raise ValueError(f"Unsupported file format: {args.input}")

if __name__ == "__main__":
    main()