#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Script to run video adaptation and analyze results"""
import json
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime
import deeplabcut.modelzoo.video_inference as modelzoo


def run_video_inference(video_path, pose_config_path, pose_model_path, detector_model_path):
    """Run video inference with adaptation using the specified models."""
    # Create timestamp for unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dest_folder = f"/home/ti_wang/Ti_workspace/PrimatePose/Vis/samples/video_adaption_SA_{timestamp}"
    
    # Convert paths to Path objects
    detector_model_path = Path(detector_model_path)
    pose_model_path = Path(pose_model_path)
    
    # Run inference
    modelzoo.video_inference_superanimal(
        videos=[video_path],
        superanimal_name="superanimal_topviewmouse",
        model_name="hrnet_w32",
        detector_name="fasterrcnn_mobilenet_v3_large_fpn",
        video_adapt=True,
        max_individuals=1,
        pseudo_threshold=0.8,
        pcutoff=0.8,
        bbox_threshold=0.9,
        batch_size=4,
        detector_batch_size=4,
        detector_epochs=1,
        pose_epochs=10,
        customized_pose_checkpoint=pose_model_path,
        customized_detector_checkpoint=detector_model_path,
        customized_model_config=pose_config_path,
        dest_folder=dest_folder,
        plot_bboxes=True
    )
    
    return dest_folder

def calculate_jitter_score(json_path):
    """
    Calculate jitter score following the provided approach.
    This method:
    1. Extracts keypoints from all frames
    2. Calculates frame-to-frame differences in x and y coordinates
    3. Removes average motion (to focus on undesired jitter, not intentional movement)
    4. Returns absolute jitter values in x and y directions
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    total_frames = len(data)
    print(f"Total frames: {total_frames}")
    
    if total_frames == 0:
        print("No frames found in JSON")
        return None
        
    # Extract keypoints from all frames
    all_kpts = []
    for frame in data:
        if 'bodyparts' not in frame:
            continue
            
        frame_keypoints = []
        
        for bodypart in frame['bodyparts']:
            # print(len(bodypart))
            for keypoint in bodypart:
                # print(keypoint)
                x, y, conf = keypoint[0], keypoint[1], keypoint[2]
                
                frame_keypoints.append([x, y])

        # If we have keypoints for this frame, add to the list
        print("frame_keypoints:", frame_keypoints)
        if frame_keypoints:
            # print("frame_keypoints:", len(frame_keypoints))
            all_kpts.append(np.array(frame_keypoints)[np.newaxis, :, :])
        
    # Convert to numpy array with shape (frames, keypoints, 3)
    print("all_kpts:", len(all_kpts))
    # print("all_kpts[0]:", all_kpts[0])
    all_kpts = np.concatenate(all_kpts, axis=0)
    
    # Extract x and y coordinates
    print("all_kpts:", all_kpts.shape)
    # print(all_kpts[0])
    xs = all_kpts[:, :, 0]
    print("xs.shape:", xs.shape)
    ys = all_kpts[:, :, 1]
    
    # Calculate frame-to-frame differences
    diff_xs = np.diff(xs, axis=0)
    diff_ys = np.diff(ys, axis=0)
    
    # Calculate average movement
    average_xs = np.nanmean(diff_xs)
    average_ys = np.nanmean(diff_ys)
    
    print(f"Average X movement: {average_xs:.4f} pixels")
    print(f"Average Y movement: {average_ys:.4f} pixels")
    
    # Center movement by subtracting the average
    diff_xs -= average_xs
    diff_ys -= average_ys
    
    # Calculate jitter score (absolute differences)
    jitter_x = np.nanmean(np.abs(diff_xs))
    jitter_y = np.nanmean(np.abs(diff_ys))
    
    # Combined jitter score
    jitter_combined = np.nanmean([jitter_x, jitter_y])
    
    print(f"X jitter score: {jitter_x:.4f} pixels")
    print(f"Y jitter score: {jitter_y:.4f} pixels")
    print(f"Combined jitter score: {jitter_combined:.4f} pixels")
    
    # Return detailed results as in the provided code
    ret = np.array([np.abs(diff_xs), np.abs(diff_ys)])
    
    return ret


def main():
    """Main function to run video inference and optionally analyze results."""
    pose_config_path = "/home/ti_wang/Ti_workspace/PrimatePose/project/pfm_pose_V82_wo_riken_chimpact_20250304/train/pytorch_config.yaml"
    pose_model_path = "/home/ti_wang/Ti_workspace/PrimatePose/Vis/samples/pfm_pose.pt"
    # pose_model_path="/home/ti_wang/Ti_workspace/PrimatePose/Vis/pre_trained_models/aptv2/aptv2_200.pt"
    detector_model_path = "/home/ti_wang/Ti_workspace/PrimatePose/Vis/samples/pfm_det.pt"
    video_path = "/home/ti_wang/Ti_workspace/PrimatePose/Vis/samples/8s_3840_2160_25fps.mp4"
    video_path="/home/ti_wang/Ti_workspace/PrimatePose/Vis/samples/monkey_videos/videoID_1000013_original.mp4"
    
    # Run inference
    output_folder = run_video_inference(video_path, pose_config_path, pose_model_path, detector_model_path)
    print(f"Results saved to: {output_folder}")


if __name__ == "__main__":
    # Choose what to run
    
    main()  
    
    # Choose which JSON file to analyze
    # json_file_path = "/home/ti_wang/Ti_workspace/PrimatePose/Vis/samples/video_adaption_SA_20250317_2004/videoID_1000013_original_superanimal_topviewmouse_pfm_det_pfm_pose_before_adapt.json"
    # json_file_path="/home/ti_wang/Ti_workspace/PrimatePose/Vis/samples/video_adaption_SA_20250317_2004/videoID_1000013_original_superanimal_topviewmouse_snapshot-fasterrcnn_mobilenet_v3_large_fpn-001_snapshot-hrnet_w32-010_after_adapt.json"
        
    # # Calculate jitter score using the new method
    # print("\n=== New Jitter Score Calculation ===")
    # jitter_scores = calculate_jitter_score(json_file_path)
    # if jitter_scores is not None:
    #     # Get mean jitter scores across all frames and keypoints
    #     jitter_x_mean = np.nanmean(jitter_scores[0])
    #     jitter_y_mean = np.nanmean(jitter_scores[1])
    #     jitter_combined = np.nanmean([jitter_x_mean, jitter_y_mean])
        # print(f"Summary - X jitter: {jitter_x_mean:.4f}, Y jitter: {jitter_y_mean:.4f}, Combined: {jitter_combined:.4f} pixels")