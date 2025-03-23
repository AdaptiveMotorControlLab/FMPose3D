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
import os
import numpy as np
from pathlib import Path
from datetime import datetime
import deeplabcut.modelzoo.video_inference as modelzoo
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

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
        
        for individual in frame['bodyparts']:
            # print(len(bodypart))
            for keypoint in individual:
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

def extract_keypoints_from_json(json_path):
    """
    Extract keypoints from JSON file
    note:
    for calculating area score and jitter socre, the current function only support single animal video;
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    all_kpts = []
    for frame in data:
        frame_keypoints = []
        for individual in frame['bodyparts']:
            for keypoint in individual:
                x, y, conf = keypoint[0], keypoint[1], keypoint[2]
                frame_keypoints.append([x, y])
        all_kpts.append(np.array(frame_keypoints)[np.newaxis, :, :])
    
    return np.concatenate(all_kpts, axis=0) # shape: [Frame, num_joints, 2]
    
def calculate_area_score(keypoints_array):
    """
    Calculate area score from keypoints array with shape [Frame, num_joints, 3]
    where the last dimension is [x, y, likelihood]
    """
    # Extract just the x,y coordinates (first two channels)
    coordinates = keypoints_array[:, :, :2]  # Shape: [Frame, num_joints, 2]
    
    # Calculate area for each frame
    areas = []
    for frame_keypoints in coordinates:
        # Filter out keypoints with low likelihood if needed
        # This is optional - you can set a threshold based on your needs
        # valid_keypoints = frame_keypoints[keypoints_array[i,:,2] > likelihood_threshold]
        
        # Remove any keypoints with NaN values
        valid_keypoints = frame_keypoints[~np.any(np.isnan(frame_keypoints), axis=1)]
        
        # Need at least 3 points for convex hull
        if len(valid_keypoints) < 3:
            areas.append(0)  # No valid area can be calculated
        else:
            hull = ConvexHull(valid_keypoints)
            areas.append(hull.area)
            
    return np.array(areas)

def plot_area_score(area_scores_before, area_scores_after, output_path):
    """
    Plot area scores for a video before and after adaptation
    
    Parameters:
    -----------
    area_scores_before : numpy.ndarray
        Array of area scores for each frame before adaptation
    area_scores_after : numpy.ndarray
        Array of area scores for each frame after adaptation
    """
    # Create figure
    fig, ax = plt.subplots(1, figsize=(9, 3), dpi=300)
    
    # Plot area scores
    ax.plot(area_scores_before, c='dimgray', alpha=0.5, label='w/o adaptation')
    ax.plot(area_scores_after, c='lightcoral', label='w/ adaptation')
    
    # Add scale bar (100 frames or 1/4 of total frames, whichever is smaller)
    scale_size = min(100, len(area_scores_before))
    if scale_size < 10:
        scale_size = 10  # Minimum scale size
        
    scalebar = AnchoredSizeBar(
        ax.transData,
        size=scale_size,
        label=f'{scale_size} frames',
        loc='lower center',
        frameon=False,
        borderpad=-1,
    )
    
    # Remove tick labels
    ax.set_xticklabels([])
    ax.xaxis.set_tick_params(length=0)
    ax.set_yticklabels([])
    ax.yaxis.set_tick_params(length=0)
    
    # Add scale bar and legend
    ax.add_artist(scalebar)
    ax.legend(frameon=False, loc='lower right')
    
    # Remove spines
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)
    
    # Save figure with white background
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, f"area_score_comparison_{timestamp}.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05, transparent=False)
    print(f"Plot saved to {output_path}")
    
    # Show figure
    plt.show()
    
    return output_path

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
    # main()
    
    # Choose which JSON file to analyze (before adaptation)
    before_json_path = "/home/ti_wang/Ti_workspace/PrimatePose/Vis/samples/video_adaption_SA_20250317_2004/videoID_1000013_original_superanimal_topviewmouse_pfm_det_pfm_pose_before_adapt.json"
    
    # JSON file after adaptation
    after_json_path = "/home/ti_wang/Ti_workspace/PrimatePose/Vis/samples/video_adaption_SA_20250317_2004/videoID_1000013_original_superanimal_topviewmouse_snapshot-fasterrcnn_mobilenet_v3_large_fpn-001_snapshot-hrnet_w32-010_after_adapt.json"
    
    # Extract keypoints and calculate area scores for before adaptation
    print("\n=== Processing Before Adaptation Data ===")
    keypoints_before = extract_keypoints_from_json(before_json_path)
    print(f"Keypoints array shape (before): {keypoints_before.shape}")
    area_scores_before = calculate_area_score(keypoints_before)
    print(f"Mean area score (before): {np.mean(area_scores_before):.2f}")    
    # Extract keypoints and calculate area scores for after adaptation
    print("\n=== Processing After Adaptation Data ===")
    keypoints_after = extract_keypoints_from_json(after_json_path)
    print(f"Keypoints array shape (after): {keypoints_after.shape}")
    area_scores_after = calculate_area_score(keypoints_after)
    print(f"Mean area score (after): {np.mean(area_scores_after):.2f}")
    
    # Plot comparison of area scores
    print("\n=== Plotting Area Score Comparison ===")
    area_score_output_folder = "/home/ti_wang/Ti_workspace/PrimatePose/Vis/samples/video_adaption_SA_20250317_2004/area_score_comparison"
    plot_path = plot_area_score(area_scores_before, area_scores_after, area_score_output_folder)
    print(f"Area score comparison plot saved to: {plot_path}")
    
    # # Calculate jitter scores for before adaptation
    # print("\n=== Jitter Score Calculation (Before Adaptation) ===")
    # jitter_before = calculate_jitter_score(before_json_path)
    # if jitter_before is not None:
    #     jitter_x_before = np.nanmean(jitter_before[0])
    #     jitter_y_before = np.nanmean(jitter_before[1])
    #     jitter_combined_before = np.nanmean([jitter_x_before, jitter_y_before])
    #     print(f"Summary (before) - X: {jitter_x_before:.4f}, Y: {jitter_y_before:.4f}, Combined: {jitter_combined_before:.4f}")
    
    # # Calculate jitter scores for after adaptation
    # print("\n=== Jitter Score Calculation (After Adaptation) ===")
    # jitter_after = calculate_jitter_score(after_json_path)
    # if jitter_after is not None:
    #     jitter_x_after = np.nanmean(jitter_after[0])
    #     jitter_y_after = np.nanmean(jitter_after[1])
    #     jitter_combined_after = np.nanmean([jitter_x_after, jitter_y_after])
    #     print(f"Summary (after) - X: {jitter_x_after:.4f}, Y: {jitter_y_after:.4f}, Combined: {jitter_combined_after:.4f}")