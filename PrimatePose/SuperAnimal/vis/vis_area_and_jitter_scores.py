import os 
import pandas as pd
import json
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
from datetime import datetime
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

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
    Reference:
        https://github.com/AdaptiveMotorControlLab/DLC-ModelZoo/blob/7e346b59a4d2947e365c3e1e550c0a5d5965c0fe/modelzoo/metric/unsupervised_metrics.py#L173
    """
    # Extract just the x,y coordinates (first two channels)
    coordinates = keypoints_array[:, :, :2]  # Shape: [Frame, num_joints, 2]
    
    # Calculate area for each frame
    areas = []
    for frame_keypoints in coordinates:
        # Remove any keypoints with NaN values
        valid_keypoints = frame_keypoints[~np.any(np.isnan(frame_keypoints), axis=1)]
        
        # Need at least 3 points for convex hull
        if len(valid_keypoints) < 3:
            areas.append(0)  # No valid area can be calculated
        else:
            hull = ConvexHull(valid_keypoints)
            areas.append(hull.area)
            
    return np.array(areas)

def plot_area_score(area_scores_before, area_scores_after, output_folder_path):
    """
    Plot area scores for a video before and after adaptation
    
    Parameters:
    -----------
    area_scores_before : numpy.ndarray
        Array of area scores for each frame before adaptation
    area_scores_after : numpy.ndarray
        Array of area scores for each frame after adaptation
    Reference:
        https://github.com/AdaptiveMotorControlLab/modelzoo-figures/blob/main/figures/Figure3.ipynb
    
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
    os.makedirs(output_folder_path, exist_ok=True)
    output_path = os.path.join(output_folder_path, f"area_score_comparison_{timestamp}.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05, transparent=False)
    print(f"Plot saved to {output_path}") 
    
    return output_path

def calculate_jitter_score(json_path):
    """
    Calculate per-frame jitter scores from pose estimation results.
    
    Jitter score quantifies tracking instability by measuring unexpected frame-to-frame 
    movement after removing global motion trends. Lower scores indicate smoother, 
    more stable tracking.
    
    Algorithm:
    1. Load pose estimation results from JSON file
    2. Extract keypoint coordinates (x, y) for all individuals across all frames
    3. Calculate frame-to-frame coordinate differences
    4. Remove global motion (average movement) to isolate tracking jitter
    5. Compute per-frame jitter as mean absolute deviation across all keypoints
    
    Args:
        json_path (str): Path to JSON file containing pose estimation results.
                        Expected format: list of frames, each containing 'bodyparts' 
                        with keypoint coordinates [x, y, confidence].
    
    Returns:
        numpy.ndarray: 1D array of per-frame jitter scores (shape: n_frames-1).
                      Each element represents the combined jitter score for one frame.
                      Returns None if no valid frames found.
                      
    Note:
        Output length is n_frames-1 since jitter is calculated from frame differences.
        Higher values indicate more jittery/unstable tracking.
    Reference:
        https://github.com/AdaptiveMotorControlLab/DLC-ModelZoo/blob/generic/modelzoo/metric/unsupervised_metrics.py#L173
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
            for keypoint in individual:
                x, y, conf = keypoint[0], keypoint[1], keypoint[2]
                
                frame_keypoints.append([x, y])
                
        # If we have keypoints for this frame, add to the list
        if frame_keypoints:
            all_kpts.append(np.array(frame_keypoints)[np.newaxis, :, :])
        
    # Convert to numpy array with shape (frames, keypoints, 3)
    print("all_kpts:", len(all_kpts))
    print("all_kpts[0]:", all_kpts[0].shape)
    all_kpts = np.concatenate(all_kpts, axis=0)
    
    # Extract x and y coordinates
    print("all_kpts:", all_kpts.shape)
    xs = all_kpts[:, :, 0]
    print("xs.shape:", xs.shape)
    ys = all_kpts[:, :, 1]
    
    # Calculate frame-to-frame differences
    diff_xs = np.diff(xs, axis=0)
    diff_ys = np.diff(ys, axis=0)
    print("diff_xs:", diff_xs.shape)
    print("diff_ys:", diff_ys.shape)
    
    # Calculate average movement
    average_xs = np.nanmean(diff_xs)
    average_ys = np.nanmean(diff_ys)
    
    print(f"Average X movement: {average_xs:.4f} pixels")
    print(f"Average Y movement: {average_ys:.4f} pixels")
    
    # Center movement by subtracting the average
    diff_xs -= average_xs
    diff_ys -= average_ys
    
    # Calculate jitter score (absolute differences)
    # average across all samples, all keypoints, and take absolute value
    jitter_x = np.nanmean(np.abs(diff_xs))
    jitter_y = np.nanmean(np.abs(diff_ys))
    
    # Combined jitter score
    jitter_combined = np.nanmean([jitter_x, jitter_y])
    
    print(f"X jitter score: {jitter_x:.4f} pixels")
    print(f"Y jitter score: {jitter_y:.4f} pixels")
    print(f"Combined jitter score: {jitter_combined:.4f} pixels")
    
    ret = np.array([np.abs(diff_xs), np.abs(diff_ys)])
    ret = np.swapaxes(ret, 0, 1)
    ret = np.nanmean(ret, axis=(1, 2))
    return ret
    # return np.array([np.abs(diff_xs), np.abs(diff_ys)])

def plot_jitter_score(jitter_before_scores, jitter_after_scores, dataset_name, output_path):

    data_for_plot = []
    
    # Add before adaptation data
    for score in jitter_before_scores:
        data_for_plot.append({
            'Dataset': dataset_name,
            'Time': 'Before',
            'Jitter Score': score
        })
    
    # Add after adaptation data  
    for score in jitter_after_scores:
        data_for_plot.append({
            'Dataset': dataset_name,
            'Time': 'After', 
            'Jitter Score': score
        })
    
    # Create DataFrame
    df_plot = pd.DataFrame(data_for_plot)
    print(f"\nDataFrame for plotting:")
    print(f"Shape: {df_plot.shape}")
    print(df_plot.head(10))
    print(f"\nSummary statistics:")
    print(df_plot.groupby('Time')['Jitter Score'].describe())
    
    # Create the boxplot
    plt.figure(figsize=(4, 5))
    
    # Create boxplot similar to Figure3
    # ax = sns.boxplot(
    #     data=df_plot, x="Time", y="Jitter Score",
    #     whis=[5, 95], width=.5, palette=['grey', 'lightpink'],
    #     showfliers=False
    # )
    ax = sns.boxplot(
    data=df_plot, x="Time", y="Jitter Score",
    whis=[0, 100], width=.5, palette=['grey', 'lightpink'],
    showfliers=False
    )
    sns.despine(trim=False, top=True, right=True, ax=ax, offset=10)
    
    plt.title(f'Jitter Score Comparison: {dataset_name}')
    plt.xlabel('Time')
    plt.ylabel('Jitter Score (pixels)')
    
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directory if it doesn't exist
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    
    folder_path = "/home/ti_wang/Ti_workspace/PrimatePose/SuperAnimal/video_inference/test_video_data/VA_customed_models_max_individuals_1_20250619_1057"
    before_adaption_json_path = glob.glob(os.path.join(folder_path, "*_before_adapt.json"))[0]
    after_adaption_json_path = glob.glob(os.path.join(folder_path, "*_after_adapt.json"))[0]

    flag_calculate_jitter_score = False

    if flag_calculate_jitter_score:
        # Calculate jitter scores for before adaptation
        print("\n=== Jitter Score Calculation (Before Adaptation) ===")
        jitter_before_scores = calculate_jitter_score(before_adaption_json_path)
        
        # Calculate jitter scores for after adaptation
        print("\n=== Jitter Score Calculation (After Adaptation) ===")
        jitter_after_scores = calculate_jitter_score(after_adaption_json_path)
        
        # Create boxplot visualization
        image_path = os.path.join("./figures/jitter_score_comparison.png")
        plot_jitter_score(jitter_before_scores, jitter_after_scores, dataset_name="Custom Dataset", output_path=image_path)
    
    flag_calculate_area_score = True
    if flag_calculate_area_score:
        keypoints_before = extract_keypoints_from_json(before_adaption_json_path)
        keypoints_after = extract_keypoints_from_json(after_adaption_json_path)
        
        area_before_scores = calculate_area_score(keypoints_before)
        area_after_scores = calculate_area_score(keypoints_after)
        
        area_score_image_folder_path = os.path.join("./figures")
        plot_area_score(area_before_scores, area_after_scores, output_folder_path=area_score_image_folder_path)
        
        