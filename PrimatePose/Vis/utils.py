import numpy as np
from typing import Optional, List, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from deeplabcut.utils import auxfun_videos
from deeplabcut.utils.visualization import (
    create_minimal_figure,
    erase_artists,
    get_cmap,
    make_multianimal_labeled_image,
    plot_evaluation_results,
    save_labeled_frame,
)

def plot_gt_and_predictions_PFM(
    image_path: Union[str, Path],
    output_dir: Union[str, Path],
    gt_bodyparts: Optional[np.ndarray] = None,
    pred_bodyparts: Optional[np.ndarray] = None,
    mode: str = "bodypart",
    colormap: str = "rainbow",
    dot_size: int = 12,
    alpha_value: float = 0.8,
    p_cutoff: float = 0.6,
    bounding_boxes: tuple[np.ndarray, np.ndarray] | None = None,
    bounding_boxes_color="k",
    bboxes_pcutoff: float = 0.6,
    skeleton: Optional[List[Tuple[int, int]]] = None,
    keypoint_names: Optional[List[str]] = None,
    keypoint_vis_mask: Optional[List[int]] = None,
    labels: List[str] = ["+", ".", "x"],
) -> None:
    """Plot ground truth and predictions on an image.
    
    Args:
        image_path: Path to the image file
        output_dir: Directory to save the visualization
        gt_bodyparts: Ground truth keypoints array [N, num_keypoints, 3] (x, y, vis_label)
        pred_bodyparts: Predicted keypoints array [N, num_keypoints, 3] (x, y, confidence)
        bounding_boxes: Tuple of (boxes, scores) for bounding box visualization
        dot_size: Size of the keypoint markers
        alpha_value: Transparency for points and lines
        p_cutoff: Confidence threshold for predictions
        mode: How to color the points ("bodypart" or "individual")
        colormap: Matplotlib colormap name
        bbox_color: Color for bounding boxes
        skeleton: List of joint pairs for skeleton visualization
        keypoint_names: List of keypoint names for labeling
        keypoint_vis_mask: List of keypoint indices to show (default: all keypoints visible)
        labels: Marker styles for [ground truth, reliable predictions, unreliable predictions]
    """
    # Set default keypoint visibility mask if not provided
    if pred_bodyparts is not None and keypoint_vis_mask is None:
        keypoint_vis_mask = [1] * pred_bodyparts.shape[1]  # All keypoints visible by default
    
    # Read image and calculate dot size
    frame = auxfun_videos.imread(str(image_path), mode="skimage")
    h, w = frame.shape[:2]
    # Calculate adaptive dot size based on image dimensions
    # Use a logarithmic scale to handle very large or small images better
    diagonal = np.sqrt(w * w + h * h)  # Image diagonal length
    base_size = np.log10(diagonal) * 3  # Logarithmic scaling
    # print("diagonal:", diagonal)
    # Fine-tune the dot size
    if diagonal > 1200:  # High resolution
        dot_size = base_size * 2.0
    elif diagonal < 800:  # Low resolution
        dot_size = base_size * 1.0
    else:  # Medium resolution
        dot_size = base_size
        
    # Ensure dot size stays within reasonable bounds
    dot_size = int(max(4, min(dot_size, 15)))*0.8  # Tighter bounds for dots
    
    # filter out the individuals that without GT keypoints 
    if gt_bodyparts is None:
        tmp_valid_bodyparts = pred_bodyparts
    else:
        tmp_valid_bodyparts = gt_bodyparts
        
    if tmp_valid_bodyparts is not None:
        valid_individuals = []
        for idx in range(tmp_valid_bodyparts.shape[0]):
            # Check if this individual has any valid keypoints
            # A keypoint is valid if its visibility (3rd value) is not -1
            has_valid_keypoints = False
            
            for kp_idx in range(tmp_valid_bodyparts.shape[1]):
                kp = tmp_valid_bodyparts[idx, kp_idx]
                # Check if keypoint is visible
                if kp[2] != -1:
                    has_valid_keypoints = True
                    break  # We found at least one valid keypoint, no need to check more
            
            # Include individual if they have at least one valid keypoint
            if has_valid_keypoints:
                valid_individuals.append(idx)
                # print("add valid individual:", idx)
                
        # print(f"Found {len(valid_individuals)} valid individuals out of {gt_bodyparts.shape[0]}")
        # Filter both ground truth and predictions
        
        # print(f"valid_individuals: {valid_individuals}")
        if valid_individuals:
            if gt_bodyparts is not None:
                gt_bodyparts = gt_bodyparts[valid_individuals]
            if pred_bodyparts is not None:
                pred_bodyparts = pred_bodyparts[valid_individuals]
            if bounding_boxes is not None:
                bounding_boxes = (
                    bounding_boxes[0][valid_individuals],
                    bounding_boxes[1][valid_individuals]
                )
    
    num_pred, num_keypoints = pred_bodyparts.shape[:2]
    
    # print("After filtering:")
    # print("num_pred, num_keypoints:", num_pred, num_keypoints)
    # if gt_bodyparts is not None:
        # print("gt_bodyparts shape:", gt_bodyparts.shape)
    
    # Create figure with optimal settings
    fig, ax = create_minimal_figure()
    fig.set_size_inches(w/100, h/100)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.invert_yaxis()
    ax.imshow(frame, "gray")

    # Set up colors based on mode
    if mode == "bodypart":
        num_colors = num_keypoints
        # if pred_unique_bodyparts is not None:
        #     num_colors += pred_unique_bodyparts.shape[1]
        colors = get_cmap(num_colors, name=colormap)
        # print("colors:", colors)
    # predictions = pred_bodyparts.swapaxes(0, 1)
    # ground_truth = gt_bodyparts.swapaxes(0, 1)
    elif mode == "individual":
        colors = get_cmap(num_pred + 1, name=colormap)
        # predictions = pred_bodyparts
        # ground_truth = gt_bodyparts
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # print("bounding_boxes:", bounding_boxes)
    
    # Draw bounding boxes if provided
    if bounding_boxes is not None:
        # print(f"bounding_boxes: {bounding_boxes}")
        for bbox, bbox_score in zip(bounding_boxes[0], bounding_boxes[1]):
            bbox_origin = (bbox[0], bbox[1])
            (bbox_width, bbox_height) = (bbox[2], bbox[3])
            rect = patches.Rectangle(
                bbox_origin,
                bbox_width,
                bbox_height,
                linewidth=2,
                edgecolor=bounding_boxes_color,
                facecolor='none',
                linestyle="--" if bbox_score < bboxes_pcutoff else "-"
            )
            ax.add_patch(rect)

    # Track existing text positions to avoid overlap
    existing_text_positions = []
    scale_factor = min(w, h) / 1000  # Normalize scale factor based on image size

    plot_individual = False
    
    if plot_individual:
        # Save individual plots for each animal
        for idx_individual in range(num_pred):
            # print("plot individual:", idx_individual)
            # Create a new figure for each individual
            fig_ind, ax_ind = create_minimal_figure()
            fig_ind.set_size_inches(w/100, h/100)
            ax_ind.set_xlim(0, w)
            ax_ind.set_ylim(0, h)
            ax_ind.invert_yaxis()
            ax_ind.imshow(frame, "gray")
            
            # Draw bounding box for this individual if available
            if bounding_boxes is not None:
                bbox = bounding_boxes[0][idx_individual]
                bbox_score = bounding_boxes[1][idx_individual]
                bbox_origin = (bbox[0], bbox[1])
                (bbox_width, bbox_height) = (bbox[2], bbox[3])
                rect = patches.Rectangle(
                    bbox_origin,
                    bbox_width,
                    bbox_height,
                    linewidth=2,
                    edgecolor=bounding_boxes_color,
                    facecolor='none',
                    linestyle="--" if bbox_score < bboxes_pcutoff else "-"
                )
                ax_ind.add_patch(rect)
            
            # Reset text positions for each individual
            existing_text_positions = []
            
            # Plot keypoints for this individual
            for idx_keypoint in range(num_keypoints):
                if keypoint_vis_mask[idx_keypoint]:
                    
                    keypoint_confidence = pred_bodyparts[idx_individual, idx_keypoint, 2]
                    # print("keypoint_confidence_individual:", keypoint_confidence)
                    if keypoint_confidence > p_cutoff:
                        x_kp = pred_bodyparts[idx_individual, idx_keypoint, 0]
                        y_kp = pred_bodyparts[idx_individual, idx_keypoint, 1]
                        
                        ax_ind.plot(
                            x_kp, 
                            y_kp, 
                            labels[1] if keypoint_confidence > p_cutoff else labels[2], 
                            color=colors(idx_keypoint), 
                            alpha=alpha_value,
                            markersize=dot_size
                        )

                        if keypoint_names is not None:
                            # Calculate and adjust text position
                            x_text = x_kp - (10 * scale_factor)
                            y_text = y_kp - (15 * scale_factor)
                            x_text = min(max(0, x_text), w - 100)
                            y_text = min(max(0, y_text), h - 10)
                            
                            while any(abs(x_text - ex) < 50 * scale_factor and abs(y_text - ey) < 20 * scale_factor 
                                    for ex, ey in existing_text_positions):
                                y_text += 20 * scale_factor
                                if y_text > h - 10:
                                    y_text = y_kp
                                    x_text += 50 * scale_factor
                            
                            existing_text_positions.append((x_text, y_text))
                            
                            ax_ind.text(
                                x_text,
                                y_text,
                                keypoint_names[idx_keypoint], 
                                color=colors(idx_keypoint), 
                                alpha=alpha_value,
                                fontsize=dot_size * 0.8
                            )
                            
                        # Plot ground truth for this individual
                        if gt_bodyparts is not None:
                            if gt_bodyparts[idx_individual, idx_keypoint, 2] != -1:
                                ax_ind.plot(
                                    gt_bodyparts[idx_individual, idx_keypoint, 0], 
                                    gt_bodyparts[idx_individual, idx_keypoint, 1], 
                                    labels[0], 
                                    color=colors(idx_keypoint), 
                                    alpha=alpha_value,
                                    markersize=dot_size
                                )
            
            # Save individual plot
            if num_pred > 1:
                # Add index for multi-animal images
                output_path = Path(output_dir) / f"{Path(image_path).stem}_animal_{idx_individual}_predictions.png"
            else:
                # No index needed for single animal
                output_path = Path(output_dir) / f"{Path(image_path).stem}_predictions.png"
                
            plt.savefig(
                output_path,
                bbox_inches='tight',
                pad_inches=0,
                transparent=False
            )
            plt.close(fig_ind)
    
    # Original combined plot
    for idx_individual in range(num_pred):
        for idx_keypoint in range(num_keypoints):
            if pred_bodyparts is not None and keypoint_vis_mask[idx_keypoint]:
                # if the keypoint is allowed to be shown and the prediction is reliable
                keypoint_confidence = pred_bodyparts[idx_individual, idx_keypoint, 2]
                if keypoint_confidence > p_cutoff:
                    pred_label = labels[1]
                else:
                    pred_label = labels[2]
                if keypoint_confidence > p_cutoff:
                    x_kp = pred_bodyparts[idx_individual, idx_keypoint, 0]
                    y_kp = pred_bodyparts[idx_individual, idx_keypoint, 1]
                    
                    ax.plot(
                        x_kp, 
                        y_kp, 
                        pred_label, 
                        color=colors(idx_keypoint), 
                        alpha=alpha_value,
                        markersize=dot_size
                    )

                    if keypoint_names is not None:
                        # Calculate initial text position
                        x_text = x_kp - (10 * scale_factor)
                        y_text = y_kp - (15 * scale_factor)
                        
                        # Ensure text stays within image bounds
                        x_text = min(max(0, x_text), w - 100)
                        y_text = min(max(0, y_text), h - 10)
                        
                        # Avoid overlapping with existing text
                        while any(abs(x_text - ex) < 50 * scale_factor and abs(y_text - ey) < 20 * scale_factor 
                                for ex, ey in existing_text_positions):
                            y_text += 20 * scale_factor
                            if y_text > h - 10:  # If we run out of vertical space
                                y_text = pred_bodyparts[idx_individual, idx_keypoint, 1]  # Reset to original y
                                x_text += 50 * scale_factor  # Move text horizontally instead
                        
                        # Record this position
                        existing_text_positions.append((x_text, y_text))
                        
                        ax.text(
                            x_text,
                            y_text,
                            keypoint_names[idx_keypoint], 
                            color=colors(idx_keypoint), 
                            alpha=alpha_value,
                            fontsize=dot_size * 0.5
                        )

                    # plot ground truth
                    if gt_bodyparts is not None:
                        if gt_bodyparts[idx_individual, idx_keypoint, 2] != -1:
                            ax.plot(
                                gt_bodyparts[idx_individual, idx_keypoint, 0], 
                                gt_bodyparts[idx_individual, idx_keypoint, 1], 
                                labels[0], 
                                color=colors(idx_keypoint), 
                                alpha=alpha_value,
                                markersize=dot_size*0.5
                            )
                
    # Save the figure
    output_path = Path(output_dir) / f"{Path(image_path).stem}_predictions.png"
    # save_labeled_frame(fig, str(image_path), str(output_dir), belongs_to_train=False)
    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches='tight',
        pad_inches=0,
        transparent=False
    )
    erase_artists(ax)
    plt.close()