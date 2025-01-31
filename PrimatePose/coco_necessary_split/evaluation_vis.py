"""Evaluating COCO models"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.patches as patches
from deeplabcut.utils import auxfun_videos
from deeplabcut.pose_estimation_pytorch import COCOLoader
from deeplabcut.pose_estimation_pytorch.apis.evaluate import evaluate, plot_gt_and_predictions
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.pose_estimation_pytorch.apis.evaluate import visualize_predictions
from deeplabcut.utils.auxfun_videos import imread
from deeplabcut.utils.visualization import get_cmap
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from deeplabcut.utils.visualization import (
    create_minimal_figure,
    erase_artists,
    get_cmap,
    make_multianimal_labeled_image,
    plot_evaluation_results,
    save_labeled_frame,
)

def compute_brightness(img, x, y, radius=20):
    crop = img[
        max(0, y - radius): min(img.shape[0], y + radius),
        max(0, x - radius): min(img.shape[1], x + radius),
        :
    ]
    return np.mean(crop)

def pycocotools_evaluation(
    kpt_oks_sigmas: list[int],
    ground_truth: dict,
    predictions: list[dict],
    annotation_type: str,
) -> None:
    """Evaluation of models using Pycocotools

    Evaluates the predictions using OKS sigma 0.1, margin 0 and prints the results to
    the console.

    Args:
        kpt_oks_sigmas: the OKS sigma for each keypoint
        ground_truth: the ground truth data, in COCO format
        predictions: the predictions, in COCO format
        annotation_type: {"bbox", "keypoints"} the annotation type to evaluate
    """
    print(80 * "-")
    print(f"Attempting `pycocotools` evaluation for {annotation_type}!")
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco = COCO()
        coco.dataset["annotations"] = ground_truth["annotations"]
        coco.dataset["categories"] = ground_truth["categories"]
        coco.dataset["images"] = ground_truth["images"]
        coco.createIndex()

        coco_det = coco.loadRes(predictions)
        coco_eval = COCOeval(coco, coco_det, iouType=annotation_type)
        coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    except Exception as err:
        print(f"Could not evaluate with `pycocotools`: {err}")
    finally:
        print(80 * "-")

# idx                 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36          
keypoint_vis_mask = [ 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1]  # Original
keypoint_vis_mask = [ 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1]  # Version 2
keypoint_vis_mask = [ 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1]  # Version 3

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

def main(
    project_root: str,
    train_file: str,
    test_file: str,
    pytorch_config_path: str,
    device: str | None,
    snapshot_path: str,
    detector_path: str | None,
    pcutoff: float,
    oks_sigma: float,
    gpus: list[int] | None,
):
    loader = COCOLoader(
        project_root=project_root,
        model_config_path=pytorch_config_path, # the path for pytorch_config.yaml
        train_json_filename=train_file,
        test_json_filename=test_file,
    )
    parameters = loader.get_dataset_parameters()
    if device is not None:
        loader.model_cfg["device"] = device

    # print("max_num_animals:", parameters.max_num_animals)
    
    pose_runner, detector_runner = get_inference_runners(
        model_config=loader.model_cfg,
        snapshot_path=snapshot_path,
        max_individuals=parameters.max_num_animals,
        num_bodyparts=parameters.num_joints,
        num_unique_bodyparts=parameters.num_unique_bpts,
        with_identity=False,
        transform=None,
        detector_path=detector_path,
        detector_transform=None,
    )
    
    output_path = Path(pytorch_config_path).parent.parent / "results"
    output_path.mkdir(exist_ok=True)
    print(output_path)
    #for mode in ["train", "test"]:
    
    print("detector_runner:", detector_runner)
    
    for mode in ["test"]:    
        scores, predictions = evaluate(
            pose_task=Task(loader.model_cfg["method"]),
            pose_runner=pose_runner,
            loader=loader,
            mode=mode,
            detector_runner=detector_runner,
            pcutoff=pcutoff,
        ) 
      
        print("scores:", scores)
        
        # predictions:      
        # bodyparts: (1, 37, 3)
        # bboxes: (1, 4)
        # bbox_scores: (1,)
        
        # get ground truth 
        gt_keypoints = loader.ground_truth_keypoints(mode) 
        
        print("finished evaluating")
        
        coco_predictions = loader.predictions_to_coco(predictions, mode=mode)
        model_name = Path(snapshot_path).stem
        if detector_path is not None:
            model_name += Path(detector_path).stem
        predictions_file = output_path / f"{model_name}-{mode}-predictions.json"
        with open(predictions_file, "w") as f:
            json.dump(coco_predictions, f, indent=4)
        
        print(80 * "-")
        print(f"{mode} results")
        
        # Define the path for the results file
        results_file_path = output_path / "test_results.txt"
        
        # Print and write the results
        print(f"\nResults from model: {snapshot_path}")
        print("Evaluation scores:")
        
        # Open the file in append mode
        with open(results_file_path, "a") as f:
            # Write a separator line and model path
            separator = "\n" + "="*50 + "\n"
            f.write(separator)
            f.write(f"Model: {snapshot_path}\n")
            
            # Write the scores
            for k, v in scores.items():
                result_line = f"  {k}: {v}\n"
                print(result_line.strip())  # Print to console
                f.write(result_line)  # Write to file
        
        visualize_PFM_predictions(
            predictions=predictions,
            ground_truth=gt_keypoints,
            output_dir=output_path,
            num_samples=10,  # Added to limit visualization to 10 samples
            random_select=True,
            keypoint_vis_mask=keypoint_vis_mask,
            plot_bboxes=True,
            keypoint_names=keypoint_name_simplified
        )
        
def visualize_PFM_predictions(
    predictions: Dict[str, Dict],
    ground_truth: Dict[str, np.ndarray],
    output_dir: Optional[Union[str, Path]] = None,
    num_samples: Optional[int] = None,
    random_select: bool = False,
    plot_bboxes: bool = True,
    skeleton: Optional[List[Tuple[int, int]]] = None,
    keypoint_vis_mask: Optional[List[int]] = None,
    keypoint_names: Optional[List[str]] = None,
    confidence_threshold: float = 0.6
) -> None:
    """Visualize model predictions alongside ground truth keypoints with additional PFM-specific configurations."""
    # Setup output directory and logging
    output_dir = Path(output_dir or "predictions_visualizations")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Configure logging with a unique handler
    log_file = output_dir / "visualization.log"
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger('PFM_visualization')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info(f"Starting visualization process. Output directory: {output_dir}")

    # Select images to process efficiently
    image_paths = list(predictions.keys())
    if num_samples and num_samples < len(image_paths):
            if random_select:
                image_paths = np.random.choice(
                    image_paths, num_samples, replace=False
                ).tolist()
            else:
                image_paths = image_paths[:num_samples]

    # Process each selected image
    for image_path in image_paths:
        # Get prediction and ground truth data
        pred_data = predictions[image_path]
        gt_keypoints = ground_truth[image_path]  # Shape: [N, num_keypoints, 3]

        # Process predicted keypoints
        pred_keypoints = pred_data["bodyparts"]

        if plot_bboxes:
            bboxes = predictions[image_path].get("bboxes", None)
            bbox_scores = predictions[image_path].get("bbox_scores", None)
            bounding_boxes = (
                (bboxes, bbox_scores)
                if bbox_scores is not None and bbox_scores is not None
                else None
            )
        else:
            bounding_boxes = None

        # Generate visualization
        plot_gt_and_predictions_PFM(
            image_path=image_path,
            output_dir=output_dir,
            gt_bodyparts=gt_keypoints,
            pred_bodyparts=pred_keypoints,
            bounding_boxes=bounding_boxes,
            skeleton=skeleton,
            keypoint_names=keypoint_names,
            p_cutoff=confidence_threshold,
            keypoint_vis_mask=keypoint_vis_mask,  # Pass the mask to plotting function
        )
        logger.info(f"Successfully visualized predictions for {image_path}")

    # Clean up logging handler
    logger.removeHandler(handler)
    handler.close()

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
        keypoint_vis_mask: List of keypoint indices to show
        labels: Marker styles for [ground truth, reliable predictions, unreliable predictions]
    """
    # Read image and calculate dot size
    frame = auxfun_videos.imread(str(image_path), mode="skimage")
    h, w = frame.shape[:2]
    # print("h, w:", h, w)
    # print("image_name:", Path(image_path).stem)
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
    dot_size = int(max(4, min(dot_size, 15)))  # Tighter bounds for dots
    
    # filter out the individuals that without GT keypoints 
    # if all the keypoints are 0, then this individual will be delete
    if gt_bodyparts is not None:
        valid_individuals = []
        for idx in range(gt_bodyparts.shape[0]):
            # Check if this individual has any valid keypoints
            # A keypoint is valid if:
            # 1. its visibility (3rd value) is not -1
            # 2. not all keypoints are [0, 0, 0]
            has_valid_keypoints = False
            all_zeros = True
            
            for kp_idx in range(gt_bodyparts.shape[1]):
                kp = gt_bodyparts[idx, kp_idx]
                # Check if keypoint is not [0, 0, 0]
                if not (kp[0] == 0 and kp[1] == 0 and kp[2] == 0):
                    all_zeros = False
                # Check if keypoint is visible
                if kp[2] != -1:
                    has_valid_keypoints = True
            
            # Only include individual if they have valid keypoints and not all zeros
            if has_valid_keypoints and not all_zeros:
                valid_individuals.append(idx)
        
        print(f"Found {len(valid_individuals)} valid individuals out of {gt_bodyparts.shape[0]}")
        
        # Filter both ground truth and predictions
        if valid_individuals:
            gt_bodyparts = gt_bodyparts[valid_individuals]
            pred_bodyparts = pred_bodyparts[valid_individuals]
            if bounding_boxes is not None:
                bounding_boxes = (
                    bounding_boxes[0][valid_individuals],
                    bounding_boxes[1][valid_individuals]
                )
    
    num_pred, num_keypoints = pred_bodyparts.shape[:2]
    
    print("After filtering:")
    print("num_pred, num_keypoints:", num_pred, num_keypoints)
    if gt_bodyparts is not None:
        print("gt_bodyparts shape:", gt_bodyparts.shape)
    
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

    print("bounding_boxes:", bounding_boxes)
    
    # Draw bounding boxes if provided
    if bounding_boxes is not None:
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
                linestyle="--" if bbox_score < bbox else "-"
            )
            ax.add_patch(rect)

    # Track existing text positions to avoid overlap
    existing_text_positions = []
    scale_factor = min(w, h) / 1000  # Normalize scale factor based on image size

    plot_individual = True
    
    if plot_individual:
        # Save individual plots for each animal
        for idx_individual in range(num_pred):
            print("plot individual:", idx_individual)
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
                    print("keypoint_confidence_individual:", keypoint_confidence)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root")
    parser.add_argument("--pytorch_config_path")
    parser.add_argument("--snapshot_path")
    parser.add_argument("--train_file", default="train.json")
    parser.add_argument("--test_file", default="test.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--detector_path", default=None)
    parser.add_argument("--gpus", default=None, nargs="+", type=int)
    parser.add_argument("--pcutoff", type=float, default=0.6)
    parser.add_argument("--oks_sigma", type=float, default=0.1)
    args = parser.parse_args()
    main(
        args.project_root,
        args.train_file,
        args.test_file,
        args.pytorch_config_path,
        args.device,
        args.snapshot_path,
        args.detector_path,
        args.pcutoff,
        args.oks_sigma,
        args.gpus,
    )