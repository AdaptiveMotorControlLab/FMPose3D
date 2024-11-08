"""Evaluating COCO models"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import numpy as np
import torch
import os
from deeplabcut.pose_estimation_pytorch import COCOLoader
from deeplabcut.pose_estimation_pytorch.apis.evaluate import evaluate
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.task import Task

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from analyse_json.split_v8_json import get_file_name_from_image_id

PRIMATE_COLOR_MAP = {
    "head": (0, 180, 0), # wait
    "neck": (0, 0, 180), # wait
    "nose": (255, 0, 0), # "
    "mouth_front_top": (0, 255, 0), # "upper_jaw"
    "mouth_front_bottom": (0, 0, 255), # "lower_jaw"
    "mouth_back_right": (255, 255, 0), # "mouth_end_right"
    "mouth_back_left": (255, 0, 255), # "mouth_end_left"
    "right_ear": (128, 0, 0), # "right_earbase"
    "left_ear": (0, 128, 128), # "left_earbase": (0, 128, 128),
    "neck": (255, 128, 0), # "neck_base"
    "upper_back": (128, 255, 0), # "neck_end"
    "throat_base": (0, 255, 128), # "throat_base"
    "upper_back": (255, 0, 128), # "back_base"
    "lower_back": (255, 128, 128), # "back_end"
    "torso_mid_back": (128, 255, 255), # "back_middle"
    "root_tail": (128, 0, 64), # "tail_base"
    "end_tail": (64, 0, 128), # "tail_end"
    "left_shoulder": (128, 64, 0), # "front_left_thai"
    "left_elbow": (64, 128, 0), # "front_left_knee"
    "left_hand": (0, 64, 128), # "front_left_paw"
    "right_shoulder": (255, 64, 64), # "front_right_thai"
    "right_elbow": (64, 255, 64), # "front_right_knee"
    "left_foot": (255, 255, 64), # "back_left_paw"
    "left_hip": (255, 64, 255), # "back_left_thai"
    "left_knee": (192, 64, 192), # "back_left_knee"
    "right_knee": (192, 192, 64), # "back_right_knee"
    "right_foot": (64, 192, 192), # "back_right_paw"
    "body_center": (192, 192, 192), #  "belly_bottom"
    "right_hip": (128, 64, 64), # "body_middle_right"`
    "left_hip": (64, 128, 128),  # "body_middle_left"
    "right_hand": (64, 64, 255), # "front_right_paw"
    "left_wrist": (128, 0, 128),
    "right_wrist": (0, 255, 255),
    "forehead": (0, 128, 0),
    "center_hip": (64, 255, 255),
    "left_ankle": (128, 128, 128),
    "right_ankle": (0, 0, 128),
    "mid_tail": (192, 192, 192),
    "mid_end_tail": (0, 128, 255), 
    "right_eye": (0, 255, 255),
    "left_eye": (128, 0, 128),
    # "right_earend": (0, 128, 0),
    # "right_antler_base": (0, 0, 128),
    # "right_antler_end": (128, 128, 0),
    # "left_earend": (192, 192, 192),
    # "left_antler_base": (128, 128, 128),
    # "left_antler_end": (64, 64, 64),
    # "throat_end": (0, 128, 255), 
    # "front_right_paw": (64, 64, 255),
    # "back_right_thai": (64, 255, 255),
}

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

from PIL import Image
def visualize_predictions(json_path="path_to_your_predictions_file.json", image_dir="/mnt/data/tiwang/v8_coco/images", num_samples=5, test_file_json="test.json", color = None):
    """
    Visualize a specified number of samples from COCO predictions and save the plots.

    Args:
        json_path (str): Path to the JSON file with COCO predictions.
        image_dir (str): Directory where the images corresponding to image IDs are stored.
        num_samples (int): Number of samples to visualize. Defaults to 5.
    """
    # Load the coco_predictions JSON file
    with open(json_path, "r") as f:
        coco_predictions = json.load(f)

    # Determine the output directory for saving images
    output_dir = os.path.join(os.path.dirname(json_path), "predictions_visualizations")
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
    
    keypoint_labels = list(PRIMATE_COLOR_MAP.keys())

    # Iterate through the first few samples and plot each one
    for i, prediction in enumerate(coco_predictions[:num_samples]):
        # Extract data from the prediction
        image_id = prediction["image_id"]
        category_id = prediction["category_id"]
        keypoints = np.array(prediction["keypoints"]).reshape(-1, 3)  # reshape for (x, y, visibility)
        score = prediction["score"]
        bbox = prediction["bbox"]
        bbox_score = prediction["bbox_scores"][0] if prediction["bbox_scores"] else None

        # Load the corresponding image
        file_name = get_file_name_from_image_id(json_path=test_file_json, image_id=image_id)
        image_path = os.path.join(image_dir, f"{file_name}")  # Adjust extension if different
        image = Image.open(image_path)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image)
        ax.set_title(f"Image ID: {image_id}, bbox_score: {bbox_score:.2f}, Score: {score:.2f}")

        # Plot bounding box
        rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        # ax.text(bbox[0], bbox[1] - 10, f"Score: {bbox_score:.2f}" if bbox_score else "No bbox score", color="red")

        # Plot keypoints with color-coding
        for idx, kp in enumerate(keypoints):
            if kp[2] > 0:  # visibility flag
                label = keypoint_labels[idx] if idx < len(keypoint_labels) else "unknown"
                color = PRIMATE_COLOR_MAP.get(label, "blue")  # Default to blue if label not found
                ax.plot(kp[0], kp[1], "o", markersize=5, color=np.array(color) / 255.0)

        # Save the figure to the output directory
        output_path = os.path.join(output_dir, f"predict_{file_name}")
        plt.savefig(output_path)
        plt.close(fig)

    print(f"Visualizations saved in: {output_dir}")

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
    #for mode in ["train", "test"]:
    for mode in ["test"]:    
        scores, predictions = evaluate(
            pose_task=Task(loader.model_cfg["method"]),
            pose_runner=pose_runner,
            loader=loader,
            mode=mode,
            detector_runner=detector_runner,
            pcutoff=pcutoff,
        )
        print("finished evaluating")
        coco_predictions = loader.predictions_to_coco(predictions, mode=mode)
        model_name = Path(snapshot_path).stem
        if detector_path is not None:
            model_name += Path(detector_path).stem
        predictions_file = output_path / f"{model_name}-{mode}-predictions.json"
        with open(predictions_file, "w") as f:
            json.dump(coco_predictions, f, indent=4)
        
        color_map = PRIMATE_COLOR_MAP
        visualize_predictions(json_path=predictions_file, num_samples=5, test_file_json=test_file, color=color_map)
        
        annotation_types = ["keypoints"]
        if detector_runner is not None:
            annotation_types.append("bbox")

        ground_truth = loader.load_data(mode=mode)
        for annotation_type in annotation_types:
            kpt_oks_sigmas = oks_sigma * np.ones(parameters.num_joints)
            pycocotools_evaluation(
                ground_truth=ground_truth,
                predictions=coco_predictions,
                kpt_oks_sigmas=kpt_oks_sigmas,
                annotation_type=annotation_type,
            )

        print(80 * "-")
        print(f"{mode} results")
        for k, v in scores.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root")
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