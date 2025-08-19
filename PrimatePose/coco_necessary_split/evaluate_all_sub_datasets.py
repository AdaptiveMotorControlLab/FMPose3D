"""Evaluating COCO models"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.patches as patches
from deeplabcut.utils import auxfun_videos
from deeplabcut.pose_estimation_pytorch import COCOLoader
from deeplabcut.pose_estimation_pytorch.apis.evaluation import evaluate, plot_gt_and_predictions
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.pose_estimation_pytorch.apis.evaluation import visualize_predictions, visualize_predictions_PFM
from deeplabcut.utils.auxfun_videos import imread
from deeplabcut.utils.visualization import get_cmap
import sys
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).resolve().parent.parent))
from deeplabcut.utils.visualization import (
    create_minimal_figure,
    erase_artists,
    get_cmap,
    make_multianimal_labeled_image,
    plot_evaluation_results,
    save_labeled_frame,
)
import os

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
keypoint_vis_mask = [ 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]  # Version 3

keypoint_vis_mask = [ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  

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


def evaluate_single_dataset(
    project_root: str,
    dataset_name: str,
    train_file: str,
    test_file: str,
    pytorch_config_path: str,
    device: str | None,
    snapshot_path: str,
    detector_path: str | None,
    output_folder: str,
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
    # Force-disable detector when no detector snapshot is provided to avoid use the default detector from torchvision.
    if detector_path is None:
        detector_runner = None
    print("detector_runner:", detector_runner)
    
    output_path = output_folder
    output_path.mkdir(exist_ok=True)
    
    
    for mode in ["test"]:    
        scores, predictions = evaluate(
            # pose_task=Task(loader.model_cfg["method"]),
            pose_runner=pose_runner,
            loader=loader,
            mode=mode,
            detector_runner=detector_runner,
            pcutoff=pcutoff,
        ) 
        # print("scores:", scores)
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
        results_file_path = output_path.parent / "test_results.txt"
        
        # Print and write the results
        print(f"\nResults from model: {snapshot_path}")
        print("Evaluation scores:")
        
        # Open the file in append mode
        with open(results_file_path, "a") as f:
            # Write a separator line and model path
            separator = "\n" + "="*50 + "\n"
            f.write(separator)
            f.write(f"Dataset: {dataset_name}\n")
            
            # Write the scores
            for k, v in scores.items():
                result_line = f"  {k}: {v}\n"
                print(result_line.strip())  # Print to console
                f.write(result_line)  # Write to file
                
        visualize_predictions_PFM(
            predictions=predictions,
            ground_truth=gt_keypoints,
            output_dir=output_path,
            num_samples=20,  # Added to limit visualization to 10 samples
            random_select=True,
            # keypoint_vis_mask=keypoint_vis_mask,
            # plot_bboxes=True,
            skeleton=True
            # keypoint_names=keypoint_name_simplified
        )
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root")
    parser.add_argument("--pytorch_config_path")
    parser.add_argument("--snapshot_path")
    # parser.add_argument("--train_file", default="train.json")
    parser.add_argument("--test_folder", default="xxxx")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector_path", default=None)
    parser.add_argument("--gpus", default=None, nargs="+", type=int)
    parser.add_argument("--pcutoff", type=float, default=0.6)
    parser.add_argument("--oks_sigma", type=float, default=0.1)
    args = parser.parse_args()
    
    
    output_path = Path(args.pytorch_config_path).parent.parent / f"results_{datetime.now().strftime('%Y%m%d%H%M')}" 
    output_path.mkdir(exist_ok=True)
    
    for test_file in os.listdir(args.test_folder):
        test_file_path = os.path.join(args.test_folder, test_file)
        dataset_name = test_file.split("_")[0]
        output_path_subdataset = output_path / dataset_name
        print(output_path_subdataset)

        # Open the file in append mode
        results_file_path = output_path / "test_results.txt"
        with open(results_file_path, "a") as f:
            # Write a separator line and model path
            f.write(f"Model: {args.snapshot_path}\n")

        evaluate_single_dataset(
            project_root=args.project_root,
            dataset_name=dataset_name,
            train_file=test_file_path,
            test_file=test_file_path,
            pytorch_config_path=args.pytorch_config_path,
            device=args.device,
            snapshot_path=args.snapshot_path,
            detector_path=args.detector_path,
            output_folder=output_path_subdataset,
            pcutoff=args.pcutoff,
            oks_sigma=args.oks_sigma,
            gpus=args.gpus,
        )