"""Evaluating COCO models"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from deeplabcut.pose_estimation_pytorch import COCOLoader
from deeplabcut.pose_estimation_pytorch.apis.evaluate import evaluate
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.pose_estimation_pytorch.apis.evaluate import visualize_predictions
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
}

# Define the skeleton structure
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

from PIL import Image

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
    
    # Create results directory in the same directory as the config file
    config_dir = Path(pytorch_config_path).parent
    output_path = config_dir / "results" / "with_tail"
    snapshot_name = Path(snapshot_path).parent.parent
    output_path = output_path / snapshot_name
    output_path.mkdir(parents=True, exist_ok=True)
    return 
    print(f"Saving results to: {output_path}")
    
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

        visualize_predictions(
            predictions=predictions,
            ground_truth=gt_keypoints,
            output_dir=output_path,
            num_samples=10,  # Added to limit visualization to 10 samples
            random_select=True,
        )
        
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