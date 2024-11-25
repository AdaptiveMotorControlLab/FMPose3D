"""Run video analysis"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from deeplabcut.pose_estimation_pytorch.apis.analyze_videos import (
    create_df_from_prediction,
    video_inference,
    VideoIterator,
)
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.task import Task
from deeplabcut.utils.make_labeled_video import _create_labeled_video

# Define the skeleton structure with indices
PFM_SKELETON_INDICES = [
    [3, 5], [4, 5], [6, 3], [7, 4],
    [5, 12], [13, 12], [14, 12], [2, 17],
    [19, 13], [20, 14], [21, 19], [22, 20],
    [23, 21], [24, 22], [25, 12], [26, 12],
    [25, 27], [26, 27], [25, 28], [26, 29],
    [27, 28], [27, 29], [28, 30], [29, 31],
    [30, 32], [31, 33], [27, 34], [34, 35],
    [35, 36], [36, 37]
]

# Define the skeleton structure with bodypart names
PFM_SKELETON = [
    ["right_eye", "nose"], ["left_eye", "nose"], ["left_ear", "right_eye"], ["right_ear", "left_eye"],
    ["nose", "neck"], ["left_shoulder", "neck"], ["right_shoulder", "neck"], ["head", "body_center"],
    ["left_elbow", "left_shoulder"], ["right_elbow", "right_shoulder"], ["left_wrist", "left_elbow"], ["right_wrist", "right_elbow"],
    ["left_hand", "left_wrist"], ["right_hand", "right_wrist"], ["left_hip", "neck"], ["right_hip", "neck"],
    ["left_hip", "center_hip"], ["right_hip", "center_hip"], ["left_hip", "left_knee"], ["right_hip", "right_knee"],
    ["center_hip", "left_knee"], ["center_hip", "right_knee"], ["left_knee", "left_ankle"], ["right_knee", "right_ankle"],
    ["left_ankle", "left_foot"], ["right_ankle", "right_foot"], ["center_hip", "root_tail"], ["root_tail", "mid_tail"],
    ["mid_tail", "mid_end_tail"], ["mid_end_tail", "end_tail"]
]
# PFM_SKELETON = [
#     # Face connections
#     ["right_eye", "nose"], ["left_eye", "nose"],
#     ["left_ear", "left_eye"], ["right_ear", "right_eye"],
    
#     # Head and neck
#     ["nose", "neck"],
#     ["head", "neck"],
    
#     # Shoulder structure
#     ["left_shoulder", "neck"], ["right_shoulder", "neck"],
    
#     # Arms
#     ["left_elbow", "left_shoulder"], ["right_elbow", "right_shoulder"],
#     ["left_wrist", "left_elbow"], ["right_wrist", "right_elbow"],
#     ["left_hand", "left_wrist"], ["right_hand", "right_wrist"],
    
#     # Torso
#     ["neck", "body_center"],
#     ["body_center", "center_hip"],
    
#     # Hips
#     ["left_hip", "center_hip"], ["right_hip", "center_hip"],
    
#     # Legs
#     ["left_hip", "left_knee"], ["right_hip", "right_knee"],
#     ["left_knee", "left_ankle"], ["right_knee", "right_ankle"],
#     ["left_ankle", "left_foot"], ["right_ankle", "right_foot"],
    
#     # Tail
#     ["center_hip", "root_tail"],
#     ["root_tail", "mid_tail"],
#     ["mid_tail", "mid_end_tail"],
#     ["mid_end_tail", "end_tail"]
# ]


def default_confidence_to_alpha(x, pcutoff=0.6):
    if pcutoff == 0:
        return x
    return np.clip((x - pcutoff) / (1 - pcutoff), 0, 1)

def main(
    video_path: str | Path,
    model_config: str,
    snapshot_path: str,
    detector_path: str | None,
    num_animals: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    video_path = Path(video_path)
    model_config_path = Path(model_config)
    model_cfg = read_config_as_dict(model_config)
    
    # Ensure detector is provided for top-down approach
    if model_cfg.get("method", "") == "top-down" and detector_path is None:
        raise ValueError("Must provide detector_path for top-down video analysis")
    
    pose_task = Task(model_cfg["method"])
    
    # Update config with device and max individuals
    model_cfg["device"] = device
    model_cfg["max_individuals"] = num_animals
    
    pose_runner, detector_runner = get_inference_runners(
        model_config=model_cfg,
        snapshot_path=snapshot_path,
        max_individuals=num_animals,
        num_bodyparts=len(model_cfg["metadata"]["bodyparts"]),
        num_unique_bodyparts=len(model_cfg["metadata"]["unique_bodyparts"]),
        with_identity=model_cfg["metadata"].get("with_identity", False),
        transform=None,
        detector_path=detector_path,
        detector_transform=None,
    )
    
    # Create video iterator and get dimensions directly
    video_iterator = VideoIterator(str(video_path))
    vid_w, vid_h = video_iterator.dimensions
    
    # Use dimensions directly for bbox
    bbox = (0, vid_w, 0, vid_h)

    # Run video inference
    predictions = video_inference(
        video=video_path,
        task=pose_task,
        pose_runner=pose_runner,
        detector_runner=detector_runner,
        cropping=None,
    )

    pred_bodyparts = np.stack([p["bodyparts"][..., :3] for p in predictions])
    pred_unique_bodyparts = None
    if len(predictions) > 0 and "unique_bodyparts" in predictions[0]:
        pred_unique_bodyparts = np.stack([p["unique_bodyparts"] for p in predictions])

    # Prepare config for output
    cfg = copy.deepcopy(model_cfg)
    cfg["individuals"] = [f"individual_{i}" for i in range(num_animals)]
    cfg["bodyparts"] = cfg["metadata"]["bodyparts"]
    cfg["uniquebodyparts"] = []
    cfg["multianimalproject"] = True
    
    # Update metadata with individuals
    model_cfg["metadata"]["individuals"] = cfg["individuals"]
    
    # Print debugging information
    print(f"Number of animals: {num_animals}")
    print(f"Number of bodyparts: {len(model_cfg['metadata']['bodyparts'])}")
    
    # Create scorer name
    dlc_scorer = ""
    if detector_path is not None:
        dlc_scorer += Path(detector_path).stem
    dlc_scorer += Path(snapshot_path).stem

    # Save results
    output_prefix = f"{video_path.stem}_{dlc_scorer}"
    output_path = video_path.parent
    output_h5 = output_path / (output_prefix + ".h5")
    
    # Print debugging information
    print(f"Predictions shape: {pred_bodyparts.shape}")
    
    # Create DataFrame with debugging information
    print("Creating DataFrame with:")
    print(f"- Number of frames: {len(pred_bodyparts)}")
    print(f"- Number of coordinates per frame: {pred_bodyparts.shape[1]}")
    print(f"- Number of individuals in config: {len(cfg['individuals'])}")
    
    _ = create_df_from_prediction(
        predictions=predictions,  # Use original predictions containing full data
        dlc_scorer=dlc_scorer,
        cfg=cfg,
        model_cfg=model_cfg,
        output_path=output_path,
        output_prefix=output_prefix,
        save_as_csv=False
    )
    
    # Create labeled video with keypoint confidences from predictions
    
    _create_labeled_video(
        str(video_path),
        str(output_h5),
        pcutoff=0.6,
        fps=video_iterator.fps,
        bbox=bbox,
        output_path=str(output_path / f"{output_prefix}_labeled.mp4"),
        skeleton_edges=PFM_SKELETON,
        confidence_to_alpha=default_confidence_to_alpha,  # Use confidence scores from predictions
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("model_config_path", help="Path to the model configuration file")
    parser.add_argument("snapshot_path", help="Path to the pose estimation model checkpoint")
    parser.add_argument("--detector_path", default=None, help="Path to the detector model checkpoint")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_animals", type=int, default=1)
    args = parser.parse_args()
    main(
        video_path=args.video_path,
        model_config=args.model_config_path,
        snapshot_path=args.snapshot_path,
        detector_path=args.detector_path,
        num_animals=args.num_animals,
        device=args.device,
    )
