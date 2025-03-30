"""Creates a DeepLabCut project from a COCO dataset"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import deeplabcut
import deeplabcut.pose_estimation_pytorch as dlc_torch
import numpy as np
import pandas as pd
from deeplabcut.generate_training_dataset import merge_annotateddatasets
from deeplabcut.utils import auxiliaryfunctions

def create_black_video(
    output_path: str,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
    num_frames: int = 10,
) -> None:
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create black frames and write to video
    for _ in range(num_frames):
        black_frame = np.zeros((height, width, 3), dtype=np.uint8)
        out.write(black_frame)

    # Release the video writer
    out.release()
    print(f"Created black video with {num_frames} frames at {output_path}")


def add_bodyparts_and_videos(
    project_dir: Path,
    bodyparts: list[str],
    individuals: list[str],
    video_sets: dict[str, dict],
) -> None:
    cfg = auxiliaryfunctions.read_config(str(project_dir / "config.yaml"))
    cfg["multianimalbodyparts"] = bodyparts
    cfg["individuals"] = individuals
    cfg["video_sets"] = video_sets
    auxiliaryfunctions.write_config(str(project_dir / "config.yaml"), cfg)


def create_shuffle_with_correct_train_val_split(project_path: Path) -> None:
    cfg = auxiliaryfunctions.read_config(str(project_path / "config.yaml"))
    trainset_dir = project_path / auxiliaryfunctions.get_training_set_folder(cfg)
    trainset_dir.mkdir(exist_ok=True, parents=True)
    df = merge_annotateddatasets(cfg, trainset_dir)

    train_indices = [idx for idx, name in enumerate(df.index) if "train" in name[1]]
    test_indices = [idx for idx, name in enumerate(df.index) if "test" in name[1]]

    deeplabcut.create_training_dataset(
        config=str(project_path / "config.yaml"),
        Shuffles=[0],
        trainIndices=[train_indices],
        testIndices=[test_indices],
    )


def main(
    output_dir: Path,
    task: str,
    experimenter: str,
    coco_project: Path,
    train_file: Path,
    test_file: Path,
):
    output_dir = output_dir.resolve()

    # DeepLabCut needs a video for a project to be created; create a tiny video
    create_black_video(output_path=str(output_dir / "video.mp4"))

    # Create the project
    config = deeplabcut.create_new_project(
        project=task,
        experimenter=experimenter,
        videos=[str(output_dir / "video.mp4")],
        working_directory=str(output_dir),
        multianimal=True,
        individuals=["animal"],
    )
    config = Path(config)

    # Check that the DeepLabCut project exists
    project_path = config.parent

    # Create the labeled data directory
    labeled_data_dir = project_path / "labeled-data"
    labeled_data_dir.mkdir(exist_ok=True)

    bodyparts = None
    max_annotations = 0
    id_to_annotations = {}

    # Parse the annotations and get the max number of annotations for an image
    for split, filepath in [("train", train_file), ("test", test_file)]:
        # Load the COCO data
        with open(filepath, "r") as f:
            coco_data = json.load(f)

        # Create a "labeled-data" folder for the images in this file
        img_dir = labeled_data_dir / split
        img_dir.mkdir(exist_ok=True, parents=True)

        # Get the bodyparts
        if bodyparts is None:
            bodyparts = coco_data["categories"][0]["keypoints"]
        assert bodyparts == coco_data["categories"][0]["keypoints"]

        for img in coco_data["images"]:
            img_annotations = [
                np.asarray(a["keypoints"]).astype(float).reshape((-1, 3))
                for a in coco_data["annotations"]
                if (
                    a["image_id"] == img["id"]
                    and a["iscrowd"] == 0
                    and a.get("num_keypoints") > 1
                )
            ]

            # Keep only annotations with at least 2 keypoints
            img_annotations = [
                anno for anno in img_annotations if np.sum(anno[:, 2]) >= 2
            ]
            id_to_annotations[f"{split}_{img['id']}"] = img_annotations
            max_annotations = max(max_annotations, len(img_annotations))

    for split, filepath in [("train", train_file), ("test", test_file)]:
        with open(filepath, "r") as f:
            coco_data = json.load(f)
        img_dir = labeled_data_dir / split

        images = []
        data = []
        for img in coco_data["images"]:
            filename = Path(img["file_name"])

            # Get the location of the ground truth file
            source_filepath = filename
            if not source_filepath.is_absolute():
                source_filepath = coco_project / "images" / filename
            assert source_filepath.exists(), f"Cannot find image {source_filepath}"

            # Get the output filename, copy the image over
            # dest_filepath = img_dir / "labeled-data" / split / filename.name
            dest_filepath = img_dir / filename.name
            shutil.copy2(source_filepath, dest_filepath)

            # Parse the annotations for the image and add to the data that will go into the dataframe
            img_data = np.full((max_annotations, len(bodyparts), 2), np.nan)
            img_annotations = id_to_annotations[f"{split}_{img['id']}"]
            for i, pose in enumerate(img_annotations):
                pose[pose[:, 2] <= 0] = np.nan
                img_data[i] = pose[:, :2]

            images.append(filename.name)
            data.append(img_data.reshape(-1))

        df = pd.DataFrame(
            data,
            index=pd.MultiIndex.from_tuples(
                [("labeled-data", split, filename) for filename in images]
            ),
            columns=pd.MultiIndex.from_product(
                [
                    [experimenter],
                    [f"idv{idv}" for idv in range(max_annotations)],
                    bodyparts,
                    ["x", "y"],
                ],
                names=["scorer", "individuals", "bodyparts", "coords"],
            ),
        )
        filepath = img_dir / f"CollectedData_{experimenter}.h5"
        df.to_hdf(filepath, key="df_with_missing")
        df.to_csv(filepath.with_suffix(".csv"))

    # this is fine
    video_sets = {
        str(project_path / "videos" / f"{v}.mp4"): {"crop": "0, 2048, 0, 1536"}
        for v in ["train", "test"]
    }
    add_bodyparts_and_videos(
        project_path,
        bodyparts=bodyparts,
        individuals=[f"idv{i}" for i in range(max_annotations)],
        video_sets=video_sets,
    )
    
    # Create a train/val shuffle
    create_shuffle_with_correct_train_val_split(project_path)

    # [Optional] generate keypoint visualization images
    print("Generating keypoint visualization images...")
    # deeplabcut.check_labels(str(project_path / "config.yaml"), draw_skeleton=True)
    print("Keypoint visualization images generated!")

if __name__ == "__main__":
    
    output_dir = Path("/home/ti_wang/Ti_workspace/PrimatePose/SuperAnimalEvaluation/test_dlc").resolve()
    main(
        output_dir=output_dir,
        task="pfm",
        experimenter="dlc",
        coco_project=Path("/home/ti_wang/data/tiwang/v8_coco"),
        train_file=Path("/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/splitted_train_datasets/deepwild_train_train.json"),
        test_file=Path("/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/splitted_test_datasets/deepwild_train_test.json"),
    )