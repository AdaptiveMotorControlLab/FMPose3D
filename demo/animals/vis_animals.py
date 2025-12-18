# SuperAnimal Demo: https://github.com/DeepLabCut/DeepLabCut/blob/main/examples/COLAB/COLAB_YOURDATA_SuperAnimal.ipynb

import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import deeplabcut
import deeplabcut.utils.auxiliaryfunctions as auxiliaryfunctions
from deeplabcut.pose_estimation_pytorch.apis import (
    superanimal_analyze_images,
)
from deeplabcut.modelzoo import build_weight_init
from deeplabcut.modelzoo.utils import (
    create_conversion_table,
    read_conversion_table_from_csv,
)

from deeplabcut.modelzoo.video_inference import video_inference_superanimal
from deeplabcut.utils.pseudo_label import keypoint_matching

superanimal_name = "superanimal_quadruped" #@param ["superanimal_topviewmouse", "superanimal_quadruped"]
model_name = "hrnet_w32" #@param ["hrnet_w32", "resnet_50"]
detector_name = "fasterrcnn_resnet50_fpn_v2" #@param ["fasterrcnn_resnet50_fpn_v2", "fasterrcnn_mobilenet_v3_large_fpn"]

# @markdown ---
# @markdown What is the maximum number of animals you expect to have in an image
max_individuals = 1  # @param {type:"slider", min:1, max:30, step:1}

image_path = "./images/dog.JPEG"

# Note you need to enter max_individuals correctly to get the correct number of predictions in the image.
predictions = superanimal_analyze_images(
    superanimal_name,
    model_name,
    detector_name,
    image_path,
    max_individuals,
    out_folder="./predictions/",
    # close_figure_after_save=False
)
print("predictions:", predictions)

# get the 2D keypoints from the predictions
xy_preds = {}
# predictions is a dict: {image_path: {"bodyparts": (N, K, 3), "bboxes": ..., "bbox_scores": ...}}
for img_path, payload in predictions.items():
    bodyparts = payload.get("bodyparts")
    if bodyparts is None:
        continue
    # bodyparts shape: (num_individuals, num_keypoints, 3) -> [:, :, :2] keeps x,y
    xy_preds[img_path] = bodyparts[..., :2]

print("2D keypoints (x,y) by image:")
for img_path, xy in xy_preds.items():
    print(f"{img_path}: shape {xy.shape}")