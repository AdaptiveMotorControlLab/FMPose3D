import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import deeplabcut
import datetime
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


def zeroshot_image_Inference(input_image_path, output_image_path):
    
    filepath=input_image_path

    image_path = os.path.abspath(filepath)
    basename = os.path.basename(image_path)
    image_name = basename.split('.')[0]
    print(image_path)
    print(image_name)
    
    # @markdown SuperAnimal Configurations
    superanimal_name = "superanimal_quadruped" #@param ["superanimal_topviewmouse", "superanimal_quadruped"]
    model_name = "hrnet_w32" #@param ["hrnet_w32"]
    detector_name = "fasterrcnn_resnet50_fpn_v2" #@param ["fasterrcnn_resnet50_fpn_v2"]

    # @markdown ---
    # @markdown What is the maximum number of animals you expect to have in an image
    max_individuals = 1  # @param {type:"slider", min:1, max:30, step:1}
                
    # Note you need to enter max_individuals correctly to get the correct number of predictions in the image.
    _ = superanimal_analyze_images(
        superanimal_name,
        model_name,
        detector_name,
        image_path,
        max_individuals,
        out_folder=output_image_path,
    )
    

def zeroshot_video_inference(input_video_path, output_video_path, video_adapt=False):
    
    # choose the superanimal moodel and model name
    superanimal_name = "superanimal_quadruped" #@param ["superanimal_topviewmouse", "superanimal_quadruped"]
    model_name = "hrnet_w32" #@param ["hrnet_w32"]
    detector_name = "fasterrcnn_resnet50_fpn_v2" #@param ["fasterrcnn_resnet50_fpn_v2"]
    
    # @markdown What is the maximum number of animals you expect to have in an image
    max_individuals = 1  # @param {type:"slider", min:1, max:30, step:1}

    video_path = input_video_path

    if video_adapt:        
        _ = video_inference_superanimal(
            videos=[video_path],
            superanimal_name=superanimal_name,
            model_name=model_name,
            detector_name=detector_name,
            video_adapt=True,
            max_individuals=max_individuals,
            pseudo_threshold=0.1,
            bbox_threshold=0.9,
            detector_epochs=1,
            pose_epochs=1,
            dest_folder=output_video_path
        )
    else:
         _ = video_inference_superanimal(
            videos=video_path,
            superanimal_name=superanimal_name,
            model_name=model_name,
            detector_name=detector_name,
            video_adapt=video_adapt,
            max_individuals=max_individuals,
            dest_folder=output_video_path,
            ) 


if __name__ == "__main__":

    flag_image_inference = False
    flag_video_inference = True
    
    if flag_image_inference:
        input_image_path = "/home/ti_wang/Ti_workspace/PrimatePose/Vis/samples/original_oap_00208_oap_anoID_46.JPG"
        output_image_path = "/home/ti_wang/Ti_workspace/PrimatePose/SuperAnimalEvaluation/vis/."
        zeroshot_image_Inference(input_image_path, output_image_path)

    if flag_video_inference:
        video_adapt_label = True
        video_path = "/home/ti_wang/Ti_workspace/PrimatePose/data/demo/a_monkey_50frames/a_monkey_50frames.mp4"
        video_name = video_path.split("/")[-1].split(".")[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_folder_name = f"{video_name}_{timestamp}"
        output_video_path = "/home/ti_wang/Ti_workspace/PrimatePose/SuperAnimalEvaluation/vis/video_inference"
        output_video_path = os.path.join(output_video_path, video_folder_name)
        zeroshot_video_inference(video_path, output_video_path, video_adapt=video_adapt_label) 