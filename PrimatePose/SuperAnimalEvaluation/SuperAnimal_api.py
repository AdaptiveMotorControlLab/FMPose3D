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
    # unsupervised video inference   
    
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

def ImageNet_transfer_learning_dlc_project(config_path, shuffle_label=1, pose_epochs=50, pose_batch_size=64):
        
    # @markdown SuperAnimal configuration parameters
    model_name = "hrnet_w32" #@param ["hrnet_w32"]
    detector_name = "fasterrcnn_resnet50_fpn_v2" #@param ["fasterrcnn_resnet50_fpn_v2"]
                
    deeplabcut.create_training_dataset(
        config_path,
        Shuffles=[shuffle_label],
        net_type=f"top_down_{model_name}",
        detector_type=detector_name,
        engine=deeplabcut.Engine.PYTORCH,
        userfeedback=False
    )

    deeplabcut.train_network(
        config_path,
        detector_epochs=0,
        epochs=pose_epochs,
        save_epochs=10,
        batch_size=pose_batch_size,  # if you get a CUDA OOM error when training on a GPU, reduce to 32, 16, ...!
        displayiters=10,
        shuffle=shuffle_label,
    )

    deeplabcut.evaluate_network(config_path, Shuffles=[shuffle_label])

def SA_training_dlc_project(project_path, config_path, shuffle_label=1, pose_epochs=50, pose_batch_size=64, SA_weight_init=None, create_train_dataset_from_existing_split=False, from_shuffle_label=None):
        
    # @markdown SuperAnimal configuration parameters
    superanimal_name = "superanimal_quadruped" #@param ["superanimal_topviewmouse", "superanimal_quadruped"]
    model_name = "hrnet_w32" #@param ["hrnet_w32"]
    detector_name = "fasterrcnn_resnet50_fpn_v2" #@param ["fasterrcnn_resnet50_fpn_v2"]
    
    if create_train_dataset_from_existing_split:
        if SA_weight_init:
            weight_init = build_weight_init(
                cfg=auxiliaryfunctions.read_config(config_path), 
                super_animal=superanimal_name,
                model_name=model_name,
                detector_name=detector_name,
                with_decoder=False,
            )
            print("Using SuperAnimal weight initialization")
        else:
            weight_init = None
            print("Not using SuperAnimal weight initialization")
            
        deeplabcut.create_training_dataset_from_existing_split(
            config_path,
            from_shuffle=from_shuffle_label,
            shuffles=[shuffle_label],
            engine=deeplabcut.Engine.PYTORCH,
            net_type=f"top_down_{model_name}",
            detector_type=detector_name,
            weight_init=weight_init,
            userfeedback=False,
        )
    else:
        deeplabcut.create_training_dataset(
            config_path,
            Shuffles=[shuffle_label],
            net_type=f"top_down_{model_name}",
            detector_type=detector_name,
            engine=deeplabcut.Engine.PYTORCH,
            userfeedback=False
        )

    deeplabcut.train_network(
        config_path,
        detector_epochs=0,
        epochs=pose_epochs,
        save_epochs=10,
        batch_size=pose_batch_size,  # if you get a CUDA OOM error when training on a GPU, reduce to 32, 16, ...!
        displayiters=10,
        shuffle=shuffle_label,
    )

    deeplabcut.evaluate_network(config_path, Shuffles=[shuffle_label])



if __name__ == "__main__":

    flag_image_inference = False
    if flag_image_inference:
        input_image_path = "/home/ti_wang/Ti_workspace/PrimatePose/Vis/samples/original_oap_00208_oap_anoID_46.JPG"
        output_image_path = "/home/ti_wang/Ti_workspace/PrimatePose/SuperAnimalEvaluation/vis/."
        zeroshot_image_Inference(input_image_path, output_image_path)

    flag_video_inference = True
    flag_video_inference = False
    if flag_video_inference:
        video_adapt_label = True
        video_path = "/home/ti_wang/Ti_workspace/PrimatePose/data/demo/a_monkey_50frames/a_monkey_50frames.mp4"
        video_name = video_path.split("/")[-1].split(".")[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_folder_name = f"{video_name}_{timestamp}"
        output_video_path = "/home/ti_wang/Ti_workspace/PrimatePose/SuperAnimalEvaluation/vis/video_inference"
        output_video_path = os.path.join(output_video_path, video_folder_name)
        zeroshot_video_inference(video_path, output_video_path, video_adapt=video_adapt_label) 
    
    flag_training_dlc_project = True
    # flag_training_dlc_project = False
    if flag_training_dlc_project:
        project_path = Path("/home/ti_wang/Ti_workspace/PrimatePose/SuperAnimalEvaluation/test_dlc/pfm-dlc-2025-03-30")
        project_path = Path("/home/ti_wang/Ti_workspace/PrimatePose/SuperAnimalEvaluation/test_dlc/pfm-dlc-2025-03-28")
        config_path = str(project_path / "config.yaml") 
        
        """
        Definition of data split: the unique combination of training images and testing images.
        We create a data split named split 0. All baselines will share the data split to make fair comparisons.
        - split 0 -> shared by all baselines
        - shuffle 0 (split0) -> imagenet transfer learning
        - shuffle 1 (split0) -> superanimal transfer learning
        - shuffle 2 (split0) -> superanimal naive fine-tuning
        - shuffle 3 (split0) -> superanimal memory-replay fine-tuning
        """
        
        imagenet_transfer_learning_shuffle = 0
        superanimal_transfer_learning_shuffle = 1
        superanimal_naive_finetune_shuffle = 2
        superanimal_memory_replay_shuffle = 3

        # Configure which approach to use
        create_train_dataset_from_existing_split_label = True  # Set to True to create dataset from existing split
        SA_weight_init_label = False  # Set to True to use SuperAnimal weight initialization
        
        SA_training_dlc_project(
            project_path=project_path, 
            config_path=config_path, 
            shuffle_label=imagenet_transfer_learning_shuffle, 
            pose_epochs=50, 
            pose_batch_size=64,
            SA_weight_init=SA_weight_init_label,
            create_train_dataset_from_existing_split=create_train_dataset_from_existing_split_label,
            from_shuffle_label=imagenet_transfer_learning_shuffle
        )