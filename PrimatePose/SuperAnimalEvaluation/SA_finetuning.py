import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import deeplabcut
import datetime
import deeplabcut.utils.auxiliaryfunctions as auxiliaryfunctions
import argparse
from deeplabcut.pose_estimation_pytorch.apis import (
    superanimal_analyze_images,
)
from deeplabcut.modelzoo import build_weight_init
from deeplabcut.modelzoo.utils import (
    create_conversion_table,
    read_conversion_table_from_csv,
)

def SA_training_dlc_project(config_path, shuffle_label=1, pose_epochs=50, pose_batch_size=64, SA_weight_init=None, create_train_dataset_from_existing_split=False, from_shuffle_label=None):
        
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
    
    parser = argparse.ArgumentParser(description='DeepLabCut training script for primate pose estimation')
    
    # Configuration
    parser.add_argument('--config_path', type=str, 
                        default="/TD_primate/pfm_oms_small-dlc-2025-05-02/config.yaml",
                        help='Path to the config.yaml file')
    
    # Shuffle indices
    parser.add_argument('--shuffle_label', type=int, default=1,
                        help='Shuffle index for baseline model (default: 1)')
    # Training parameters
    parser.add_argument('--pose_epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--pose_batch_size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--SA_weight_init', type=bool, default=False,
                        help='Use SuperAnimal weight initialization (default: False)')
    parser.add_argument('--create_train_dataset', type=bool, default=False,
                        help='Create training dataset (default: False)')
    parser.add_argument('--create_train_dataset_from_existing_split', type=bool, default=False,
                        help='Create training dataset from existing split (default: False)')
    parser.add_argument('--from_shuffle_label', type=int, default=0,
                        help='Shuffle index for training dataset (default: 0)')
    parser.add_argument('--train_network', type=bool, default=False,
                        help='Train network (default: False)')
    parser.add_argument('--evaluate_network', type=bool, default=False,
                        help='Evaluate network (default: False)')
    parser.add_argument('--model_name', type=str, default="hrnet_w32",
                        help='Model name (default: hrnet_w32)')
    parser.add_argument('--detector_name', type=str, default="fasterrcnn_resnet50_fpn_v2",
                        help='Detector name (default: fasterrcnn_resnet50_fpn_v2)')
    parser.add_argument('--superanimal_name', type=str, default="superanimal_quadruped",
                        help='SuperAnimal name (default: superanimal_quadruped)')
    # project_path = Path("/home/ti_wang/Ti_workspace/PrimatePose/SuperAnimalEvaluation/test_dlc/pfm-dlc-2025-03-28")
 
    args = parser.parse_args()
    imagenet_transfer_learning_shuffle = 0
    superanimal_transfer_learning_shuffle = 1
    superanimal_naive_finetune_shuffle = 2
    superanimal_memory_replay_shuffle = 3

    # Configure which approach to use
    create_train_dataset_from_existing_split_label = True  # Set to True to create dataset from existing split
    SA_weight_init_label = False  # Set to True to use SuperAnimal weight initialization

    # SA_training_dlc_project(
    #     config_path=args.config, 
    #     shuffle_label=args.shuffle_label, 
    #     pose_epochs=args.pose_epochs, 
    #     pose_batch_size=args.pose_batch_size,
    #     SA_weight_init=args.SA_weight_init,
    #     create_train_dataset_from_existing_split=create_train_dataset_from_existing_split_label,
    #     from_shuffle_label=imagenet_transfer_learning_shuffle
    # )
    
    # @markdown SuperAnimal configuration parameters
    superanimal_name = "superanimal_quadruped" #@param ["superanimal_topviewmouse", "superanimal_quadruped"]
    model_name = "hrnet_w32" #@param ["hrnet_w32"]
    detector_name = "fasterrcnn_resnet50_fpn_v2" #@param ["fasterrcnn_resnet50_fpn_v2"]

    if args.SA_weight_init:
        weight_init = build_weight_init(
            cfg=auxiliaryfunctions.read_config(args.config_path), 
            super_animal=superanimal_name,
            model_name=model_name,
            detector_name=detector_name,
            with_decoder=False,
        )
        print("Using SuperAnimal weight initialization")
    else:
        weight_init = None
        print("Not using SuperAnimal weight initialization")
            
    if args.create_train_dataset:
        print("Creating training dataset")
        deeplabcut.create_training_dataset(
            args.config_path,
            Shuffles=[args.shuffle_label],
            net_type=f"top_down_{model_name}",
            detector_type=detector_name,
            engine=deeplabcut.Engine.PYTORCH,
            weight_init=weight_init,
            userfeedback=False
        )     
        
    if args.create_train_dataset_from_existing_split:
        deeplabcut.create_training_dataset_from_existing_split(
            args.config_path,
            from_shuffle=args.from_shuffle_label,
            shuffles=[args.shuffle_label],
            engine=deeplabcut.Engine.PYTORCH,
            net_type=f"top_down_{model_name}",
            detector_type=detector_name,
            weight_init=weight_init,
            userfeedback=False,
        )
    
    if args.train_network:
        print("Training network")
        # deeplabcut.train_network(
        #     args.config_path,
        #     detector_epochs=0,
        #     epochs=args.pose_epochs,
        #     save_epochs=10,
        #     batch_size=args.pose_batch_size,
        #     displayiters=1000,
        #     shuffle= args.shuffle_label,
        # )    
    
    if args.evaluate_network:
        print("Evaluating network")
        # deeplabcut.evaluate_network(args.config_path, Shuffles=[args.shuffle_label])