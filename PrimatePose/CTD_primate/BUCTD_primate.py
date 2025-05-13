import requests
import shutil
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import os
import argparse
from deeplabcut.generate_training_dataset import merge_annotateddatasets
import deeplabcut
import deeplabcut.pose_estimation_pytorch as dlc_torch
import deeplabcut.utils.auxiliaryfunctions as auxiliaryfunctions
import matplotlib.pyplot as plt
import numpy as np
import yaml

def create_training_dataset_pfm(config, net_type="resnet_50", BU_SHUFFLE=1):
    cfg = auxiliaryfunctions.read_config(config)
    
    # Create the training dataset folder
    project_path = Path(config).parent
    trainset_dir = project_path / auxiliaryfunctions.get_training_set_folder(cfg)
    trainset_dir.mkdir(exist_ok=True, parents=True)
    
    # Merge annotation datasets and get dataframe
    df = merge_annotateddatasets(cfg, trainset_dir)
    
    # Identify train and test indices based on the dataset folders
    train_indices = [idx for idx, name in enumerate(df.index) if "train" in name[1]]
    test_indices = [idx for idx, name in enumerate(df.index) if "test" in name[1]]
    
    print(f"Using {len(train_indices)} images for training and {len(test_indices)} for testing")
    
    # Create the training dataset with the identified indices
    deeplabcut.create_training_dataset(
        config,
        Shuffles=[BU_SHUFFLE],
        trainIndices=[train_indices],
        testIndices=[test_indices],
        net_type=net_type,
        engine=deeplabcut.Engine.PYTORCH,
        userfeedback=False,
    )
    return config

def backup_files(config, shuffle=1):
        # back up files
        train_scripts = "train_CTD.sh" 
        # Dynamically determine the model directory
        cfg = auxiliaryfunctions.read_config(config)
        task = cfg['Task']
        date= cfg['date']
        prefix = task + date
        TrainingFraction = cfg['TrainingFraction'][0] * 100
        model_folder = f"{prefix}-trainset{int(TrainingFraction)}shuffle{shuffle}"
        # Build complete backup directory path
        backup_dir = os.path.join(Path(config).parent, "dlc-models-pytorch", f"iteration-0", model_folder)
        # Create backup directory if it doesn't exist
        os.makedirs(backup_dir, exist_ok=True)
        # Copy files to backup directory
        shutil.copy2(train_scripts, os.path.join(backup_dir, train_scripts))
        current_script = os.path.basename(__file__)
        shutil.copy2(current_script, os.path.join(backup_dir, current_script))
        print(f"Backed up {train_scripts} and {current_script} to {backup_dir}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepLabCut training script for primate pose estimation')
    
    # Configuration
    parser.add_argument('--config', type=str, 
                        default="/TD_primate/pfm_oms_small-dlc-2025-05-02/config.yaml",
                        help='Path to the config.yaml file')
    
    # Shuffle indices
    parser.add_argument('--bu_shuffle', type=int, default=1,
                        help='Shuffle index for baseline model (default: 1)')
    parser.add_argument('--ctd_shuffle', type=int, default=7,
                        help='Shuffle index for CTD model (default: 7)')
    
    # Training parameters
    parser.add_argument('--CTD_epochs', type=int, default=400,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--bu_net_type', type=str, default="resnet_50",
                        help='Network type for baseline')
    parser.add_argument('--ctd_net_type', type=str, default="ctd_prenet_rtmpose_m",
                        help='Network type for CTD (default: ctd_prenet_rtmpose_m)')
    
    # Actions
    parser.add_argument('--create_BU_dataset', type=bool, default=False,
                        help='Create training dataset using create_training_dataset_pfm')
    parser.add_argument('--train_BU', type=bool, default=False,
                        help='Train the baseline network')
    parser.add_argument('--evaluate_BU', type=bool, default=False,
                        help='Evaluate the baseline network')
    parser.add_argument('--create_CTD_dataset', type=bool, default=False,
                        help='Create CTD training dataset')
    parser.add_argument('--train_CTD', type=bool, default=False,
                        help='Train the CTD network')
    parser.add_argument('--evaluate_CTD', type=bool, default=False,
                        help='Evaluate the CTD network')
    
    args = parser.parse_args()
    
    print("training!")
    
    config = args.config
    BU_SHUFFLE = args.bu_shuffle
    CTD_SHUFFLE = args.ctd_shuffle
    
    # Execute actions based on arguments
    if args.create_BU_dataset:
        config = create_training_dataset_pfm(config,net_type=args.bu_net_type, BU_SHUFFLE=BU_SHUFFLE)
        print("config:", config)
    
    if args.train_BU:
        print("Training BU network")
        backup_files(config, shuffle=BU_SHUFFLE)
        print("BU_shuffle:", BU_SHUFFLE)
        deeplabcut.train_network(
            config,
            shuffle=BU_SHUFFLE,
            batch_size=args.batch_size,
        )
    
    if args.evaluate_BU:
        print("Evaluating BU network")
        deeplabcut.evaluate_network(config, Shuffles=[BU_SHUFFLE])
    
    if args.create_CTD_dataset:
        print("Creating CTD training dataset")
        deeplabcut.create_training_dataset_from_existing_split(
            config,
            from_shuffle=BU_SHUFFLE,
            shuffles=[CTD_SHUFFLE],
            net_type=args.ctd_net_type,
            engine=deeplabcut.Engine.PYTORCH,
            ctd_conditions=(BU_SHUFFLE, -1),
        )
            
    if args.train_CTD:
        print("Training CTD network")
        backup_files(config, shuffle=CTD_SHUFFLE)
        deeplabcut.train_network(
            config,
            shuffle=CTD_SHUFFLE,
            epochs=args.CTD_epochs,
            batch_size=args.batch_size,
        )
        
    if args.evaluate_CTD:
        print("Evaluating CTD network")
        print(CTD_SHUFFLE)
        deeplabcut.evaluate_network(config, Shuffles=[CTD_SHUFFLE])