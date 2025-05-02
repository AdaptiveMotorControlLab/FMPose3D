import requests
import shutil
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import os
from deeplabcut.generate_training_dataset import merge_annotateddatasets
import deeplabcut
import deeplabcut.pose_estimation_pytorch as dlc_torch
import deeplabcut.utils.auxiliaryfunctions as auxiliaryfunctions
import matplotlib.pyplot as plt
import numpy as np


def create_training_dataset_pfm(config, BU_SHUFFLE=1):
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
        net_type="resnet_50",
        engine=deeplabcut.Engine.PYTORCH,
        userfeedback=False,
    )
    return config, train_indices, test_indices

config = "/home/ti_wang/Ti_workspace/PrimatePose/CTD_primate/pfm_deepwild-dlc-2025-05-02/config.yaml"

create_training_dataset_pfm(config)

BU_SHUFFLE = 0

deeplabcut.train_network(
    config,
    shuffle=BU_SHUFFLE,
    epochs=100,
)

# deeplabcut.evaluate_network(config, Shuffles=[BU_SHUFFLE])    