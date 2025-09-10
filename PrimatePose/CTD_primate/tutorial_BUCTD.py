import requests
import shutil
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import os

import deeplabcut
import deeplabcut.pose_estimation_pytorch as dlc_torch
import deeplabcut.utils.auxiliaryfunctions as auxiliaryfunctions
import matplotlib.pyplot as plt
import numpy as np

def prepare_dataset():
    download_path = Path.cwd()
    config = str(download_path / "trimice-dlc-2021-06-22" / "config.yaml")

    print(f"Downloading the tri-mouse dataset into {download_path}")
    url_record = "https://zenodo.org/api/records/5851157"
    response = requests.get(url_record)
    if response.status_code == 200:
        file = response.json()["files"][0]
        title = file["key"]
        print(f"Downloading {title}...")
        with requests.get(file['links']['self'], stream=True) as r:
            with ZipFile(BytesIO(r.content)) as zf:
                zf.extractall(path=download_path)
    else:
        raise ValueError(f"The URL {url_record} could not be reached.")

    # Check that the config was downloaded correctly
    print(f"Config path: {config}")
    if not Path(config).exists():
        print(f"Could not find config at {config}: check that the dataset was downloaded correctly!")

    return config

def create_training_dataset(config, BU_SHUFFLE=1):
    cfg = auxiliaryfunctions.read_config(config)
    train_frac = cfg["TrainingFraction"][0]
    print(f"Using {int(100 * train_frac)}% of the data in the training set.")

    num_images = 112
    train_images = int(train_frac * num_images)

    seed = 0
    # rng is a random number generator instance from NumPy's random module, initialized with a specific seed for reproducibility.
    rng = np.random.default_rng(seed)

    train_indices = rng.choice(num_images, size=train_images, replace=False, shuffle=False).tolist()
    test_indices = [idx for idx in range(num_images) if idx not in train_indices]


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


def plot_generative_sampling(dataset: dlc_torch.PoseDataset, save_dir: str) -> None:
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Sample the same image 3 times and plot the results
    for i in range(3):
        item = dataset[0]

        # Remove ImageNet normalization from the image so it displays well
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = item["image"].transpose((1, 2, 0))
        img = np.clip(img * std + mean, 0, 1)

        # Get the ground trouth and "conditional pose"
        gt_pose = item["annotations"]["keypoints"][0]
        gen_samples = item["context"]["cond_keypoints"][0]

        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        for ax in axs:
            ax.imshow(img)
            ax.axis("off")

        # plot the ground truth on the left and conditions on the right
        for ax, title, keypoints in zip(
            axs,
            ["Ground Truth Pose", "Pose Conditions"],
            [gt_pose, gen_samples],
        ):
            ax.set_title(title)
            for x, y, vis in keypoints:
                if vis > 0:
                    ax.scatter([x], [y])
        
        # Save the figure to the specified directory
        save_path = os.path.join(save_dir, f"sample_{i}.png")
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)  # Close the figure to free memory
        
        print(f"Saved sample to {save_path}")

if __name__ == "__main__":
    
    config = prepare_dataset()
    
    create_training_dataset(config)

    BU_SHUFFLE = 1

    # deeplabcut.train_network(
    #     config,
    #     shuffle=BU_SHUFFLE,
    #     epochs=100,
    # )

    # deeplabcut.evaluate_network(config, Shuffles=[BU_SHUFFLE])
    
    CTD_SHUFFLE = 2

    # deeplabcut.create_training_dataset_from_existing_split(
    #     config,
    #     from_shuffle=BU_SHUFFLE,
    #     shuffles=[CTD_SHUFFLE],
    #     net_type="ctd_prenet_cspnext_m",
    #     engine=deeplabcut.Engine.PYTORCH,
    #     ctd_conditions=(BU_SHUFFLE, -1),
    # )
    
    ctd_loader = dlc_torch.DLCLoader(config, shuffle=CTD_SHUFFLE)
    
    
    
    # We'll edit the model config here directly; In practice, edit the pytorch_config file instead.
    # The parameters that can be set here are the parameters of the `dlc_torch.GenSamplingConfig`
    ctd_loader.model_cfg["data"]["gen_sampling"] = {
        "jitter_prob": 0.1,
        "swap_prob": 0.1,
        "inv_prob": 0.1,
        "miss_prob": 0.1,
    }
    
    # print("data_train_config:", ctd_loader.model_cfg["data"]["train"])
    transform = dlc_torch.build_transforms(ctd_loader.model_cfg["data"]["train"])
    dataset = ctd_loader.create_dataset(transform, mode="train", task=ctd_loader.pose_task)

    # Fix the seeds for reproducibility; you can change the seed from `0` to another value
    # to change the results
    dlc_torch.fix_seeds(0)
    image_save_dir = "/home/ti_wang/Ti_workspace/PrimatePose/CTD_primate/trimice-dlc-2021-06-22/sampled_images"
    # plot_generative_sampling(dataset, image_save_dir)
    
    # train the CTD model
    # deeplabcut.train_network(config, shuffle=CTD_SHUFFLE)
    
    download_path = Path.cwd()
    video_name = "videocompressed1.mp4"
    video_path = str(download_path / "videos" /video_name)

    deeplabcut.analyze_videos(
        config,
        [video_path],
        shuffle=CTD_SHUFFLE,
        ctd_tracking=True,
    )
    deeplabcut.create_labeled_video(
        config,
        [video_path],
        shuffle=CTD_SHUFFLE,
        track_method="ctd",
        color_by="individual",
    )
