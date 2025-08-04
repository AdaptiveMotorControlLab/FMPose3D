"""Creates a base model configuration file to train a model on a COCO dataset

"""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
import deeplabcut.utils.auxiliaryfunctions as af
from deeplabcut.generate_training_dataset import MakeInference_yaml
from deeplabcut.pose_estimation_pytorch.config import make_pytorch_pose_config
from deeplabcut.pose_estimation_pytorch.data import COCOLoader
import logging
from cal_the_bad_samples import cal_different_kinds_of_samples

def get_base_config(
    project_path: str,
    pose_config_path: str,
    model_architecture: str,
    detector_architecture: str,
    bodyparts: list[str],
    unique_bodyparts: list[str],
    individuals: list[str],
    multi_animal: bool,
) -> dict:
    cfg = {
        "project_path": project_path,
        "multianimalproject": multi_animal,
        "bodyparts": bodyparts,
        "multianimalbodyparts": bodyparts,
        "uniquebodyparts": unique_bodyparts,
        "individuals": individuals,
    }

    top_down = False
    if model_architecture.startswith("top_down_"):
        top_down = True
        model_architecture = model_architecture[len("top_down_") :]

    return make_pytorch_pose_config(
        project_config=cfg,
        pose_config_path=pose_config_path,
        net_type=model_architecture,
        top_down=top_down,
        detector_type=detector_architecture
    )

def make_inference_config(
    dlc_path: str,
    output_path: str,
    bodyparts: list[str],
    num_individuals: int,
):
    default_config_path = Path(dlc_path) / "inference_cfg.yaml"
    items2change = {
        "minimalnumberofconnections": int(len(bodyparts) / 2),
        "topktoretain": num_individuals,
        "withid": False,  # TODO: implement
    }
    MakeInference_yaml(items2change, output_path, default_config_path)


def str_to_bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(
    project_root: str,
    train_file: str,
    test_file: str,
    output: str,
    model_arch: str,
    detector_arch: str,
    multi_animal: bool,
    debug: bool,
    dino_pretrained: bool = True,
):
    
    output_path = Path(output)
    project_name = output_path.name
    
    if debug==True:    
        # output_path = Path(output)
        parent_path = output_path.parent
        output_path = parent_path / "Debug" / project_name
        # output_path = Path(output)
          
    if output_path.exists():
        raise RuntimeError(
            f"The output path must not exist yet, as otherwise we would risk overwriting"
            f" existing configurations ({output_path} exists)"
        )

    train_dict = COCOLoader.load_json(project_root, train_file)
    num_individuals, bodyparts = COCOLoader.get_project_parameters(train_dict)
    dlc_path = af.get_deeplabcut_path()
  
    output_path.mkdir(parents=True)
    train_dir = output_path / "train"
    test_dir = output_path / "test"
    train_dir.mkdir()
    test_dir.mkdir()

    pose_config_path = str(train_dir / "pytorch_config.yaml")
    pytorch_cfg = get_base_config(
        project_path=project_root,
        pose_config_path=pose_config_path,
        model_architecture=model_arch,
        detector_architecture=detector_arch,
        bodyparts=bodyparts,
        unique_bodyparts=[],
        individuals= [f"individual{i}" for i in range(num_individuals)],
        multi_animal=multi_animal,
    )
    # Configure backbone-specific settings
    if pytorch_cfg["model"]["backbone"]["type"] == "HRNet":
        pytorch_cfg["model"]["backbone"]["freeze_bn_stats"] = False
    elif "vit" in model_arch.lower() or "dino" in model_arch.lower():
        print("Configuring ViT backbone with DINO pretraining...")
        
        # Parse model architecture to extract patch size and model variant
        # Examples: "top_down_vit_base_patch16_224", "vit_base_patch8_224"
        model_name = model_arch
        if model_arch.startswith("top_down_"):
            model_name = model_arch[len("top_down_"):]
        
        # Extract patch size from model name
        if "patch8" in model_name:
            patch_size = 8
            vit_model_name = "vit_base_patch8_224"
        elif "patch16" in model_name:
            patch_size = 16
            vit_model_name = "vit_base_patch16_224"
        else:
            # Default to patch16 if not specified
            patch_size = 16
            vit_model_name = "vit_base_patch16_224"
            print(f" Patch size not specified in {model_arch}, defaulting to patch16")
        
        print(f"🔧 Using ViT model: {vit_model_name} (patch_size={patch_size}, DINO={dino_pretrained})")
        
        # Override backbone configuration for ViT + DINO
        pytorch_cfg["model"]["backbone"] = {
            "type": "ViT",
            "model_name": vit_model_name,
            "img_size": 224,
            "pretrained": not dino_pretrained,  # Use ImageNet weights if DINO is disabled
            "dino_pretrained": dino_pretrained,
            "dino_arch": "vit_base",
            "patch_size": patch_size,
            "drop_rate": 0.0,
            "drop_path_rate": 0.1,
            "freeze_bn_stats": False,
            "freeze_bn_weights": False,
        }
        # Update backbone output channels
        pytorch_cfg["model"]["backbone_output_channels"] = 768
        
        # Update head configuration to match ViT output channels (768)
        for head_name, head_config in pytorch_cfg["model"]["heads"].items():
            if "heatmap_config" in head_config:
                head_config["heatmap_config"]["channels"] = [768]
            if "locref_config" in head_config:
                head_config["locref_config"]["channels"] = [768]
        
        # Update data configuration for 224x224 images (DINO standard)
        pytorch_cfg["data"]["train"]["top_down_crop"]["width"] = 224
        pytorch_cfg["data"]["train"]["top_down_crop"]["height"] = 224
        pytorch_cfg["data"]["inference"]["top_down_crop"]["width"] = 224
        pytorch_cfg["data"]["inference"]["top_down_crop"]["height"] = 224
        
        # Update net_type to reflect the actual ViT variant used
        pytorch_cfg["net_type"] = model_name
        
        print(f"✅ ViT + DINO configuration applied successfully (net_type: {model_name})")
    else:
        print(f"Using default configuration for model: {pytorch_cfg['model']['backbone']['type']}")
    
    # pytorch_cfg["detector"]["train_settings"]["epochs"] = False
    
    # Set save_epochs=1 and eval_interval=1 for both detector and runner
    pytorch_cfg["detector"]["runner"]["snapshots"]["save_epochs"] = 1
    pytorch_cfg["detector"]["runner"]["eval_interval"] = 1
    pytorch_cfg["detector"]["train_settings"]["dataloader_pin_memory"] = False
    pytorch_cfg["runner"]["snapshots"]["save_epochs"] = 1
    pytorch_cfg["runner"]["snapshots"]["max_snapshots"] = 5
    
    pytorch_cfg["runner"]["eval_interval"] = 1

    af.write_plainconfig(str(train_dir / "pytorch_config.yaml"), pytorch_cfg)
    make_inference_config(
        dlc_path,
        str(test_dir / "inference_cfg.yaml"),
        bodyparts,
        num_individuals,
    )
    print(f"Saved your model configuration in {output_path}")

    txt_path = str(train_dir / "sapmle.txt")
    cal_different_kinds_of_samples(train_file, txt_path)
    cal_different_kinds_of_samples(test_file, txt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root")
    parser.add_argument("output")
    parser.add_argument("--model_arch", default="top_down_vit_base_patch16_224")
    parser.add_argument("--detector_arch", default="fasterrcnn_mobilenet_v3_large_fpn")
    parser.add_argument("--dino_pretrained", type=str_to_bool, default=True, 
                        help="Use DINO pretraining (True/False)")
    parser.add_argument("--train_file", default="train.json")
    parser.add_argument("--test_file", default="test.json")
    parser.add_argument("--multi_animal", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()

    main(
        args.project_root,
        args.train_file,
        args.test_file,
        args.output,
        args.model_arch,
        args.detector_arch,
        args.multi_animal,
        args.debug,
        args.dino_pretrained
    )