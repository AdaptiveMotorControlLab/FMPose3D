"""File to train a model on a COCO dataset"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from deeplabcut.pose_estimation_pytorch import COCOLoader, utils
from deeplabcut.pose_estimation_pytorch.apis.training import train
from deeplabcut.pose_estimation_pytorch.runners.logger import setup_file_logging
from deeplabcut.pose_estimation_pytorch.task import Task
from collections import defaultdict
import wandb

def main(
    project_root: str,
    train_file: str,
    test_file: str,
    model_config_path: str,
    device: str | None,
    gpus: list[int] | None,
    epochs: int | None,
    save_epochs: int | None,
    detector_epochs: int | None,
    detector_save_epochs: int | None,
    snapshot_path: str | None,
    detector_path: str | None,
    batch_size: int = 32,
    dataloader_workers: int = 16,
    detector_batch_size: int = 32,
    detector_dataloader_workers: int= 16,
    debug: bool = False,
):
    log_path = Path(model_config_path).parent / "log.txt"
    setup_file_logging(log_path)

    loader = COCOLoader(
        project_root=project_root,
        model_config_path=model_config_path,
        train_json_filename=train_file,
        test_json_filename=test_file,
    )
    print(loader)
    
    utils.fix_seeds(loader.model_cfg["train_settings"]["seed"])

    if epochs is None:
        epochs = loader.model_cfg["train_settings"]["epochs"]
    if save_epochs is None:
        save_epochs = loader.model_cfg["runner"]["snapshots"]["save_epochs"]

    updates = {
        "runner.snapshots.save_epochs": save_epochs,
        "train_settings.epochs": epochs,
        "train_settings.batch_size": batch_size,
        "train_settings.dataloader_workers": dataloader_workers,   
    }
    
    det_cfg = loader.model_cfg["detector"]
    if det_cfg is not None:
        if detector_epochs is None:
            detector_epochs = det_cfg["train_settings"]["epochs"]
        else:
            updates["detector.train_settings.epochs"] = detector_epochs
        if detector_save_epochs is None:
            detector_save_epochs = det_cfg["runner"]["snapshots"]["save_epochs"]
        else:
            updates["detector.runner.snapshots.save_epochs"] = detector_save_epochs
        updates["detector.train_settings.batch_size"] = detector_batch_size
        updates["detector.train_settings.dataloader_workers"] = detector_dataloader_workers

    loader.update_model_cfg(updates) # update model_cfg with the updates dictionary

    print ("loader.model_cfg:", loader.model_cfg)
    
    pose_task = Task(loader.model_cfg["method"])
    
    if pose_task == Task.TOP_DOWN:
        logger_config = None
        
        if loader.model_cfg.get("logger"): # None
            logger_config = copy.deepcopy(loader.model_cfg["logger"])
            logger_config["run_name"] += "-detector"

        print("############################################")
        print("logger_config:", logger_config)
        # print("flag : 86")

        if args.debug:
            logger_config = dict(type = "WandbLogger",
                                project_name = "primatepose",
                                tags = ["server8"],
                                group = "Dubug_v83_server8",
                                run_name = args.run_name,
                                )
        else:
            logger_config = dict(type = "WandbLogger",
                                project_name = "primatepose",
                                tags = ["eval"],
                                group = "pfm_v83_server8",
                                run_name = args.run_name,
                                )
        
        # skipping detector training if a detector_path is given
        if args.detector_path is None and detector_epochs > 0 and args.train_detector:
            train(
                loader=loader,
                run_config=loader.model_cfg["detector"],
                task=Task.DETECT,
                device=device,
                gpus=gpus,
                logger_config=logger_config,
                snapshot_path=detector_path,
            )
            
    if epochs > 0 and args.train_pose:
        train(
            loader=loader,
            run_config=loader.model_cfg,
            task=pose_task,
            device=device,
            gpus=gpus,
            logger_config=logger_config,
            # logger_config=loader.model_cfg.get("logger"),
            snapshot_path=snapshot_path,
        )
        
import os
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root")
    parser.add_argument("--pytorch_config")
    parser.add_argument("--train_file", default="train.json")
    parser.add_argument("--test_file", default="test.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpus", default=None, nargs="+", type=int)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save-epochs", type=int, default=None)
    parser.add_argument("--detector-epochs", type=int, default=None)
    parser.add_argument("--detector-save-epochs", type=int, default=None)
    parser.add_argument("--snapshot_path", default=None)
    parser.add_argument("--detector_path", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--train-pose", action="store_true", help="Whether to train pose model")
    parser.add_argument("--train-detector", action="store_true", help="Whether to train detector model")
    parser.add_argument("--run-name", type=str, default="default_run", help="Run name for wandb logging")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for pose model training")
    parser.add_argument("--dataloader-workers", type=int, default=16, help="Number of dataloader workers")
    parser.add_argument("--detector-batch-size", type=int, default=32, help="Batch size for detector model training")
    parser.add_argument("--detector-dataloader-workers", type=int, default=16, help="Number of dataloader workers for detector")
    args = parser.parse_args()
    
    # backup the train.sh file and the current file in the same folder    
    train_dir = os.path.dirname(args.pytorch_config)
    debug_dir = os.path.dirname(train_dir)
    print("debug_dir:", debug_dir)
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    train_sh_path = os.path.join(current_dir, "train.sh")
    
    # shutil.copy(current_file_path, os.path.join(debug_dir, "train.py"))
    # shutil.copy(train_sh_path, os.path.join(debug_dir, "train.sh"))
    
    main(
        args.project_root,
        args.train_file,
        args.test_file,
        args.pytorch_config,
        args.device,
        args.gpus,
        args.epochs,
        args.save_epochs,
        args.detector_epochs,
        args.detector_save_epochs,
        args.snapshot_path,
        args.detector_path,
        batch_size=args.batch_size,
        dataloader_workers=args.dataloader_workers,
        detector_batch_size=args.detector_batch_size,
        detector_dataloader_workers=args.detector_dataloader_workers,
        debug=args.debug
    )