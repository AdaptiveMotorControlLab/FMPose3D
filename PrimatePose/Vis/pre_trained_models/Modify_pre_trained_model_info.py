# for video adaption

import torch

pose_model_path = "/home/ti_wang/Ti_workspace/PrimatePose/Vis/pre_trained_models/aptv2/snapshot-200.pt" 

state_dict = torch.load(pose_model_path, weights_only=False)

target_path = "/home/ti_wang/Ti_workspace/PrimatePose/Vis/pre_trained_models/aptv2/aptv2_200.pt"

torch.save({"model": state_dict["model"]}, target_path)
# /home/ti_wang/Ti_workspace/PrimatePose/clustering