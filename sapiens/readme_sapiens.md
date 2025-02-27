

# Instruction 


# Lite

https://github.com/facebookresearch/sapiens/blob/main/lite/README.md


## Lite Pose

export SAPIENS_ROOT=/home/ti_wang/Ti_workspace/sapiens
export SAPIENS_LITE_ROOT=$SAPIENS_ROOT/lite

mkdir $SAPIENS_ROOT/sapiens_lite_host

mkdir -p ./torchscript/pretrain/checkpoints/sapiens_0.3b

mkdir -p ./torchscript/pose/checkpoints/sapiens_0.3b

## lite seg


- download the model from huggingface:    
    - https://huggingface.co/facebook/sapiens-seg-0.3b-torchscript/blob/main/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2

mkdir $SAPIENS_ROOT/sapiens_lite_host

mkdir -p ./torchscript/seg/checkpoints/sapiens_0.3b

cd /home/ti_wang/Ti_workspace/sapiens/lite/scripts/demo/torchscript


## lite depth


git clone https://huggingface.co/facebook/sapiens-depth-1b-torchscript


# Config Env


