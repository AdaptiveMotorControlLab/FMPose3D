pfm_root=$(dirname $(dirname $(realpath $0)))
data_path_prefix="/mnt/data/tiwang"
data_root=${data_path_prefix}"/v8_coco"

# data_path_prefix="${project_root}/data/tiwang"

debug=0
gpu_id="1"
mode="train"
name=omc
file=${name}_pose_hrnet
# file=${name}_ori_kepts_bbox_pose_hrnet
# file=${name}_pose_resnet
# file=${name}_detector_fasterrcnn
dataset_file=${name}

# for splitted datasets
train_file=${data_path_prefix}/primate_data/splitted_train_datasets/${dataset_file}_train.json
test_file=${data_path_prefix}/primate_data/splitted_test_datasets/${dataset_file}_test.json
# test_file=/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/samples/lote_test_sampled_20.json
# test_file=${data_path_prefix}/primate_data/samples/oms_test_sampled_nums_50.json

pytorch_config_path=${pfm_root}/project/split/${file}_train/train/pytorch_config.yaml
# pytorch_config_path=/home/ti_wang/Ti_workspace/PrimatePose/project/pfm_pose_hrnet_train/train/pytorch_config.yaml

# snapshot_path=${pfm_root}/project/split/${file}_train/train/snapshot-200.pt
# snapshot_path=${pfm_root}/project/split/${file}_train/train/snapshot-best-081.pt
snapshot_path=/home/ti_wang/Ti_workspace/PrimatePose/project/pfm_pose_hrnet_train/train/snapshot-best-056.pt
# snapshot_path=../project/split/riken_ori_kepts_bbox_pose_hrnet_train/train/snapshot-200.pt
# snapshot_path=/home/ti_wang/Ti_workspace/PrimatePose/project/pfm_goodpose_merged_pose_hrnet_train/train/snapshot-best-020.pt
# detector_snapshot_path=${project_root}/project/split/${file}_train/train/snapshot-detector-020.pt
# detector_snapshot_path=/home/ti_wang/Ti_workspace/PrimatePose/project/split/ak_pose_hrnet_train/train/snapshot-detector-248.pt
# detector_snapshot_path=/home/ti_wang/Ti_workspace/PrimatePose/project/pfm_merged_checked_detector_fasterrcnn_train/train/snapshot-detector-best-083.pt
detector_snapshot_path=""
# detector_snapshot_path=/home/ti_wang/Ti_workspace/PrimatePose/project/split/oms_detector_fasterrcnn_train/train/snapshot-detector-best-021.pt


# no detector
python evaluation.py --project_root $data_root --pytorch_config_path $pytorch_config_path --snapshot_path $snapshot_path --train_file $train_file --test_file $test_file
# --detector_path $detector_snapshot_path