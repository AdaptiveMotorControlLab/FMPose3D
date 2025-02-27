import json

path_json = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_train_datasets/riken_train.json"

with open(path_json, "r") as f:
    data = json.load(f)

images = data["images"]
print(len(images))