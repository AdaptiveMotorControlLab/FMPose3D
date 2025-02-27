# split_v8_json_file_to_train_and_test

- PFM v8 format test.json file:
  - structure for the json file:
    ```bash
        Root structure: Dictionary
        images: list (length: 1x)
            First element type: dict
            First element keys: ['file_name', 'width', 'height', 'id', 'source_dataset']
        annotations: list (length: 1x)
            First element type: dict
            First element keys: ['id', 'image_id', 'category_id', 'bbox', 'keypoints', 'num_keypoints', 'area', 'iscrowd']
        categories: list (length: 1)
            First element type: dict
            First element keys: ['id', 'name', 'supercategory', 'keypoints']
    ```

The target for this function:
- split the PFM v8 format json file into train and test json files.
    - the input parameter should include the path to the json file that will be split, and the ratio of the train and test data. (default is 80% for train and 20% for test)
    - the train json file should contain 80% of the data, and the test json file should contain 20% of the data.
    - save the split json files to the specified path: /home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/split_for_original_dataset_without_testset
    - the split json files should be named as {dataset_name}_train.json and {dataset_name}_test.json


# find_annotations_with_pose.py

- target: 
    - find the annotations with pose in the json file. And keep the ouput json file's structure is same as the input json file.
    - the path to the input json file: /home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_train_datasets/kinka_train.json
    - the path to the output file: /home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_train_datasets/kinka_train_with_pose.json
    - judge whether the annotation has pose: 
        - check all the vis_labels in the 'keypoints', if all these vis_labels are -1, then the annotation is considered without pose. Otherwise, we consider it has pose.


# extract_annotations_with_tails_ap.py

Target:
- extract the annotations with tails  from the json file of ap dataset.
    - the path to ap test dataset: /home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_test_datasets/mp_test.json
- the tail keypoints are: ["root_tail", "mid_tail", "mid_end_tail", "end_tail"], if one of these keypoints exists, the annotation is considered with a tail.
- save the extracted annotations to the specified folder: /home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/samples, with the name of {dataset_name}_{mode}_with_tails.json

- save the printed text to a text file, with the name of {dataset_name}_{mode}_with_tails.txt, with the same path as the json file.


# split_v8_json.py

This script handles the conversion and splitting of primate pose annotation data between V7 and V8 formats.

Main functionalities:
1. Converts V7 format JSON annotations to V8 format
2. Splits the dataset based on source datasets (e.g., oms, kinka)
3. Preserves all keypoint annotations and metadata
4. Handles special cases like the OMS dataset mapping

Key features:
- Input: V7 format JSON files (train/test/val)
- Output: Multiple V8 format JSON files, one for each source dataset
- Maintains data integrity during conversion
- Includes utility functions for:
  - Checking image file existence
  - Looking up filenames by image ID

Usage example:
```python
input_file = '/path/to/v7/pfm_train_apr15.json'
output_dir = '/path/to/output/splitted_datasets/'
model = "train"  # or "test" or "val"
convert_v7_to_v8_and_split_datasets(input_file, output_dir, model)
```
```text
dataset_id_to_name {'1': 'oap', '2': 'omc', '3': 'mp', '4': 'ap10k', '5': 'aptv2', '6': 'ak', '7': 'mbw', '8': 'kinka', '9': 'riken', '10': 'mit', '11': 'lote', '12': 'deepwild', '13': 'anipose', '14': 'oap_new', '15': 'omc_new', '16': 'ak_new', '18': 'chimpact', '17': 'oms'}
```

# split_v8_json_with_ori_kepts.py

save the splited json files in: /mnt/data/tiwang/primate_data/data_v8.1/splitted_{model}_datasets/

if the dataset_name in Dataset_skeleton_config then:
- in the output_category:
    - name: use the original dataset_name
    - keypoints: replace it with the 'keypoints' in Dataset_skeleton_config

for output_annotation:
- replace 'num_keypoints' with the 'num_keypoints' in Dataset_skeleton_config

# json_replace_keypoints_with_original_keypoints_v8.py

- target:
    - replace the keypoints with the original keypoints in the json file.
    - json files is the v8 version pfm json file.
    - the format of the json file:
    - structure for the PFM v8 format test.json file:
            ```bash
                Root structure: Dictionary
                images: list (length: 1x)
                    First element type: dict
                    First element keys: ['file_name', 'width', 'height', 'id', 'source_dataset']
                annotations: list (length: 1x)
                    First element type: dict
                    First element keys: ['id', 'image_id', 'category_id', 'bbox', 'keypoints', 'num_keypoints', 'area', 'iscrowd']
                categories: list (length: 1)
                    First element type: dict
                    First element keys: ['id', 'name', 'supercategory', 'keypoints']
            ```
Step:
- read the v8 pfm json file
    - path to the json file: /home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_test_datasets/riken_test.json
    - the output json file should be saved to the same path as the input json file, with the name of {json_file_name}_original_keypoints.json

- replace the keypoints with the original keypoints
    - data[idx]['annotations']'[keypoints'] = file[annotations][keypoints]


# cal_keypoint_number_subdatasets.py

- target:
    - calculate the number of keypoints for each dataset in the json file.
        - the json files are stored in the folder: /home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_train_datasets
        - load the json file
            ```python
                with open(json_file_path, 'r') as f:
                    data = json.load(f)
            ```
        - get the 'keypoints' from the 'annotations'
            ```python
                keypoints = data['annotations']['keypoints']
            ```
            - the keypoints is a list, each element is a list, each list contains 3 elements: [x, y, visibility], where visibility is -1, 0, 1, or 2.
            - if the visibility is -1 or 0, it means the keypoint is not exist.
        - for every sample in annotations, count the number of each keypoints, if the vis_label is -1 or 0, it means the keypoint is not exist.
        - we should record the number for each 
    - save the result to a text file, with the name of {dataset_name}_keypoints_number.txt, to a given folder: /home/ti_wang/Ti_workspace/PrimatePose/analyse_json/keypint_number_subdatasets

# evaluation.py
- for the saved images, I need to create a folder like this:
    output_path = Path(pytorch_config_path).parent.parent / "results" / "with_tail"
- further I want to use which snapshot to evaluate the model, I need to further create a folder under the output_path, with output_path/"pfm_pose_hrnet_train".
the "pfm_pose_hrnet_train" is from the snapshot_path:
    -snapshot_path=/home/ti_wang/Ti_workspace/PrimatePose/project/pfm_pose_hrnet_train/train/snapshot-best-056.pt  

# riken

## cal_num_only_one_annotations.py

target file: /home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/splitted_train_datasets/riken_train.json

target: calculate for riken dataset:
            - how many annotations only has one annotation;
            - how many annotations only has tail annotation;
             - the tail index of PFM format are: [33, 34, 36]
            - how many empty annotations; 
                - all vis_labels are -1;


# Dataset Version

## V8.1

transform the v7 version data to coco format that needed by dlc;

## V8.2

in V8.1, for some sub-datasets, the keypoint mapping is not correct, so we need to correct the keypoint mapping;

based on V8.1, we correct the keypoint mapping for some sub-datasets, and save the correct keypoint mapping to the json file;


## process of V8.2

file: V8.2_correct_keypoint_mapping.py

- create a folder name PFM_V8.2 in folder: /home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data
- the data of PFM_v8.0 is stored in folder: /home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/data_v8.0
   - structure of this folder:
      ```bash
      .
        ├── splitted_test_datasets
        │   ├── ak_new_test.json
        ...
        │   └── riken_test.json
        ├── splitted_train_datasets
        │   ├── ak_new_train.json
        ...
        │   └── riken_train.json
        └── splitted_val_datasets
            ├── anipose_val.json
            ...
            └── riken_val.json
    ```
    the structure of the V8.0 json file is like this:
    ```bash
    JSON structure:
images
    [List of 14374 items]
        file_name
            (str)
        width
            (int)
        height
            (int)
        id
            (int)
        source_dataset
            (str)
annotations
    [List of 14374 items]
        id
            (int)
        image_id
            (int)
        category_id
            (int)
        bbox
            [List of 4 items]
                (int)
        keypoints
            [List of 111 items]
                (int)
        num_keypoints
            (int)
        area
            (int)
        iscrowd
            (int)
categories
    [List of 1 items]
        id
            (int)
        name
            (str)
        supercategory
            (str)
        keypoints
            [List of 37 items]
                (str)
    ```

 - load PFM_v8.0 and correct the keypoint mapping for some sub-datasets;
    - create an element like dict that can contains keypoint mapping; we only put the problems dataset mapping in this element.

    - when we load the dataset, we need to extract the dataset_name; 
        - if dataset_name in the element:
            - keypoints = data['annotations']['keypoints']
            - The super set (PFM) has 37 keypoints, so len(keypoints[0]) = 37
            - for example, the keypoint mapping from the original dataset to v8.0 format in oap dataset is "V8.0_keypoint_mapping". The number is the index of the original dataset, -1 means this keypoint is not exist in the super set (PFM). 
            "V8.0_keypoint_mapping"[1]=3, means the idx_PFM = 1 keypoint in PFM  is the idx_original=3 keypint in the original dataset.:
            ```text
                "V8.0_keypoint_mapping": [
                -1, 3, 1, 2, 0, -1, -1, 
                -1, -1, -1, -1, 4, 5, 8, 
                -1, -1, -1, -1, 6, 9, 7, 
                10, -1, -1, -1, -1, 11, 12, 14, 
                13, 15, -1, -1, -1, -1, -1, -1
                ],
                "V8.2_keypoint_mapping": [ 
                    -1, 3, 1, 2, 0, -1, -1, 
                    -1, -1, -1, -1, 4, 5, 8, 
                    -1, -1, -1, -1, 6, 9, -1, -1,
                    7, 10, -1, -1, 11, 12, 14, 
                    13, 15, -1, -1, -1, -1, -1, -1
                ]
            ``` 
              we need to change the keypoint position to V8.2 format;
            - use the pre-defined V8.2 keypoint mapping to correct the keypoint mapping.


def transform_keypoints(keypoints, v80_mapping, v82_mapping):
```bash
    # Plan A: use keypoint in V80 to replace keypoint in V82
    # v82_keypoint = [-1] * len(v82_mapping)*3
    # for pfm_idx_v82, v82_orig_idx in v82_mapping:
    #   if v82_orig_idx == -1, continue
    #   for pfm_idx_v80, v80_orig_idx in v80_mapping:
    #       if v80_orig_idx == v82_orig_idx:
    #           v82_keypoint[pfm_idx_v82*3] = v81_keypoint[pfm_idx_v80*3]
    #           v82_keypoint[pfm_idx_v82*3+1] = v81_keypoint[pfm_idx_v80*3+1]   
                # v82_keypoint[pfm_idx_v82*3+2] = v81_keypoint[pfm_idx_v80*3+2]
                # brea
    # Plan B: record the keypoint mapping from original to pfm of v80 and v82. like: v80[original_idx] = pfm_idx;
    #  use one for loop to fill the value of v82_keypoint using v80_keypoint.
    # plan B is faster and more efficient;
```