
- current data used is PFM_v8.2, which is stored in folder: /home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2
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

the structure of the V8.2 json file is like this:
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