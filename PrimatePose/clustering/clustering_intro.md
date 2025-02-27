

# read the h5 file and analyse the data structure

/home/ti_wang/Ti_workspace/PrimatePose/clustering/CollectedData_dlc.h5

# Target

my target is to transform my primate data to a h5 file that can be used for clustering.

my primate data is stored in json files, the structure is like this:

- PFM format test.json file:
  - path: /home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_test_datasets/mbw_test.json
  - structure for the json file:
    ```bash
        Root structure: Dictionary
        images: list (length: 13)
            First element type: dict
            First element keys: ['file_name', 'width', 'height', 'id', 'source_dataset']
        annotations: list (length: 13)
            First element type: dict
            First element keys: ['id', 'image_id', 'category_id', 'bbox', 'keypoints', 'num_keypoints', 'area', 'iscrowd']
        categories: list (length: 1)
            First element type: dict
            First element keys: ['id', 'name', 'supercategory', 'keypoints']
    ```

for the h5 file, one example is /home/ti_wang/Ti_workspace/PrimatePose/clustering/CollectedData_dlc.h5.


```bash
Multi-Index Column Structure:
----------------------------

Column levels:
Level 0 name: scorer
Unique values: ['dlc']
Count: 1

Level 1 name: individuals
Unique values: ['mus1', 'mus2', 'mus3']
Count: 3

Level 2 name: bodyparts
Unique values: ['snout', 'leftear', 'rightear', 'shoulder', 'spine1', 'spine2', 'spine3', 'spine4', 'tailbase', 'tail1', 'tail2', 'tailend']
Count: 12

Level 3 name: coords
Unique values: ['x', 'y']
Count: 2


Example of full column names (first 3):
---------------------------------------
('dlc', 'mus1', 'snout', 'x')
('dlc', 'mus1', 'snout', 'y')
('dlc', 'mus1', 'leftear', 'x')
File Structure Overview:
-----------------------
Number of frames: 11
Shape: (11, 72)

Column Structure:
----------------
Scorers: ['dlc']
Individuals: ['mus1', 'mus2', 'mus3']
Bodyparts: ['snout', 'leftear', 'rightear', 'shoulder', 'spine1', 'spine2', 'spine3', 'spine4', 'tailbase', 'tail1', 'tail2', 'tailend']
Coordinates: ['x', 'y']

Data Structure Example:
----------------------
scorer                                            dlc                                                              ...                                                                        
individuals                                      mus1                                                              ...        mus3                                                            
bodyparts                                       snout                 leftear                rightear              ...       tail1                   tail2                 tailend            
coords                                              x           y           x           y           x           y  ...           x           y           x           y           x           y
labeled-data/videocompressed1/img0000.png   58.457483  224.952005   45.363951  250.763052   65.027234  245.568977  ...  338.111184  415.438456  308.547788  396.157981  282.840488  367.237268
labeled-data/videocompressed1/img0074.png   46.350399  163.677952   66.671373  174.177122   72.428982  153.178782  ...  604.192480  308.273433  593.913946  336.427679  572.016199  365.028818
labeled-data/videocompressed1/img0592.png  160.232840  298.436592  145.867626  326.032925  168.549543  325.276861  ...  592.279137  312.154107  573.893293  280.635517  561.886211  263.000115
labeled-data/videocompressed1/img1150.png   89.690471  208.655027   99.005195  222.627112  106.249979  201.410242  ...  384.139220  432.208383  417.258235  445.662983  448.307313  451.872798
labeled-data/videocompressed1/img1503.png  138.446965  266.125328  160.235571  269.440985  150.288599  246.705048  ...   71.186484  220.179788   63.134173  254.757359   60.292181  288.861265

[5 rows x 72 columns]

```


steps:

1. load the json file, data = json.load(open(json_path, 'r'))
2. extract data that we need:
    - bodyparts: data["categories"][0]["keypoints"]
    - calculate each image has how many annotations:
        ```python
        image_ids_to_image_name = {}
        for img in data["images"]:
            image_ids_to_image_name[img["id"]] = img["file_name"]
        imageID_to_annotations = {}
        for ann in data["annotations"]:
            if ann["image_id"] not in image_to_annotations:
                image_to_annotations[ann["image_id"]] = []
            image_to_annotations[ann["image_id"]].append(ann)
        ```
    - the image that has the most annotations is the number of individuals, we get N;
    the data for each row:
        - we can use the image_name as the row index; 
            - index = ["image_name_1", "image_name_2", "image_name_3", ...]
        - we set data = prediction_data, we get it from data["annotations"][:]["keypoints"]
            - data["annotations"][0]["keypoints"].shape = (37, 3); x, y, vis_label; is visible =-1, keep the x y to 0;
            - we should convert the coco format data to the format that we need for h5 file -> (num_images, num_keypoints, 2)
    - build the dataframe
        - df = pd.DataFrame(prediction_data, index=index, columns=build_dlc_dataframe_columns(scorers, bodyparts, individuals))
        ```python
        - for the scorers, we can set it as: scorers: ['ti']
        - for individuals, we can set it as: individuals: ["monkey1","monkey2","monkey3", "monkey4", "monkeyN"], N is the length for individuals
    - save the dataframe to h5 file
        - df.to_hdf(h5_path, key='data', mode='w')

- save the h5 file to this path: /home/ti_wang/Ti_workspace/PrimatePose/clustering/data
- and transform the h5 file to a csv file

# visualize_images_according_to_image_ids

target: make a function that can visualize the images according to the image_ids

the input:
    - image_ids: a list of image_ids
    - json_file_path: the path of the json file
steps:
 get the image_ids_to_annotations mapping from the json file
 for each image_id in image_ids:
    - get the annotations for the image_id
        do the visualization for each annotation
        - def visualize_annotation_pfm(img, annotation, skeleton, keypoint_names, vis_bbox=True, label=["+", ".", "x], mode="GT")
            - bbox = annotation["bbox"]
            - x1, y1, width, height = bbox
            - x2, y2 = x1 + width, y1 + height
            - if vis_bbox is True:
                - cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            - keypoints = annotation["keypoints"]
            - color
                - num_colors = len(keypoint_names)
                - colors = get_cmap(num_colors, name="rainbow")
                    - function definition:
                    ```python
                        def get_cmap(n: int, name: str = "hsv") -> Colormap:
                            """
                            Args:
                                n: number of distinct colors
                                name: name of matplotlib colormap

                            Returns:
                                A function that maps each index in 0, 1, ..., n-1 to a distinct
                                RGB color; the keyword argument name must be a standard mpl colormap name.
                            """
                            return plt.cm.get_cmap(name, n)
                    ```
            - vis keypoints
            if skeleton is not None:
                - vis skeleton
                - if model=="GT":
                    plase use "+" to represent the keypoints
                - if model=="Pred":
                    if confidence is greater than pcuoff:
                        please use "." to represent the keypoints
                    else:
                        please use "x" to represent the keypoints
            return img
        save the img to the path: /home/ti_wang/Ti_workspace/PrimatePose/clustering/data/{dataset_name}/{image_id_ann_id}.png

 - please refer to this code:
    ```python
    def visualize_annotation(img, annotation, color_map, categories, skeleton, image_id, annotation_id=None, dataset_config=None):
    # Bounding box visualization
    bbox = annotation["bbox"]
    x1, y1, width, height = bbox
    x2, y2 = x1 + width, y1 + height
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    print("________________________")
    
    id = annotation["id"]
    if "keypoints" in annotation:
        keypoints = np.array(annotation["keypoints"]).reshape(-1, 3)
        keypoint_names = categories[0]["keypoints"]  # Get keypoint names from categories
        # keypoint_names = categories  # Get keypoint names from categories
         
        # Calculate scaling factor based on image size
        img_height, img_width = img.shape[:2]
        scale_factor = max(img_width, img_height) / 1000
        
        # print("scale_factor:", scale_factor)
        # Set minimum and maximum limits for scaling factor
        # scale_factor = max(0.5, min(scale_factor, 2))
        
        existing_text_positions = []  # Track existing text positions to avoid overlap
        
        for i, (x_kp, y_kp, v) in enumerate(keypoints):
            if v > 0:
                keypoint_label = keypoint_names[i]
                # draw the keypoint
                cv2.circle(
                    img,
                    center=(int(x_kp), int(y_kp)),
                    radius=int(7 * scale_factor),
                    color=color_map[keypoint_label],
                    thickness=-1,
                )
                
                # bright = compute_brightness(img, int(x1), int(y1))
                # txt_color = (10, 10, 10) if bright > 128 else (255, 255, 255)
                # txt_color = (10, 10, 10) if bright > 128 else (235, 235, 215)
               
                # Get the background color at the text position
                # print("x_kp:", x_kp, "y_kp:", y_kp)
                # print("img.shape:", img.shape)
                # print("image_id", image_id)
                # print("id", id)
                
                bg_color = img[int(y_kp), int(x_kp)].astype(int)
                txt_color = get_contrasting_color(bg_color)
                
                # adjust font scale and thickness based on scale factor
                font_scale = max(0.2, min(scale_factor, 1))*0.8
                thickness = max(1, int(scale_factor))
                
                y_text = int(y_kp) - int(15 * scale_factor)
                x_text = int(x_kp) - int(10 * scale_factor)
                # Ensure text does not go out of image bounds
                x_text = min(max(0, x_text), img_width - 100)
                y_text = min(max(0, y_text), img_height - 10)
            
                # Avoid overlapping text: Adjust position if it overlaps with previously drawn text
                for (existing_x, existing_y) in existing_text_positions:
                    if abs(x_text - existing_x) < 50 and abs(y_text - existing_y) < 20:
                        y_text += int(20 * scale_factor)  # Move text slightly downward if overlap detected
                # Record this position
                existing_text_positions.append((x_text, y_text))
                
                cv2.putText(
                    img,
                    keypoint_label,
                    (int(x_kp), y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    txt_color,
                    thickness + 0,
                    cv2.LINE_AA,
                )
                
                # Draw the black text as an outline
                # cv2.putText(
                #     img,
                #     keypoint_label,
                #     (int(x_kp), y_text),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     font_scale,
                #     (0, 0, 0),  # Black text as the outline
                #     thickness + 2,  # The outline is thicker than the main text
                #     cv2.LINE_AA,
                # )
                
                # # Draw the white text on top
                # cv2.putText(
                #     img,
                #     keypoint_label,
                #     (int(x_kp), y_text),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     font_scale,
                #     txt_color,  # The original white text
                #     thickness,
                #     cv2.LINE_AA,
                # )
                # Draw skeleton
                # for connection in skeleton:
                #     idx1, idx2 = connection
                #     x1_kp, y1_kp, v1 = keypoints[idx1 - 1]
                #     x2_kp, y2_kp, v2 = keypoints[idx2 - 1]
                #     if v1 > 0 and v2 > 0:
                #         keypoint_label1 = keypoint_names[idx1 - 1]
                #         color = color_map.get(keypoint_label1, (255, 255, 255))
                #         cv2.line(
                #             img,
                #             (int(x1_kp), int(y1_kp)),
                #             (int(x2_kp), int(y2_kp)),
                #             color,
                #             thickness=int(2 * scale_factor)
                #         )        
                 
                # Draw skeleton in original format
                connected_pfm_indices = find_connections(i, dataset_config)
                for pfm_idx in connected_pfm_indices:
                    x2_kp, y2_kp, v2 = keypoints[pfm_idx]
                    if v2 > 0:
                        cv2.line(img, (int(x_kp), int(y_kp)), (int(x2_kp), int(y2_kp)), (0, 255, 0), 2)
                        
    return img
    ```

# visualize_annoations

target: make a function that can visualize the images according to the image_ids

the input:
    - image_ids: a list of image_ids
    - json_file_path: the path of the json file
steps:
 get the image_ids_to_annotations mapping from the json file
 for each image_id in image_ids:
    - get the annotations for the image_id
        - visualize_annotation_pfm_(img, annotation, color_map, skeleton, keypoint_names, vis_bbox=True)
            - bbox = annotation["bbox"]
            - x1, y1, width, height = bbox
            - x2, y2 = x1 + width, y1 + height
            - if vis_bbox is True:
                - cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            - keypoints = annotation["keypoints"]
            - vis keypoints