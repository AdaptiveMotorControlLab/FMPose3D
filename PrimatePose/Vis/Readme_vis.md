

## vis_image_video_dlc.py

make a script to estimation primate pose in image or videos using pre-trained model using DLC;

do not use "try", please make the code concise, add the necessary comments to understand the code
Steps:
    load pre-trained model (please refer to inference_pose_detector_SAM2_aptv2.py);  
    the path to: image or video Path;
    - if it's image, then load it;
    output dir?
   - for the visualization code, please use  plot_gt_and_predictions_PFM(*) in evaluation_vis.py.

- video:
for the process_video() function, I hope this can be realised by using process image multi-times;

we need to extract the original video to a folder of originl images ( use idx represents the time order), and put this folder in the target_folder;  

and save the generate images to a folder named results, also in the tatget_folder;

finally, we need to contract the generated images to video;


## Dynamic Skeleton

PFM_SKELETON = [
    [1, 11], # [head, neck]
    [2, 4], [3, 4], [5, 2], [6, 3],
    [12, 11], [13, 11], 
    [18, 12], [19, 13], [20, 18], [21, 19],
    [22, 20], [23, 21],
    [26, 11],
    [24, 26], [25, 26], [24, 27], [25, 28],
    [27, 29], [28, 30],
    [29, 31], [30, 32], [26, 33], [33, 34],
    [34, 35], [35, 36]
    ]

target: finish function: def get_dynamic_skeleton(skeleton, keypoints, p_cutoff=0.6)

How to make it dynamic?

The pipeline of Dynamic skeleton:

Plan A:
- make a Dictionary to store special connection rules
    dict_name_to_idx = {"L_S" : 12, "R_S" : 13, "L_Elbow": 18, "R_Elbow": 19, "neck": 11,
                    "L_W": 20, "R_H": 23}
    # templete
    special_connections = {
        # Format: (point_to_check, original_connection, alternative_connection)
        "L_S": {(("L_S", "L_Elbow"), ("L_S", "neck")), ("L_Elbow", "neck")},  # L_S: ori_connection: {L_S to L_elbow, L_S to neck}; alt_connection: L_Elbow to neck if L_S is above threshold
        "R_S": {(("R_S", "R_Elbow"), ("R_S", "neck")), ("R_Elbow", "neck")},  # R_S: ori_connection: {R_S to R_elbow, R_S to neck}; alt_connection: R_Elbow to neck if R_S is above threshold
        "L_W": {(("L_H", "L_W"), ("L_W", "L_Elbow")), ("L_H", "L_Elbow")},  # L_W: ori_connection: {L_H to L_W, L_W to L_Elbow}; alt_connection: L_H to L_Elbow if L_W is above threshold
        "R_H": {(("R_H", "R_W"), ("R_H", "R_Elbow")), ("R_H", "R_Elbow")},  # R_H: ori_connection: {R_H to R_W, R_H to R_Elbow}; alt_connection: R_H to R_Elbow if R_H is above threshold
    }
- initially, we set dynamic_skeleton = skeleton.copy()
- For each keypoint in special_connections:
    - check wether it's conf below the p_cutoff;
        - if it's conf above the p_cutpff; skip;
        - if it's conf below the p_cutoff: make changes!
            -  for each keypoint pair (a, b), since here we mainly consider the connection, so(b, a) also the same connection;
            - For dynamic_skeleton, we drop the original_connection; and add the alternative_connection to the dynamic_skeleton;

Plan B:

modifies the dynamic_skeleton;
if one keypoint below the threshold, then remove it, and change all connections to this node with another keypoint;


Plan C:
we can use tree structure to realise this;
please refer to human skeleton code:
```python
h36m_skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                  16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                         joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                         joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])

```