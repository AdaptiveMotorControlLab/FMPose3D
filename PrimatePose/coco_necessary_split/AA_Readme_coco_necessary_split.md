# functions


## analyze_video.sh and analyze_video.py

This script is used to analyze the video and get the pose estimation results from in the wild videos.

## evaluation_vis.py

write a similar function to replace the function

visualize_PFM_predictions(*):
- additional config:
    - show_keypoint_name: False
    - keypoint_vis_mask: None (only show the unmasked keypoints)
    - skeleton: if this option is provided, the skeleton will be drawn
    - output_dir: the directory to save the visualization results

keypoint_vis_mask should be defined by a given list  [xx, xx, xx, xx], please only use keypoint_vis_mask for visualization code, we do not need to use keypoint_vis_mask to filter the keypoints when processing the data.

### plot_gt_and_predictions_PFM(*) in evaluation_vis.py file:

target:
- if bounding_boxes is provided, visualize the bounding boxes on the image
- if GT and pred are provided, visualize the ground truth and predictions on the image
 - pred_bodyparts: [N, num_keypoints, 3] (x, y, confidence)
 - gt_bodyparts: [N, num_keypoints, 3] (x, y, vis_label)
 - for idx in range(len(pred_bodyparts)):
    - labels: List[str] = ["+", ".", "x"]
        - GT -> "+"
        - prediction with high confidence -> "."
        - prediction with low confidence -> "x"
    - if the keypoint_vis_mask[idx]=1 and confidence > pcutoff, the keypoint will be visualized, othersize not;
    - if gt_bodyparts is provided, the GT keypoint will be visualized, othersize not;
    - if keypoint_names is provided, show the keypoint names on the image: keypoint_names[idx]
- save the plotted image to the output_dir

- dot_size:
    - Currently, for high resolution images, the dot_size is a little too big; for low resolution images, the dot_size is a little bit too small.


plot_individual = True

if this is true, please save each animal as one image;
for multi-animal in one image, please add the index to the plotted image name;


# riken bandy

## build_config_riken_bandy.sh

## train_riken_bandy.sh

