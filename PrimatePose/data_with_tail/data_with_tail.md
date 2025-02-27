

# Add skeleton to the def visualize_predictions(*) --failed


- the path to the file that contains the visualize_predictions:
    - DeepLabCut/deeplabcut/pose_estimation_pytorch/apis/evaluate.py

- add skeleton option to the visualize_predictions function
    - pass the skeleton to:
        ```python
            plot_gt_and_predictions(
            image_path=image_path,
            output_dir=output_dir,
            gt_bodyparts=visible_gt,
            pred_bodyparts=visible_pred,
            bounding_boxes=bounding_boxes,
            dot_size=dot_size,  # Pass the adaptive dot size
            )
        ```
    - finally plot the skeleton:
        ```python
            # Plot regular bodyparts
            ax = make_multianimal_labeled_image(
                frame,
                ground_truth,
                predictions[:, :, :2],
                predictions[:, :, 2:],
                colors,
                dot_size,
                alpha_value,
                p_cutoff,
                ax=ax,
                bounding_boxes=bounding_boxes,
                bounding_boxes_color=bounding_boxes_color,
                bboxes_cutoff=bboxes_pcutoff,
            )
    ```
the logic for plotting the skeleton in make_multianimal_labeled_image function should be:
- for loop, for each connection(e.g. [A,B] ) in the skeleton,
    - if both keypoint A and B is reliable, then we plot the line between A and B
    - if one of the keypoint is not reliable, then we do not plot the line
