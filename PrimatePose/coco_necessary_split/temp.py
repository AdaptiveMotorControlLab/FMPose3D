


    # Draw bounding boxes if provided
    if bounding_boxes is not None:
        boxes, scores = bounding_boxes
        for box, score in zip(boxes, scores):
            if score > bboxes_pcutoff:
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                                  color=bounding_boxes_color, alpha=alpha_value)
                ax.add_patch(rect)

    if pred_bodyparts is not None:
        num_individuals, num_keypoints = pred_bodyparts.shape[:2]
        colors = get_cmap(num_keypoints if mode == "bodypart" else num_individuals + 1, name=colormap)

        # Draw skeletons first if provided
        if skeleton is not None:
            for individual_idx in range(num_individuals):
                for joint1_idx, joint2_idx in skeleton:
                    if keypoint_vis_mask is not None and not (keypoint_vis_mask[joint1_idx] and keypoint_vis_mask[joint2_idx]):
                        continue
                        
                    joint1 = pred_bodyparts[individual_idx, joint1_idx]
                    joint2 = pred_bodyparts[individual_idx, joint2_idx]
                    
                    if joint1[2] > p_cutoff and joint2[2] > p_cutoff:
                        color = colors(joint1_idx) if mode == "bodypart" else colors(individual_idx)
                        ax.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], 
                               color=color, alpha=alpha_value)

        # Draw keypoints
        for idx in range(num_keypoints):
            if keypoint_vis_mask is not None and not keypoint_vis_mask[idx]:
                continue
                
            for individual_idx in range(num_individuals):
                color = colors(idx) if mode == "bodypart" else colors(individual_idx)
                
                # Draw ground truth if available
                if gt_bodyparts is not None:
                    gt = gt_bodyparts[individual_idx, idx]
                    if gt[2] > 0:  # If keypoint is visible in ground truth
                        ax.plot(gt[0], gt[1], labels[0], color=color, 
                               alpha=alpha_value, markersize=dot_size)
                
                # Draw prediction
                pred = pred_bodyparts[individual_idx, idx]
                if pred[2] > p_cutoff:
                    marker = labels[1] if pred[2] > 0.8 else labels[2]
                    ax.plot(pred[0], pred[1], marker, color=color, 
                           alpha=alpha_value, markersize=dot_size)
                    
                    # Add keypoint name if provided
                    if keypoint_names is not None:
                        ax.text(pred[0], pred[1], keypoint_names[idx], 
                               color=color, fontsize=8)