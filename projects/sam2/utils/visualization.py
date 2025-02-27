import numpy as np
import torch
import cv2

# Define skeleton for visualization
PFM_SKELETON = [
    [3, 5], [4, 5], [6, 3], [7, 4],
    [5, 12], [13, 12], [14, 12], [2, 17],
    [19, 13], [20, 14], [21, 19], [22, 20],
    [23, 21], [24, 22], [25, 12], [26, 12],
    [25, 27], [26, 27], [25, 28], [26, 29],
    [27, 28], [27, 29], [28, 30], [29, 31],
    [30, 32], [31, 33], [27, 34], [34, 35],
    [35, 36], [36, 37]
]

# Define color palette for different objects (BGR format)
# Avoid Blue, Red, Yellow as they are reserved for special purposes
COLOR_PALETTE = [
    (128, 0, 128),   # Purple
    (0, 165, 255),   # Orange
    (255, 0, 255),   # Magenta
    (0, 128, 128),   # Brown
    (255, 191, 0),   # Deep Sky Blue  
    (180, 105, 255), # Pink
    (128, 128, 0),   # Teal
    (147, 20, 255),  # Deep Pink
    (127, 255, 212), # Aquamarine
]

def calculate_point_size(frame_width, frame_height):
    """Calculate appropriate point size based on image resolution"""
    # 基于图像对角线长度计算点的大小
    diagonal = np.sqrt(frame_width**2 + frame_height**2)
    
    # 设置点的大小为对角线长度的千分之一到千分之二之间
    point_size = max(3, int(diagonal * 0.0015))  # 最小为3像素
    
    return point_size

def draw_predictions(frame, mask, bbox, keypoints, confidences, color):
    """Draw all predictions on frame with specified color"""
    frame_vis = frame.copy()
    height, width = frame.shape[:2]
    
    # Calculate skeleton color (slightly darker)
    skeleton_color = tuple(int(c * 0.8) for c in color)
    
    # Calculate appropriate point size and line thickness based on image resolution
    point_size = calculate_point_size(width, height)
    line_thickness = max(1, int(point_size / 2*1.5))
    
    # Draw mask
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    if mask is not None:
        if len(mask.shape) > 2:
            mask = mask.squeeze()
        
        mask = (mask > 0).astype(np.uint8)
        
        # Create 3-channel mask with specified color
        mask_3d = np.stack([
            mask * color[0],  # Blue channel
            mask * color[1],  # Green channel
            mask * color[2]   # Red channel
        ], axis=2).astype(np.uint8)
        
        # Apply mask overlay
        frame_vis = cv2.addWeighted(frame_vis, 1, mask_3d, 0.2, 0)

    # Draw bbox with specified color
    if isinstance(bbox, np.ndarray):
        bbox = bbox.flatten()
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    cv2.rectangle(frame_vis, (x1, y1), (x1+x2, y1+y2), color, line_thickness)
    
    confidence_threshold = 0.35
    # Draw keypoints and skeleton
    if isinstance(keypoints, np.ndarray):
        if len(keypoints.shape) == 3:  # Shape: (N, 37, 2)
            for instance_idx in range(len(keypoints)):
                instance_keypoints = keypoints[instance_idx]
                instance_confidences = confidences[instance_idx]
                
                # Draw skeleton connections with skeleton color
                for connection in PFM_SKELETON:
                    idx1, idx2 = connection[0]-1, connection[1]-1
                    if (instance_confidences[idx1] > confidence_threshold and 
                        instance_confidences[idx2] > confidence_threshold):
                        pt1 = tuple(map(int, instance_keypoints[idx1]))
                        pt2 = tuple(map(int, instance_keypoints[idx2]))
                        cv2.line(frame_vis, pt1, pt2, skeleton_color, line_thickness)
                
                # Draw keypoints with specified color
                for kp_idx, (kp, conf) in enumerate(zip(instance_keypoints, instance_confidences)):
                    if conf > confidence_threshold:
                        x, y = int(kp[0]), int(kp[1])
                        cv2.circle(frame_vis, (x, y), point_size, color, -1)
    
    return frame_vis