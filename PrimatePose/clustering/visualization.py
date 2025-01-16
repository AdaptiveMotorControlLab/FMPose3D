# refer to this code:
# DeepLabCut/deeplabcut/utils/visualization.py
#
#

import json
import cv2
import numpy as np
import os
from typing import List, Dict, Optional, Union, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

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


def get_cmap(n: int, name: str = "rainbow") -> Colormap:
    """
    Get a matplotlib colormap with n distinct colors.
    
    Args:
        n: number of distinct colors
        name: name of matplotlib colormap

    Returns:
        A function that maps each index in 0, 1, ..., n-1 to a distinct RGB color
    """
    return plt.cm.get_cmap(name, n)

def get_color_map(keypoint_names: List[str], colormap_name: str = "rainbow") -> Dict[str, Tuple[int, int, int]]:
    """
    Generate a color map for keypoints using matplotlib colormap.
    
    Args:
        keypoint_names: List of keypoint names
        colormap_name: Name of matplotlib colormap to use
        
    Returns:
        Dictionary mapping keypoint names to BGR colors
    """
    cmap = get_cmap(len(keypoint_names), colormap_name)
    colors = {}
    
    for i, kp in enumerate(keypoint_names):
        # Get RGB colors from matplotlib (0-1 range)
        rgb = cmap(i)[:3]
        # Convert to BGR (0-255 range) for OpenCV
        bgr = tuple(int(x * 255) for x in rgb[::-1])
        colors[kp] = bgr
        
    return colors

def get_contrasting_color(bg_color: np.ndarray) -> Tuple[int, int, int]:
    """
    Get a contrasting text color based on background color.
    
    Args:
        bg_color: Background color in BGR format
        
    Returns:
        Contrasting color in BGR format
    """
    brightness = (0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0])
    return (0, 0, 0) if brightness > 128 else (255, 255, 255)

def visualize_annotation_pfm(
    img: np.ndarray,
    annotation: dict,
    keypoint_names: List[str],
    skeleton: Optional[List[List[int]]] = None,
    mode: str = "GT",
    confidence_threshold: float = 0.5,
    vis_bbox: bool = False,
    vis_text: bool = False,
    colormap_name: str = "rainbow",
    dpi: int = 200  # Added DPI parameter
) -> None:  # Changed return type to None since we're using plt directly
    """
    Visualize a single annotation on an image.
    
    Args:
        img: Input image as numpy array (BGR format)
        annotation: Annotation dictionary containing bbox and keypoints
        keypoint_names: List of keypoint names
        skeleton: Optional list of keypoint connections
        mode: Either "GT" or "Pred" to determine marker style
        confidence_threshold: Threshold for confidence in prediction mode
        vis_bbox: Whether to visualize bounding box
        vis_text: Whether to display keypoint labels
        colormap_name: Name of matplotlib colormap to use
        dpi: Resolution for the figure
    """
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure with the same size as the image
    dpi = plt.rcParams['figure.dpi']
    height, width = img.shape[:2]
    figsize = width / float(dpi), height / float(dpi)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_rgb)
    
    # Create color map
    cmap = get_cmap(len(keypoint_names), colormap_name)
    
    # Draw bounding box if requested
    if vis_bbox and "bbox" in annotation:
        bbox = annotation["bbox"]
        x1, y1, width, height = [int(x) for x in bbox]
        rect = plt.Rectangle((x1, y1), width, height, fill=False, color='g', linewidth=1)
        ax.add_patch(rect)
    
    # Draw keypoints if they exist
    if "keypoints" in annotation:
        keypoints = np.array(annotation["keypoints"]).reshape(-1, 3)
        
        # Calculate scale factor based on image size
        img_height, img_width = img.shape[:2]
        scale_factor = max(img_width, img_height) / 1000
        
        # Draw skeleton if provided
        if skeleton is not None:
            for connection in skeleton:
                idx1, idx2 = connection
                x1_kp, y1_kp, v1 = keypoints[idx1 - 1]
                x2_kp, y2_kp, v2 = keypoints[idx2 - 1]
                if v1 > 0 and v2 > 0:
                    color = cmap(idx1 - 1)[:3]
                    ax.plot([x1_kp, x2_kp], [y1_kp, y2_kp], '-', color=color, 
                           linewidth=max(1, 1.5*scale_factor))
        
        # Draw keypoints
        for i, (x_kp, y_kp, v) in enumerate(keypoints):
            if v > 0:  # Only visualize visible keypoints
                color = cmap(i)[:3]  # Get RGB color directly from colormap
                
                # Calculate marker size based on image dimensions
                min_dim = min(img_width, img_height)
                marker_size = int(min_dim * 0.015)  # 1.5% of smallest dimension
                marker_size = max(6, min(marker_size, 25))  # Keep size between 6 and 25 pixels
                
                # Plot keypoint as '+'
                ax.plot(x_kp, y_kp, '+', color=color, markersize=marker_size, markeredgewidth=max(1, marker_size//8))
                
                # Add text labels if requested
                if vis_text:
                    y_text = y_kp - 15 * scale_factor
                    plt.text(x_kp, y_text, keypoint_names[i], 
                            color=color, 
                            fontsize=10*scale_factor,
                            ha='center', 
                            va='bottom')
    
    # Remove axes and white space
    ax.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

def visualize_sample_by_image_ids(
    image_ids: List[int],
    json_file_path: str,
    image_dir: str,
    output_dir: Optional[str] = None,
    mode: str = "GT",
    confidence_threshold: float = 0.5,
    vis_text: bool = False,
    colormap_name: str = "rainbow",
    dpi: int = 100  # Default DPI
) -> None:
    """
    Visualize images with their annotations based on image IDs.
    
    Args:
        image_ids: List of image IDs to visualize
        json_file_path: Path to the COCO format JSON file
        image_dir: Directory containing the images
        output_dir: Optional directory to save visualized images
        mode: Either "GT" or "Pred" to determine marker style
        confidence_threshold: Threshold for confidence in prediction mode
        vis_text: Whether to display keypoint labels
        colormap_name: Name of matplotlib colormap to use
        dpi: Resolution of output images
    """
    # Set matplotlib backend to Agg for non-interactive plotting
    plt.switch_backend('Agg')
    
    # Load JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Create mapping dictionaries
    image_id_to_name = {img['id']: img['file_name'] for img in data['images']}
    
    image_id_to_anns = {}
    for ann in data['annotations']:
        if ann['image_id'] not in image_id_to_anns:
            image_id_to_anns[ann['image_id']] = []
        image_id_to_anns[ann['image_id']].append(ann)
    
    # Get keypoint names and skeleton information
    keypoint_names = data['categories'][0]['keypoints']
    skeleton = PFM_SKELETON
    
    for image_id in image_ids:
        # Load image
        image_path = os.path.join(image_dir, image_id_to_name[image_id])
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        # Draw annotations
        annotations = image_id_to_anns.get(image_id, [])
        
        for ann in annotations:
            visualize_annotation_pfm(
                img=img,
                annotation=ann,
                keypoint_names=keypoint_names,
                # skeleton=skeleton,
                mode=mode,
                confidence_threshold=confidence_threshold,
                vis_text=vis_text,
                colormap_name=colormap_name,
                dpi=dpi
            )
            
            # Save image if output directory is provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                dataset_name = os.path.basename(json_file_path).split('.')[0]
                ann_id = ann.get('id', 'unknown')
                
                # Keep original file extension
                original_ext = os.path.splitext(image_id_to_name[image_id])[1]
                output_path = os.path.join(
                    output_dir,
                    f"{dataset_name}_{image_id}_{ann_id}_{image_id_to_name[image_id]}"
                )
                
                # Save with original format
                plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0, format=original_ext[1:])
            plt.close()  # Close the figure after saving

if __name__ == "__main__":
    # Example usage
    dataset_name = "ak"
    mode = "test"
    json_file_path = f"/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_{mode}_datasets/{dataset_name}_{mode}.json"
    image_dir = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/v8_coco/images"
    output_dir = f"/home/ti_wang/Ti_workspace/PrimatePose/clustering/data/{dataset_name}_{mode}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize some images
    image_ids = [39050, 39051, 39052]  # Replace with actual image IDs
    visualize_sample_by_image_ids(
        image_ids=image_ids,
        json_file_path=json_file_path,
        image_dir=image_dir,
        output_dir=output_dir,
        mode=mode,
        # vis_text=False
    )
