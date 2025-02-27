import json
from pathlib import Path
from typing import Dict, List, Set
import os

def remove_single_keypoint_annotations(input_json_path: str, output_json_path: str) -> None:
    """
    Remove annotations that have only one keypoint and save to a new JSON file.
    An annotation is considered to have only one keypoint if only one visibility label is > 0.
    
    Args:
        input_json_path: Path to input JSON file
        output_json_path: Path to save filtered JSON file
    """
    # Read input JSON
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Pre-allocate data structures
    valid_image_ids: Set[int] = set()
    filtered_annotations: List[Dict] = []
    
    # Track statistics
    total_annotations = 0
    removed_annotations = 0
    empty_annotations = 0
    
    # Process annotations in a single pass
    for ann in data['annotations']:
        total_annotations += 1
        # Get visibility labels (every 3rd element starting from index 2)
        vis_labels = ann['keypoints'][2::3]
        
        # Count visible keypoints (visibility > 0)
        visible_count = sum(1 for label in vis_labels if label > 0)
        
        # Keep annotations with more than 1 visible keypoint
        if visible_count > 1:
            filtered_annotations.append(ann)
            valid_image_ids.add(ann['image_id'])
        else:  # visible_count == 1
            removed_annotations += 1
    
    # Create output JSON with same structure
    output_data = {
        'images': [img for img in data['images'] if img['id'] in valid_image_ids],
        'annotations': filtered_annotations,
        'categories': data['categories']
    }
    
    # Ensure output directory exists
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save filtered data
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    # Print statistics
    print(f"\nProcessing {input_json_path}:")
    print(f"Total annotations: {total_annotations}")
    print(f"Empty annotations (no visible keypoints): {empty_annotations}")
    print(f"Removed annotations (with one keypoint): {removed_annotations}")
    print(f"Remaining annotations: {len(filtered_annotations)}")
    print(f"Output saved to: {output_json_path}\n")

def main():
    # Define paths
    root_dir = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2"
    dataset_name = "riken"
    
    # Define input and output paths
    input_paths = {
        "train": os.path.join(root_dir, "samples", f"{dataset_name}_train_pose.json"),
        "test": os.path.join(root_dir, "samples", f"{dataset_name}_test_pose.json")
    }
    
    # Process train and test splits
    for split in ["train", "test"]:
        input_path = input_paths[split]
        output_path = os.path.join(root_dir, "samples", f"{dataset_name}_{split}_pose_no_single.json")
        
        if os.path.exists(input_path):
            remove_single_keypoint_annotations(input_path, output_path)
        else:
            print(f"Warning: Input file not found: {input_path}")

if __name__ == '__main__':
    main()