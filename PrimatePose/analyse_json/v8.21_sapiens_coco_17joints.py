"""
this file is used to covert pfm v8.2 format to coco format with 17 keypoints, 
then I can use the coco format to train the sapiens model
"""

import json
import os
from pathlib import Path
import argparse

def convert_to_coco_format(input_file, output_file):
    """
    Convert PFM V8.2 format to COCO format with 17 keypoints.
    
    Args:
        input_file: Path to the input JSON file
        output_file: Path to save the output JSON file
    """
    # PFM to COCO keypoint mapping
    pfm_to_coco = [
        {"id": 0, "pfm_idx": 4, "keypoint_name": "nose"},
        {"id": 1, "pfm_idx": 2, "keypoint_name": "left_eye"},
        {"id": 2, "pfm_idx": 3, "keypoint_name": "right_eye"},
        {"id": 3, "pfm_idx": 5, "keypoint_name": "left_ear"},
        {"id": 4, "pfm_idx": 6, "keypoint_name": "right_ear"},
        {"id": 5, "pfm_idx": 12, "keypoint_name": "left_shoulder"},
        {"id": 6, "pfm_idx": 13, "keypoint_name": "right_shoulder"},
        {"id": 7, "pfm_idx": 18, "keypoint_name": "left_elbow"},
        {"id": 8, "pfm_idx": 19, "keypoint_name": "right_elbow"},
        {"id": 9, "pfm_idx": 20, "keypoint_name": "left_wrist"},
        {"id": 10, "pfm_idx": 21, "keypoint_name": "right_wrist"},
        {"id": 11, "pfm_idx": 24, "keypoint_name": "left_hip"},
        {"id": 12, "pfm_idx": 25, "keypoint_name": "right_hip"},
        {"id": 13, "pfm_idx": 27, "keypoint_name": "left_knee"},
        {"id": 14, "pfm_idx": 28, "keypoint_name": "right_knee"},
        {"id": 15, "pfm_idx": 29, "keypoint_name": "left_ankle"},
        {"id": 16, "pfm_idx": 30, "keypoint_name": "right_ankle"}
    ]
    
    # Get indices of PFM keypoints to extract
    pfm_indices = [item["pfm_idx"] for item in pfm_to_coco]
    coco_keypoint_names = [item["keypoint_name"] for item in pfm_to_coco]
    
    print(f"Loading data from {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Create a new dataset with the same structure
    new_data = {
        "images": data["images"],
        "annotations": [],
        "categories": []
    }
    
    # Update categories with only the 17 COCO keypoints
    for category in data["categories"]:
        new_category = category.copy()
        new_category["keypoints"] = coco_keypoint_names
        new_data["categories"].append(new_category)
    
    # Process annotations to extract only the 17 keypoints
    print("Processing annotations...")
    for i, annotation in enumerate(data["annotations"]):
        if (i + 1) % 10000 == 0:
            print(f"Processed {i+1}/{len(data['annotations'])} annotations")
            
        new_annotation = annotation.copy()
        old_keypoints = annotation["keypoints"]
        
        # Each keypoint has 3 values (x, y, v)
        new_keypoints = []
        
        # Extract only the specified keypoints
        for pfm_idx in pfm_indices:
            keypoint_start_idx = pfm_idx * 3
            new_keypoints.extend(old_keypoints[keypoint_start_idx:keypoint_start_idx+3])
        
        # Count visible keypoints
        num_keypoints = 0
        for i in range(0, len(new_keypoints), 3):
            visibility = new_keypoints[i+2]
            if visibility > 0:  # Count only visible keypoints
                num_keypoints += 1
        
        new_annotation["keypoints"] = new_keypoints
        new_annotation["num_keypoints"] = num_keypoints
        new_data["annotations"].append(new_annotation)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the new dataset
    print(f"Saving COCO format data to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=4)
    
    print(f"Conversion complete. Original dataset had {len(data['annotations'])} annotations with {len(data['categories'][0]['keypoints'])} keypoints.")
    print(f"New dataset has {len(new_data['annotations'])} annotations with {len(new_data['categories'][0]['keypoints'])} keypoints.")

def main():
    parser = argparse.ArgumentParser(description='Convert PFM V8.2 to COCO format with 17 keypoints')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test', 'val'], 
                        help='Dataset mode (train, test, or val)')
    args = parser.parse_args()
   
    args.mode = "train"
    # Define input and output paths
    args.species = "oms"
    base_dir = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2"
    input_file = f"{base_dir}/splitted_{args.mode}_datasets/{args.species}_{args.mode}.json"
    output_dir = f"{base_dir}/8.21_sapiens"
    output_file = f"{output_dir}/{args.species}_{args.mode}_pose_v8_21.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert the dataset
    convert_to_coco_format(input_file, output_file)

if __name__ == "__main__":
    main()
