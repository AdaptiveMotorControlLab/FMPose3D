import json
import os
from pathlib import Path
from collections import defaultdict

# Define input and output directories
input_dir = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_train_datasets"
output_dir = "/home/ti_wang/Ti_workspace/PrimatePose/analyse_json/keypint_number_subdatasets"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def analyze_keypoints(json_file_path):
    # Load JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Get keypoint names from categories
    keypoint_names = data['categories'][0]['keypoints']
    num_keypoints = len(keypoint_names)
    
    # Initialize counters for each keypoint
    keypoint_counts = defaultdict(int)
    total_annotations = len(data['annotations'])
    
    # Analyze each annotation
    for annotation in data['annotations']:
        keypoints = annotation['keypoints']
        # Process keypoints in groups of 3 (x, y, visibility)
        for i in range(0, len(keypoints), 3):
            visibility = keypoints[i + 2]
            if visibility > 0: # Count only visible keypoints
                keypoint_name = keypoint_names[i // 3]
                keypoint_counts[keypoint_name] += 1
            else:
                keypoint_name = keypoint_names[i // 3]
                keypoint_counts[keypoint_name] += 0
                
    return keypoint_counts, total_annotations, keypoint_names

def cal_keypoint_number_folder_level():
    # Define input and output directories
    input_dir = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_train_datasets"
    output_dir = "/home/ti_wang/Ti_workspace/PrimatePose/analyse_json/keypint_number_subdatasets"
    
    # Process each JSON file in the input directory
    for json_file in os.listdir(input_dir):
        if not json_file.endswith('.json'):
            continue
            
        json_file_path = os.path.join(input_dir, json_file)
        dataset_name = json_file.split('.')[0]  # Extract dataset name from filename
        
        # Analyze keypoints
        keypoint_counts, total_annotations, keypoint_names = analyze_keypoints(json_file_path)
        
        # Prepare output file
        output_file = os.path.join(output_dir, f"{dataset_name}_keypoints_number.txt")
        
        # Write results to file
        with open(output_file, 'w') as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Total number of annotations: {total_annotations}\n")
            f.write("\nKeypoint statistics:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Keypoint Name':<30} {'Count':>8} {'Percentage':>12}\n")
            f.write("-" * 50 + "\n")
            
            for keypoint_name in keypoint_names:
                count = keypoint_counts[keypoint_name]
                # percentage = (count / total_annotations) * 100
                f.write(f"{keypoint_name:<30} {count:>8} %\n") # {percentage:>11.2f}

def cal_keypoint_number_file_level():
    # Define input and output directories
    input_file = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_train_datasets"
    output_dir = "/home/ti_wang/Ti_workspace/PrimatePose/analyse_json/keypint_number_subdatasets"
    
    json_file = input_file
    # Process each JSON file in the input directory            
    json_file_path = os.path.join(input_file)
    dataset_name = json_file.split('.')[0]  # Extract dataset name from filename
    
    # Analyze keypoints
    keypoint_counts, total_annotations, keypoint_names = analyze_keypoints(json_file_path)
    
    # Prepare output file
    output_file = os.path.join(output_dir, f"{dataset_name}_keypoints_number.txt")
    
    # Write results to file
    with open(output_file, 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total number of annotations: {total_annotations}\n")
        f.write("\nKeypoint statistics:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Keypoint Name':<30} {'Count':>8} {'Percentage':>12}\n")
        f.write("-" * 50 + "\n")
        
        for keypoint_name in keypoint_names:
            count = keypoint_counts[keypoint_name]
            # percentage = (count / total_annotations) * 100
            f.write(f"{keypoint_name:<30} {count:>8} %\n") # {percentage:>11.2f}
                
if __name__ == "__main__":
    # cal_keypoint_number_folder_level()
    cal_keypoint_number_file_level()