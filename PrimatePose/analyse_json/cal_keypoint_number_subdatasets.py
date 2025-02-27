import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def plot_keypoint_distribution(keypoint_counts, keypoint_names, total_annotations, output_dir, dataset_name):
    """
    Plot histogram of keypoint distributions
    """
    plt.figure(figsize=(15, 8))
    
    # Create the bar plot
    counts = [keypoint_counts[name] for name in keypoint_names]
    x = np.arange(len(keypoint_names))
    
    plt.bar(x, counts)
    
    # Customize the plot
    plt.xticks(x, keypoint_names, rotation=45, ha='right')
    plt.xlabel('Keypoint Name')
    plt.ylabel('Count')
    plt.title(f'Keypoint Distribution for {dataset_name}\nTotal Annotations: {total_annotations}')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f"{dataset_name}_keypoint_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_keypoints(json_file_path):
    # Load JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Get keypoint names from categories
    keypoint_names = data['categories'][0]['keypoints']
    num_keypoints = len(keypoint_names)
    
    # Initialize counters for each keypoint
    keypoint_counts = defaultdict(int)
    # total_annotations = len(data['annotations'])
    total_annotations = 0
    # Analyze each annotation
    for annotation in data['annotations']:
        keypoints = annotation['keypoints']
        
        # Process keypoints in groups of 3 (x, y, visibility)
        count_label = 0
        for i in range(0, len(keypoints), 3):
            visibility = keypoints[i + 2]
            if visibility > 0: # Count only visible keypoints
                keypoint_name = keypoint_names[i // 3]
                keypoint_counts[keypoint_name] += 1
                count_label += 1
            else:
                keypoint_name = keypoint_names[i // 3]
                keypoint_counts[keypoint_name] += 0
        if count_label > 0:
            total_annotations += 1
        
    return keypoint_counts, total_annotations, keypoint_names

def cal_keypoint_number_folder_level():

    # Define input and output directories
    input_dir = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_train_datasets"
    output_dir = "/home/ti_wang/Ti_workspace/PrimatePose/analyse_json/keypint_number_subdatasets"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each JSON file in the input directory
    for json_file in os.listdir(input_dir):
        if not json_file.endswith('.json'):
            continue
            
        json_file_path = os.path.join(input_dir, json_file)
        dataset_name = json_file.split('.')[0]  # Extract dataset name from filename
        
        # Analyze keypoints
        keypoint_counts, total_annotations, keypoint_names = analyze_keypoints(json_file_path)
        
        # Plot the distribution
        plot_keypoint_distribution(keypoint_counts, keypoint_names, total_annotations, output_dir, dataset_name)
        
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
    input_file = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/pfm_train_v8.json"
    output_dir = "/home/ti_wang/Ti_workspace/PrimatePose/analyse_json/keypint_number_subdatasets/pfm_train_v8"
    
    os.makedirs(output_dir, exist_ok=True)
    
    json_file = input_file
    # Process each JSON file in the input directory            
    json_file_path = os.path.join(input_file)
    dataset_name = "pfm_train_v8"
    print(dataset_name)
    
    # Analyze keypoints
    keypoint_counts, total_annotations, keypoint_names = analyze_keypoints(json_file_path)
    
    # Plot the distribution
    plot_keypoint_distribution(keypoint_counts, keypoint_names, total_annotations, output_dir, dataset_name)
    
    # Prepare output file
    output_file = os.path.join(output_dir, f"{dataset_name}_keypoints_number.txt")
    print(output_file)
    
    # Write results to file
    with open(output_file, 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total number of annotations: {total_annotations}\n")
        f.write("\nKeypoint statistics:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Keypoint Name':<30} {'Count':>8}\n")
        f.write("-" * 40 + "\n")
        
        for keypoint_name in keypoint_names:
            count = keypoint_counts[keypoint_name]
            f.write(f"{keypoint_name:<30} {count:>8}\n")

if __name__ == "__main__":
    # cal_keypoint_number_folder_level()
    cal_keypoint_number_file_level()