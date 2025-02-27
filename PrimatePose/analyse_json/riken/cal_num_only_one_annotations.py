import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_single_keypoint_distribution(single_keypoint_dist, keypoint_names, title="Distribution of Single Keypoint Annotations", filename="single_keypoint_distribution.png"):
    """
    Create a pie chart for the distribution of single keypoint annotations.
    Group small percentages into 'Others' category for better visualization.
    
    Args:
        single_keypoint_dist: Dictionary with keypoint indices as keys and counts as values
        keypoint_names: Dictionary mapping keypoint indices to their names
        title: Title for the plot
        filename: Name of the file to save the plot
    """
    # Sort the distribution by count
    sorted_dist = sorted(single_keypoint_dist.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate total for percentages
    total = sum(x[1] for x in sorted_dist)
    
    # Separate data into main slices and 'Others'
    threshold = 5.0  # percentage threshold for grouping into 'Others'
    main_data = []
    others_count = 0
    others_details = []
    
    for idx, count in sorted_dist:
        percentage = (count/total) * 100
        if percentage >= threshold:
            main_data.append((idx, count, percentage))
        else:
            others_count += count
            others_details.append((idx, count, percentage))
    
    # Prepare final data for plotting
    counts = [x[1] for x in main_data]
    names = [keypoint_names.get(x[0], f"Keypoint {x[0]}") for x in main_data]
    percentages = [x[2] for x in main_data]
    
    # Add 'Others' category if it exists
    if others_count > 0:
        counts.append(others_count)
        others_percentage = (others_count/total) * 100
        names.append("Others")
        percentages.append(others_percentage)
    
    # Create figure with adjusted size for better proportions
    plt.figure(figsize=(8, 8))
    
    # Create subplot to control pie chart size
    plt.subplot(121)  # 1 row, 2 cols, first position
    
    # Create pie chart with improved label formatting
    patches, texts, autotexts = plt.pie(counts, 
                                      labels=[f"{name}" for name in names],
                                      autopct='%1.1f%%',
                                      startangle=90,
                                      radius=1.2)  # Slightly larger pie
    
    # Adjust font sizes
    plt.setp(autotexts, size=10, weight='bold')
    plt.setp(texts, size=10)
    
    # Add title
    plt.title(title, fontsize=14, pad=20)
    
    # Create subplot for legend and details
    plt.subplot(122)  # 1 row, 2 cols, second position
    plt.axis('off')  # Hide axes
    
    # Add a legend with detailed information
    legend_labels = [f"{name}\n({count:,}, {pct:.1f}%)" for name, count, pct in zip(names, counts, percentages)]
    legend = plt.legend(patches, legend_labels, 
                       title="Keypoint Distribution",
                       loc="center left",
                       bbox_to_anchor=(0, 0.5),
                       fontsize=10)
    plt.setp(legend.get_title(), fontsize=12)
    
    # Add 'Others' details in text box if they exist
    if others_count > 0:
        others_text = "Others includes:\n"
        for idx, count, pct in others_details:
            keypoint_name = keypoint_names.get(idx, f"Keypoint {idx}")
            others_text += f"{keypoint_name}: {count} ({pct:.1f}%)\n"
        plt.text(0.1, 0.1, others_text, fontsize=9, transform=plt.gca().transAxes)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot with adjusted padding
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()

def analyze_annotations(json_path):
    """
    Analyze annotations in a JSON file to calculate:
    1. How many annotations have only one keypoint annotated (and their distribution)
    2. How many annotations have only tail keypoint annotated (when multiple keypoints are visible)
    3. How many annotations have only body center keypoint annotated (including single keypoint cases)
    4. How many annotations are completely empty (all vis_labels are -1)
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        A tuple of (one_annotation_count, only_tail_count, only_body_center_count, empty_count, total_annotations, single_keypoint_dist)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    one_annotation_count = 0
    only_tail_count = 0
    only_body_center_count = 0
    empty_count = 0
    total_annotations = 0
    
    # Dictionary to store distribution of single keypoint annotations
    single_keypoint_dist = defaultdict(int)
    
    # Define tail indices for the PFM format
    tail_indices = [33, 34, 36]
    body_center_indices = [16]
    
    # Get keypoint names if available
    keypoint_names = {}
    if 'categories' in data and len(data['categories']) > 0:
        keypoints = data['categories'][0].get('keypoints', [])
        keypoint_names = {i: name for i, name in enumerate(keypoints)}
    
    # Analyze annotations
    for annotation in data.get('annotations', []):
        total_annotations += 1
        keypoints = annotation.get('keypoints', [])
        
        # Count visible keypoints (visibility value > 0)
        visible_keypoints = []
        num_keypoints = len(keypoints) // 3  # Each keypoint has 3 values (x, y, visibility)
        
        for i in range(num_keypoints):
            vis = keypoints[i * 3 + 2]  # The visibility value
            if vis > 0:
                visible_keypoints.append(i)
        
        # Check conditions
        if len(visible_keypoints) == 0:
            empty_count += 1
        elif len(visible_keypoints) == 1:
            one_annotation_count += 1
            # Record which keypoint is annotated
            keypoint_idx = visible_keypoints[0]
            single_keypoint_dist[keypoint_idx] += 1
            # If it's a body center keypoint, count it in only_body_center_count
            if keypoint_idx in body_center_indices:
                only_body_center_count += 1
        # If multiple keypoints are visible and all are tail keypoints
        elif len(visible_keypoints) > 1 and all(kp in tail_indices for kp in visible_keypoints):
            only_tail_count += 1

    
    return (one_annotation_count, only_tail_count, only_body_center_count, empty_count, total_annotations, single_keypoint_dist, keypoint_names)

def main():
    # Create a file to save the output
    output_file = "riken_annotation_analysis.txt"
    
    # Redirect print output to both console and file
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    # Add sys import at the top
    sys.stdout = Logger(output_file)
    
    # Define the root directory for PFM_V8.2 data
    root_dir = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2"
    
    # Define paths for riken dataset files
    riken_files = {
        "riken_train": os.path.join(root_dir, "splitted_train_datasets", "riken_train.json"),
        "riken_val": os.path.join(root_dir, "splitted_val_datasets", "riken_val.json"),
        "riken_test": os.path.join(root_dir, "splitted_test_datasets", "riken_test.json")
    }
    
    # Results dictionary to store statistics for each file
    results = {}
    total_stats = {
        "total_annotations": 0,
        "annotations_with_pose": 0,
        "one_annotation": 0,
        "only_tail": 0,
        "only_body_center": 0,
        "empty": 0,
        "single_keypoint_dist": defaultdict(int)
    }
    
    # Store keypoint names from any of the files (they should be the same)
    keypoint_names = {}
    
    # Analyze each riken file
    for dataset_name, json_file in riken_files.items():
        if os.path.exists(json_file):
            one_count, tail_count, body_center_count, empty_count, total_count, single_dist, kp_names = analyze_annotations(json_file)
            
            # Store keypoint names if we haven't yet
            if not keypoint_names and kp_names:
                keypoint_names = kp_names
            
            ann_with_pose = total_count - empty_count
            results[dataset_name] = {
                "file_path": json_file,
                "total_annotations": total_count,
                "annotations_with_pose": ann_with_pose,
                "one_annotation": one_count,
                "only_tail": tail_count,
                "only_body_center": body_center_count,
                "empty": empty_count,
                "one_annotation_percentage": round((one_count / ann_with_pose) * 100, 2) if ann_with_pose > 0 else 0,
                "only_tail_percentage": round((tail_count / ann_with_pose) * 100, 2) if ann_with_pose > 0 else 0,
                "only_body_center_percentage": round((body_center_count / ann_with_pose) * 100, 2) if ann_with_pose > 0 else 0,
                "empty_percentage": round((empty_count / ann_with_pose) * 100, 2) if ann_with_pose > 0 else 0,
                "single_keypoint_dist": single_dist
            }
            
            # Update totals
            total_stats["total_annotations"] += total_count
            total_stats["annotations_with_pose"] += ann_with_pose
            total_stats["one_annotation"] += one_count
            total_stats["only_tail"] += tail_count
            total_stats["only_body_center"] += body_center_count
            total_stats["empty"] += empty_count
            
            # Update total single keypoint distribution
            for k, v in single_dist.items():
                total_stats["single_keypoint_dist"][k] += v
        else:
            print(f"File not found: {json_file}")
    
    # Calculate overall percentages
    if total_stats["total_annotations"] > 0:
        total_ann = total_stats["total_annotations"]
        total_ann_with_pose = total_stats["annotations_with_pose"]
        total_stats["one_annotation_percentage"] = round((total_stats["one_annotation"] / total_ann_with_pose) * 100, 2)
        total_stats["only_tail_percentage"] = round((total_stats["only_tail"] / total_ann_with_pose) * 100, 2)
        # total_stats["only_body_center_percentage"] = round((total_stats["only_body_center"] / total_ann_with_pose) * 100, 2)
        # total_stats["empty_percentage"] = round((total_stats["empty"] / total_ann) * 100, 2)
    
    # Print results
    print("\n=== Riken Dataset Analysis Results ===\n")
    
    print("=== Individual Splits Statistics ===")
    for dataset, stats in results.items():
        print(f"\n{dataset}:")
        print(f"  Total annotations: {stats['total_annotations']}")
        print(f"  Annotations_with_pose: {stats['total_annotations'] - stats['empty']}")
        print(f"  Annotations with only one keypoint: {stats['one_annotation']} ({stats['one_annotation_percentage']}%)")
        print(f"  Annotations with only tail keypoints (multiple keypoints): {stats['only_tail']} ({stats['only_tail_percentage']}%)")
        # print(f"  Annotations with only body center keypoint(s): {stats['only_body_center']} ({stats['only_body_center_percentage']}%)")
        # print(f"  Empty annotations: {stats['empty']} ({stats['empty_percentage']}%)")
        
        if stats['one_annotation'] > 0:
            print("\n  Distribution of single keypoint annotations:")
            single_dist = stats['single_keypoint_dist']
            # Sort by frequency
            sorted_dist = sorted(single_dist.items(), key=lambda x: x[1], reverse=True)
            for keypoint_idx, count in sorted_dist:
                keypoint_name = keypoint_names.get(keypoint_idx, f"Keypoint {keypoint_idx}")
                percentage = round((count / stats['one_annotation']) * 100, 2)
                print(f"    {keypoint_name}: {count} ({percentage}%)")
            
            # Create plot for this dataset's distribution
            mode = dataset.split('_')[1]  # Extract train/val/test from dataset name
            plot_single_keypoint_distribution(
                single_dist, 
                keypoint_names,
                f"Distribution of Single Keypoint Annotations - Riken {mode.capitalize()} Set",
                f"riken_keypoint_distribution_{mode}.png"
            )
    
    print("\n=== Overall Riken Statistics ===")
    print(f"Total annotations across all riken splits: {total_stats['total_annotations']}")
    print(f"Total annotations with pose: {total_stats['annotations_with_pose']}")
    print(f"Total annotations with only one keypoint: {total_stats['one_annotation']} ({total_stats['one_annotation_percentage']}%)")
    print(f"Total annotations with only tail keypoints (multiple keypoints): {total_stats['only_tail']} ({total_stats['only_tail_percentage']}%)")
    # print(f"Total annotations with only body center keypoint(s): {total_stats['only_body_center']} ({total_stats['only_body_center_percentage']}%)")
    # print(f"Total empty annotations: {total_stats['empty']} ({total_stats['empty_percentage']}%)")
    
    if total_stats['one_annotation'] > 0:
        print("\nOverall Distribution of single keypoint annotations:")
        single_dist = total_stats['single_keypoint_dist']
        # Sort by frequency
        sorted_dist = sorted(single_dist.items(), key=lambda x: x[1], reverse=True)
        for keypoint_idx, count in sorted_dist:
            keypoint_name = keypoint_names.get(keypoint_idx, f"Keypoint {keypoint_idx}")
            percentage = round((count / total_stats['one_annotation']) * 100, 2)
            print(f"  {keypoint_name}: {count} ({percentage}%)")
        
        # Create plot for overall distribution
        plot_single_keypoint_distribution(
            total_stats['single_keypoint_dist'],
            keypoint_names,
            "Distribution of Single Keypoint Annotations - Riken Dataset (All Splits)",
            "riken_keypoint_distribution_all.png"
        )

    # Close the output file
    sys.stdout.log.close()
    # Restore original stdout
    sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    main()
