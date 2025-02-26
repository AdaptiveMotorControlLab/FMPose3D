import json
import os
from collections import defaultdict

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
            
            results[dataset_name] = {
                "file_path": json_file,
                "total_annotations": total_count,
                "annotations_with_pose": total_count - empty_count,
                "one_annotation": one_count,
                "only_tail": tail_count,
                "only_body_center": body_center_count,
                "empty": empty_count,
                "one_annotation_percentage": round((one_count / total_count) * 100, 2) if total_count > 0 else 0,
                "only_tail_percentage": round((tail_count / total_count) * 100, 2) if total_count > 0 else 0,
                "only_body_center_percentage": round((body_center_count / total_count) * 100, 2) if total_count > 0 else 0,
                "empty_percentage": round((empty_count / total_count) * 100, 2) if total_count > 0 else 0,
                "single_keypoint_dist": single_dist
            }
            
            # Update totals
            total_stats["total_annotations"] += total_count
            total_stats["annotations_with_pose"] += total_count - empty_count
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
    
    print("\n=== Overall Riken Statistics ===")
    print(f"Total annotations across all riken splits: {total_stats['total_annotations']}")
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

if __name__ == "__main__":
    main()
