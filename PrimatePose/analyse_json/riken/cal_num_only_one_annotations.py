import json
import os

def analyze_annotations(json_path):
    """
    Analyze annotations in a JSON file to calculate:
    1. How many annotations have only one keypoint annotated
    2. How many annotations have only tail keypoint annotated
    3. How many annotations are completely empty (all vis_labels are -1)
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        A tuple of (one_annotation_count, only_tail_count, empty_count)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    one_annotation_count = 0
    only_tail_count = 0
    empty_count = 0
    total_annotations = 0
    
    # Get the index of tail keypoint
    tail_index = None
    if 'categories' in data and len(data['categories']) > 0:
        keypoints = data['categories'][0].get('keypoints', [])
        try:
            tail_index = next((i for i, k in enumerate(keypoints) if 'tail' in k.lower()), None)
        except:
            pass
    
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
            if tail_index is not None and visible_keypoints[0] == tail_index:
                only_tail_count += 1
    
    return (one_annotation_count, only_tail_count, empty_count, total_annotations)

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
        "one_annotation": 0,
        "only_tail": 0,
        "empty": 0
    }
    
    # Analyze each riken file
    for dataset_name, json_file in riken_files.items():
        if os.path.exists(json_file):
            one_count, tail_count, empty_count, total_count = analyze_annotations(json_file)
            
            results[dataset_name] = {
                "file_path": json_file,
                "total_annotations": total_count,
                "one_annotation": one_count,
                "only_tail": tail_count,
                "empty": empty_count,
                "one_annotation_percentage": round((one_count / total_count) * 100, 2) if total_count > 0 else 0,
                "only_tail_percentage": round((tail_count / total_count) * 100, 2) if total_count > 0 else 0,
                "empty_percentage": round((empty_count / total_count) * 100, 2) if total_count > 0 else 0
            }
            
            # Update totals
            total_stats["total_annotations"] += total_count
            total_stats["one_annotation"] += one_count
            total_stats["only_tail"] += tail_count
            total_stats["empty"] += empty_count
        else:
            print(f"File not found: {json_file}")
    
    # Calculate overall percentages
    if total_stats["total_annotations"] > 0:
        total_ann = total_stats["total_annotations"]
        total_stats["one_annotation_percentage"] = round((total_stats["one_annotation"] / total_ann) * 100, 2)
        total_stats["only_tail_percentage"] = round((total_stats["only_tail"] / total_ann) * 100, 2)
        total_stats["empty_percentage"] = round((total_stats["empty"] / total_ann) * 100, 2)
    
    # Print results
    print("\n=== Riken Dataset Analysis Results ===\n")
    
    print("=== Individual Splits Statistics ===")
    for dataset, stats in results.items():
        print(f"\n{dataset}:")
        print(f"  Total annotations: {stats['total_annotations']}")
        print(f"  Annotations with only one keypoint: {stats['one_annotation']} ({stats['one_annotation_percentage']}%)")
        print(f"  Annotations with only tail keypoint: {stats['only_tail']} ({stats['only_tail_percentage']}%)")
        print(f"  Empty annotations: {stats['empty']} ({stats['empty_percentage']}%)")
    
    print("\n=== Overall Riken Statistics ===")
    print(f"Total annotations across all riken splits: {total_stats['total_annotations']}")
    print(f"Total annotations with only one keypoint: {total_stats['one_annotation']} ({total_stats['one_annotation_percentage']}%)")
    print(f"Total annotations with only tail keypoint: {total_stats['only_tail']} ({total_stats['only_tail_percentage']}%)")
    print(f"Total empty annotations: {total_stats['empty']} ({total_stats['empty_percentage']}%)")

if __name__ == "__main__":
    main()

