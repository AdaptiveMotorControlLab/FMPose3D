import json
import os
from typing import Dict, List

def extract_annotations_with_tails(
    json_path: str,
    output_dir: str,
    tail_keypoints: List[str] = ["root_tail", "mid_tail", "mid_end_tail", "end_tail"]
) -> str:
    """
    Extract annotations that have tail keypoints from a JSON dataset.
    
    Args:
        json_path (str): Path to the input JSON file
        output_dir (str): Directory to save the output JSON file
        tail_keypoints (List[str]): List of tail keypoint names to check
        
    Returns:
        str: Path to the created output JSON file
    """
    # Create a list to store output messages
    output_messages = []
    
    # Load JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get keypoint names from categories
    all_keypoints = data['categories'][0]['keypoints']
    
    # Find indices of tail keypoints
    tail_indices = []
    for tail_kp in tail_keypoints:
        try:
            idx = all_keypoints.index(tail_kp)
            tail_indices.append(idx)
        except ValueError:
            msg = f"Warning: Keypoint '{tail_kp}' not found in dataset"
            print(msg)
            output_messages.append(msg)
    
    msg = f"Tail keypoint indices: {tail_indices}"
    print(msg)
    output_messages.append(msg)
    
    # Create new dataset for annotations with tails
    tail_data = {
        'images': [],
        'annotations': [],
        'categories': data['categories']
    }
    
    # Keep track of images that have tail annotations
    images_with_tails = set()
    
    # Extract annotations with tails
    for ann in data['annotations']:
        keypoints = ann['keypoints']
        has_tail = False
        
        # Check if any tail keypoint exists
        for idx in tail_indices:
            # Each keypoint has 3 values (x, y, visibility)
            kp_idx = idx * 3
            # print(f"keypoint index: {kp_idx}")
            # If either x or y is non-zero, the keypoint exists
            # or the visibility is not -1, we consider the keypoint exists
            # print(ann["id"], idx, keypoints[kp_idx], keypoints[kp_idx + 1], keypoints[kp_idx + 2])
            # if keypoints[kp_idx] != 0 or keypoints[kp_idx + 1] != 0 or keypoints[kp_idx + 2] != -1:
            if keypoints[kp_idx + 2] != -1:
                has_tail = True
                # print("tail keypoints found")
                break
            
        if has_tail:
            tail_data['annotations'].append(ann)
            images_with_tails.add(ann['image_id'])
    
    # Extract corresponding images
    for img in data['images']:
        if img['id'] in images_with_tails:
            tail_data['images'].append(img)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filenames
    dataset_name = os.path.splitext(os.path.basename(json_path))[0]
    mode = dataset_name.split('_')[-1]  # Extract mode (train/test) from filename (e.g. mp_test.json)
    output_path = os.path.join(output_dir, f"{dataset_name}_with_tails.json")
    txt_path = os.path.join(output_dir, f"{dataset_name}_with_tails.txt")
    
    # Save extracted dataset
    with open(output_path, 'w') as f:
        json.dump(tail_data, f, indent=2)
    
    # Print and save statistics
    stats = [
        "\nExtraction complete:",
        f"Original dataset: {len(data['images'])} images, {len(data['annotations'])} annotations",
        f"Extracted dataset: {len(tail_data['images'])} images, {len(tail_data['annotations'])} annotations with tails"
    ]
    
    for msg in stats:
        print(msg)
        output_messages.append(msg)
    
    # Save all output messages to text file
    with open(txt_path, 'w') as f:
        f.write('\n'.join(output_messages))
        f.write(f"\n\nJSON file saved to: {output_path}")
    
    return output_path

def test_extraction():
    """Test the tail annotation extraction function"""
    # Input JSON file
    json_path = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_test_datasets/ak_test.json"   
    # Output directory
    output_dir = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/samples/with_tail"
    
    # Extract annotations with tails
    output_path = extract_annotations_with_tails(
        json_path=json_path,
        output_dir=output_dir
    )
    
    print(f"\nFiles saved to: {output_path}")

if __name__ == "__main__":
    test_extraction()
