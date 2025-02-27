import json
from pathlib import Path
from typing import Dict, List, Set

def find_annotations_with_pose(input_json_path: str, output_json_path: str) -> None:
    """
    Find annotations with pose data and save them to a new JSON file while maintaining structure.
    An annotation is considered to have pose if NOT ALL visibility labels are -1.
    
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
    
    # Process annotations in a single pass
    for ann in data['annotations']:
        # Get visibility labels (every 3rd element starting from index 2)
        vis_labels = ann['keypoints'][2::3]
        
        # Check if NOT ALL visibility labels are -1
        if not all(label == -1 for label in vis_labels):
            filtered_annotations.append(ann)
            valid_image_ids.add(ann['image_id'])
    
    # Create output JSON with same structure using dict comprehension for images
    output_data = {
        'images': [img for img in data['images'] if img['id'] in valid_image_ids],
        'annotations': filtered_annotations,
        'categories': data['categories']
    }
    
    # Ensure output directory exists and save
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        print(f"Output saved to {output_json_path}")
if __name__ == '__main__':
    mode = 'test'
    dataset_name = 'chimpact'
    input_path = f'/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_{mode}_datasets/{dataset_name}_{mode}.json'
    output_path = f'/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/samples/{dataset_name}_{mode}_pose.json'
        
    find_annotations_with_pose(input_path, output_path)
