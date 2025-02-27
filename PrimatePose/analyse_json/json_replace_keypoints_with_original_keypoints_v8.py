import json
import os
from pathlib import Path

def replace_keypoints_with_original(json_path: str) -> None:
    """Replace keypoints with original keypoints in a V8 format JSON file.
    
    Args:
        json_path (str): Path to the V8 format JSON file
        
    The function will:
    1. Load the V8 format JSON file
    2. Replace keypoints with original keypoints for each annotation
    3. Save the modified data to a new JSON file with '_original_keypoints' suffix
    """
    
    # Load input JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Replace keypoints with original keypoints for each annotation
        for annotation in data['annotations']:
            if 'keypoints_orig' in annotation:
                annotation['keypoints'] = annotation['keypoints_orig']
            else:
                print(f"Warning: No original keypoints found for annotation {annotation['id']}")

    # Generate output filename
    input_path = Path(json_path)
    output_filename = f"{input_path.stem}_ori_keypoints{input_path.suffix}"
    output_path = input_path.parent / output_filename

    # Save modified data to new JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Successfully saved modified data to {output_path}")

if __name__ == "__main__":
    # Example usage
    json_path = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/data_v8.1/splitted_train_datasets/riken_train.json"
    replace_keypoints_with_original(json_path)
