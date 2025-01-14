import json
import os
import random
from typing import Dict, List, Tuple

def split_json_dataset(
    json_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[str, str]:
    """
    Split a PFM v8 format JSON file into train and test sets.
    
    Args:
        json_path (str): Path to the input JSON file
        output_dir (str): Directory to save the split JSON files
        train_ratio (float): Ratio of data to use for training (default: 0.8)
        random_seed (int): Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple[str, str]: Paths to the created train and test JSON files
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Load JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get all image IDs
    image_ids = [img['id'] for img in data['images']]
    
    # Randomly shuffle image IDs
    random.shuffle(image_ids)
    
    # Split image IDs into train and test sets
    split_idx = int(len(image_ids) * train_ratio)
    train_image_ids = set(image_ids[:split_idx])
    test_image_ids = set(image_ids[split_idx:])
    
    # Create train and test dictionaries
    train_data = {
        'images': [],
        'annotations': [],
        'categories': data['categories']  # Categories remain the same
    }
    
    test_data = {
        'images': [],
        'annotations': [],
        'categories': data['categories']  # Categories remain the same
    }
    
    # Split images
    for img in data['images']:
        if img['id'] in train_image_ids:
            train_data['images'].append(img)
        else:
            test_data['images'].append(img)
    
    # Split annotations
    for ann in data['annotations']:
        if ann['image_id'] in train_image_ids:
            train_data['annotations'].append(ann)
        else:
            test_data['annotations'].append(ann)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filenames
    dataset_name = os.path.splitext(os.path.basename(json_path))[0]
    train_path = os.path.join(output_dir, f"{dataset_name}_train.json")
    test_path = os.path.join(output_dir, f"{dataset_name}_test.json")
    
    # Save split datasets
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # Print statistics
    print(f"\nDataset split complete:")
    print(f"Original dataset: {len(data['images'])} images, {len(data['annotations'])} annotations")
    print(f"Train dataset: {len(train_data['images'])} images, {len(train_data['annotations'])} annotations")
    print(f"Test dataset: {len(test_data['images'])} images, {len(test_data['annotations'])} annotations")
    
    return train_path, test_path

def test_split():
    """Test the dataset splitting function"""
    # Input JSON file
    json_path = "/path/to/your/input.json"
    
    # Output directory
    output_dir = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/split_for_original_dataset_without_testset"
    
    # Split the dataset
    train_path, test_path = split_json_dataset(
        json_path=json_path,
        output_dir=output_dir,
        train_ratio=0.8,
        random_seed=42
    )
    
    print(f"\nFiles saved to:")
    print(f"Train: {train_path}")
    print(f"Test: {test_path}")

if __name__ == "__main__":
    test_split()
