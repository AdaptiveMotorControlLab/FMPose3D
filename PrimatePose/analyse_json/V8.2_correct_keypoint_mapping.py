import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any
import logging
import copy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("v8.2_keypoint_correction.log"),
        logging.StreamHandler()
    ]
)

# Define the problematic datasets and their correct keypoint mappings
PROBLEMATIC_DATASETS = {
    # Example of a dataset with actual mapping changes (based on the oap example)
    "oap": {
        "num_keypoints": 17,
        "V8.0_keypoint_mapping": [
            -1, 3, 1, 2, 0, -1, -1, 
            -1, -1, -1, -1, 4, 5, 8, 
            -1, -1, -1, -1, 6, 9, 7, 
            10, -1, -1, -1, -1, 11, 12, 14, 
            13, 15, -1, -1, -1, -1, -1, -1
        ],
        "V8.2_keypoint_mapping": [ 
            -1, 3, 1, 2, 0, -1, -1, 
            -1, -1, -1, -1, 4, 5, 8, 
            -1, -1, -1, -1, 6, 9, -1, -1,
            7, 10, -1, -1, 11, 12, 14, 
            13, 15, -1, -1, -1, -1, -1, -1
        ]
    }
    # Add more problematic datasets here as needed
}

def create_output_directory(base_path: str) -> str:
    """Create the output directory for V8.2 data"""
    output_dir = os.path.join(base_path, "PFM_V8.2")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for train, test, val
    for subset in ["splitted_train_datasets", "splitted_test_datasets", "splitted_val_datasets"]:
        os.makedirs(os.path.join(output_dir, subset), exist_ok=True)
    
    logging.info(f"Created output directory structure at {output_dir}")
    return output_dir

def validate_keypoints_structure(json_data: Dict[str, Any], dataset_name: str) -> bool:
    """Validate the keypoints structure in the JSON data"""
    # Check if annotations exist
    if 'annotations' not in json_data or not json_data['annotations']:
        logging.warning(f"No annotations found in {dataset_name} dataset")
        return False
    
    # Check keypoints structure in the first annotation
    first_annotation = json_data['annotations'][0]
    if 'keypoints' not in first_annotation:
        logging.warning(f"No keypoints found in {dataset_name} dataset annotations")
        return False
    
    # Check if keypoints have the expected structure (list of values)
    keypoints = first_annotation['keypoints']
    if not isinstance(keypoints, list):
        logging.warning(f"Keypoints in {dataset_name} dataset are not in list format")
        return False
    
    # In V8.0, each keypoint has 3 values (x, y, visibility)
    # So the total length should be a multiple of 3
    if len(keypoints) % 3 != 0:
        logging.warning(f"Keypoints in {dataset_name} dataset do not have the expected structure (multiple of 3)")
        return False
    
    return True

def transform_keypoints(keypoints: List[int], v80_mapping: List[int], v82_mapping: List[int]) -> List[int]:
    """
    Transform keypoints from V8.0 format to V8.2 format using the mapping arrays
    
    Args:
        keypoints: The original keypoints array (flat array with x,y,v values)
        v80_mapping: The V8.0 keypoint mapping array
        v82_mapping: The V8.2 keypoint mapping array
        
    Returns:
        The transformed keypoints array
    """
    # Create a new keypoints array with the same length as the original
    new_keypoints = copy.deepcopy(keypoints)
    
    # Get the number of keypoints (each keypoint has 3 values: x, y, visibility)
    num_keypoints = len(keypoints) // 3
    
    # Check if the mapping arrays have the expected length
    if len(v80_mapping) != len(v82_mapping):
        logging.warning(f"Mapping arrays have different lengths: {len(v80_mapping)} vs {len(v82_mapping)}")
        return keypoints  # Return original keypoints if mapping arrays are invalid
    
    # Create a mapping from V8.0 positions to V8.2 positions
    position_mapping = {}
    for i, (v80_pos, v82_pos) in enumerate(zip(v80_mapping, v82_mapping)):
        if v80_pos != -1 and v82_pos != -1 and v80_pos != v82_pos:
            position_mapping[v80_pos] = v82_pos
    
    # If there are no changes in the mapping, return the original keypoints
    if not position_mapping:
        return keypoints
    
    # Apply the mapping to transform keypoints
    for v80_pos, v82_pos in position_mapping.items():
        if v80_pos < num_keypoints and v82_pos < num_keypoints:
            # Copy the x, y, visibility values from the old position to the new position
            new_keypoints[v82_pos*3] = keypoints[v80_pos*3]       # x
            new_keypoints[v82_pos*3+1] = keypoints[v80_pos*3+1]   # y
            new_keypoints[v82_pos*3+2] = keypoints[v80_pos*3+2]   # visibility
            
            # Set the old position to invalid if it's different from the new position
            if v80_pos != v82_pos:
                new_keypoints[v80_pos*3] = 0       # x
                new_keypoints[v80_pos*3+1] = 0     # y
                new_keypoints[v80_pos*3+2] = -1    # visibility (invalid)
    
    return new_keypoints

def correct_keypoint_mapping(json_data: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """Correct the keypoint mapping for a specific dataset"""
    if dataset_name.lower() not in PROBLEMATIC_DATASETS:
        return json_data  # No correction needed
    
    # Validate keypoints structure
    if not validate_keypoints_structure(json_data, dataset_name):
        logging.error(f"Invalid keypoints structure in {dataset_name} dataset. Skipping correction.")
        return json_data
    
    # Get the correct mapping
    correct_mapping = PROBLEMATIC_DATASETS[dataset_name.lower()]
    
    # Check if we have mapping arrays for this dataset
    if 'V8.0_keypoint_mapping' in correct_mapping and 'V8.2_keypoint_mapping' in correct_mapping:
        v80_mapping = correct_mapping['V8.0_keypoint_mapping']
        v82_mapping = correct_mapping['V8.2_keypoint_mapping']
        
        # Transform keypoints in each annotation
        annotations_updated = 0
        keypoints_transformed = 0
        
        for annotation in json_data.get('annotations', []):
            # Transform keypoints if needed
            if 'keypoints' in annotation:
                old_keypoints = annotation['keypoints']
                new_keypoints = transform_keypoints(old_keypoints, v80_mapping, v82_mapping)
                
                # Only update if there was an actual change
                if new_keypoints != old_keypoints:
                    annotation['keypoints'] = new_keypoints
                    keypoints_transformed += 1
            
            annotations_updated += 1
        
        logging.info(f"Processed {annotations_updated} annotations for {dataset_name} dataset")
        logging.info(f"Transformed keypoints in {keypoints_transformed} annotations")
    else:
        logging.warning(f"No mapping arrays found for {dataset_name} dataset. Skipping keypoint transformation.")
    
    return json_data

def copy_non_problematic_dataset(input_file_path: str, output_file_path: str) -> None:
    """Copy a dataset file that doesn't need correction"""
    try:
        shutil.copy2(input_file_path, output_file_path)
        logging.info(f"Copied file from {input_file_path} to {output_file_path}")
    except Exception as e:
        logging.error(f"Error copying file {input_file_path}: {str(e)}")

def process_dataset_files(input_base_path: str, output_base_path: str) -> None:
    """Process all dataset files and correct keypoint mappings"""
    # Define the subsets to process
    subsets = ["splitted_train_datasets", "splitted_test_datasets", "splitted_val_datasets"]
    
    # Track statistics
    total_files = 0
    corrected_files = 0
    copied_files = 0
    error_files = 0
    
    for subset in subsets:
        input_dir = os.path.join(input_base_path, subset)
        output_dir = os.path.join(output_base_path, subset)
        
        # Skip if input directory doesn't exist
        if not os.path.exists(input_dir):
            logging.warning(f"Input directory {input_dir} does not exist. Skipping.")
            continue
        
        # Process each JSON file in the directory
        for filename in os.listdir(input_dir):
            if not filename.endswith('.json'):
                continue
            
            total_files += 1
            
            # Extract dataset name from filename (e.g., "riken_train.json" -> "riken")
            dataset_name = filename.split('_')[0]
            
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename)
            
            try:
                # Check if this dataset needs correction
                if dataset_name.lower() in PROBLEMATIC_DATASETS:
                    logging.info(f"Processing {filename} for keypoint mapping correction")
                    
                    # Load the JSON data
                    with open(input_file_path, 'r') as f:
                        json_data = json.load(f)
                    
                    # Correct the keypoint mapping
                    json_data = correct_keypoint_mapping(json_data, dataset_name)
                    
                    # Save the processed data
                    with open(output_file_path, 'w') as f:
                        json.dump(json_data, f, indent=4)
                    
                    logging.info(f"Saved corrected file to {output_file_path}")
                    corrected_files += 1
                else:
                    logging.info(f"No correction needed for {filename}, copying file")
                    copy_non_problematic_dataset(input_file_path, output_file_path)
                    copied_files += 1
            except Exception as e:
                logging.error(f"Error processing file {filename}: {str(e)}")
                error_files += 1
    
    # Log summary statistics
    logging.info("=== Processing Summary ===")
    logging.info(f"Total files processed: {total_files}")
    logging.info(f"Files corrected: {corrected_files}")
    logging.info(f"Files copied without changes: {copied_files}")
    logging.info(f"Files with errors: {error_files}")

def main():
    # Define paths
    base_path = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data"
    input_path = os.path.join(base_path, "data_v8.0")
    
    logging.info("Starting V8.2 keypoint mapping correction")
    logging.info(f"Input path: {input_path}")
    
    # Verify input path exists
    if not os.path.exists(input_path):
        logging.error(f"Input path {input_path} does not exist. Aborting.")
        return
    
    # Create output directory
    output_path = create_output_directory(base_path)
    logging.info(f"Output path: {output_path}")
    
    # Process all dataset files
    process_dataset_files(input_path, output_path)
    
    logging.info(f"V8.2 keypoint mapping correction completed. Results saved to {output_path}")

if __name__ == "__main__":
    main()