import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any
import logging
import copy

# Define the problematic datasets and their correct keypoint mappings
PROBLEMATIC_DATASETS = {
    # Example of a dataset with actual mapping changes (based on the oap example)
    "oap": {
        "original_num_keypoints": 16,
        "V8.0_keypoint_mapping": [
            -1, 3, 1, 2, 0, -1, -1, 
            -1, -1, -1, -1, 4, 5, 8, 
            -1, -1, -1, -1, 6, 9, 7, 
            10, -1, -1, -1, -1, 11, 12, 14, 
            -1, -1, 13, 15, -1, -1, -1, -1
        ],
        "V8.2_keypoint_mapping": [ 
            -1, 3, 1, 2, 0, -1, -1, 
            -1, -1, -1, -1, 4, 5, 8, 
            -1, -1, -1, -1, 6, 9, -1, -1,
            7, 10, -1, -1, 11, 12, 14, 
            13, 15, -1, -1, -1, -1, -1, -1
        ]
    },
    "omc": {
        "original_num_keypoints": 17,
        "V8.0_keypoint_mapping": [
            -1, 3, 1, 0, 2, -1, -1, 
            -1, -1, -1, -1, 4, 8, 5, 
            -1, -1, -1, -1, 9, 6, 10, 
            7, -1, -1, -1, -1, -1, 14, 12, 
            15, 13, -1, -1, 11, -1, -1, 16
        ],
        "V8.2_keypoint_mapping": [
            -1, 3, 1, 0, 2, -1, -1, 
            -1, -1, -1, -1, 4, 8, 5, 
            -1, -1, -1, -1, 9, 6, 10, 
            7, -1, -1, -1, -1, 11, 14, 12, 
            15, 13, -1, -1, -1, -1, -1, 16
        ],
    },
    "ap10k": {
        "original_num_keypoints": 17,
        "V8.0_keypoint_mapping": [
            -1, -1, 0, 1, 2, -1, -1, -1, -1, -1, -1,
            3, 5, 8, -1, -1, -1, -1, 6, 9, 7, 10, -1, -1,
            11, 14, 4, 12, 15, 13, 16, -1, -1, -1, -1, -1, -1
        ],
        "V8.2_keypoint_mapping": [
            -1, -1, 0, 1, 2, -1, -1, -1, -1, -1, -1,
            3, 5, 8, -1, -1, -1, -1, 6, 9, -1, -1, 7, 10, 
            11, 14, -1, 12, 15, -1, -1, 13, 16, 4, -1, -1, -1
        ],
    },
    "aptv2": {
        "original_num_keypoints": 17,
        "V8.0_keypoint_mapping": [
            -1, -1, 0, 1, 2, -1, -1, -1, -1, -1, -1,
            3, 5, 8, -1, -1, -1, -1, 6, 9, 7, 10, -1, -1,
            11, 14, 4, 12, 15, 13, 16, -1, -1, -1, -1, -1, -1
        ],
        "V8.2_keypoint_mapping": [
            -1, -1, 0, 1, 2, -1, -1, -1, -1, -1, -1,
            3, 5, 8, -1, -1, -1, -1, 6, 9, -1, -1, 7, 10, 
            11, 14, -1, 12, 15, -1, -1, 13, 16, 4, -1, -1, -1
        ],
    },
    "ak":{
        "original_num_keypoints": 23,
        "V8.0_keypoint_mapping": [
            -1, 0, 1, 2, -1, -1, -1, 3, 6,
            4, 5, -1, 7, 8, -1, 13, -1, -1,
            9, 10, 11, 12, -1, -1, 14, 15,
            -1, 16, 17, 18, 19, -1, -1, 20,
            21, 22, -1
            ],
        "V8.2_keypoint_mapping": [
            -1, 0, 1, 2, -1, -1, -1, 3, 6,
            4, 5, -1, 7, 8, -1, -1, -1, -1,
            9, 10, 11, 12, -1, -1, 14, 15,
            -1, 16, 17, 18, 19, -1, -1, 20,
            21, 22, -1
            ]
    },
    "mbw":{
        "original_num_keypoints": 16,
        "V8.0_keypoint_mapping": [
            -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 2, 5, -1, 8, -1, -1, 3, 6, 4, 7, -1,
            -1, 10, 13, 9, 11, 14, 12, 15, -1, -1, -1, -1, -1, -1
        ],
        "V8.2_keypoint_mapping": [
            -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            1, 2, 5, -1, -1, -1, -1, 3, 6, 4, 7, -1,
            -1, 10, 13, 9, 11, 14, 12, 15, -1, -1, -1, -1, -1, -1
        ]
    },
    "riken": {
        "original_num_keypoints": 21,
        "V8.0_keypoint_mapping": [
            -1, -1, 1, 2, 0, 3, 4, -1, -1, -1, -1,
            -1, 5, 6, -1, -1, 20, -1, 7, 8, 9, 10,
            -1, -1, 11, 12, -1, 13, 14, 15, 16, -1, -1,
            17, 18, -1, 19],
        "V8.2_keypoint_mapping": [
            -1, -1, 1, 2, 0, 3, 4, -1, -1, -1, -1,
            -1, 5, 6, -1, -1, -1, -1, 7, 8, 9, 10,
            -1, -1, 11, 12, -1, 13, 14, 15, 16, -1, -1,
            17, 18, -1, 19],
    },
    "mit": {
        "original_num_keypoints": 17,
        "V8.0_keypoint_mapping": [
            0, 3, -1, -1, -1, 1, 2, -1, -1, -1, -1, -1,
            7, 9, 4, -1, -1, 5, -1, -1, 8, 10, -1, -1,
            11, 13, -1, -1, -1, -1, -1, 12, 14, 6, -1, -1, -1
        ],
        "V8.2_keypoint_mapping": [
            -1, 3, -1, -1, -1, 1, 2, -1, -1, -1, -1, -1,
            7, 9, 4, -1, -1, 5, -1, -1, 8, 10, -1, -1,
            11, 13, -1, -1, -1, -1, -1, 12, 14, 6, -1, -1, -1
        ]
    }    
    # Add more problematic datasets here as needed
}

def transform_keypoints(keypoints: List[int], v80_mapping: List[int], v82_mapping: List[int]) -> List[int]:
    """
    Transform keypoints from V8.0 format to V8.2 format using the mapping arrays
    
    Args:
        keypoints: The original keypoints array (flat array with x,y,v values)
        v80_mapping: The V8.0 keypoint mapping array (PFM index -> original dataset index)
        v82_mapping: The V8.2 keypoint mapping array (PFM index -> original dataset index)
        
    Returns:
        The transformed keypoints array
    """
    # Create a new keypoints array with the same length as the original
    new_keypoints = copy.deepcopy(keypoints)
    
    # Create a new keypoints array for V8.2 with the same shape as keypoints but filled with -1
    keypoint_v82 = [-1] * len(keypoints)
    
    # Check if the mapping arrays have the expected length
    if len(v80_mapping) != len(v82_mapping):
        logging.warning(f"Mapping arrays have different lengths: {len(v80_mapping)} vs {len(v82_mapping)}")
        return keypoints  # Return original keypoints if mapping arrays are invalid
    
    # Log the mapping arrays for debugging
    logging.debug(f"V8.0 mapping: {v80_mapping}")
    logging.debug(f"V8.2 mapping: {v82_mapping}")
    
    orig_to_pfm_mapping_v80 = {}
    
    # Build the mappings
    for pfm_idx, orig_idx in enumerate(v80_mapping):
        if orig_idx != -1:
            orig_to_pfm_mapping_v80[orig_idx] = pfm_idx
            
    for pfm_idx, orig_idx_v82 in enumerate(v82_mapping):
        if orig_idx_v82 != -1:
            # for orig_idx_v82, find the same keypoint in v80_mapping in pfm format
            orig_v82_to_pfm_v80 = orig_to_pfm_mapping_v80[orig_idx_v82]
            
            keypoint_v82[pfm_idx * 3] = new_keypoints[orig_v82_to_pfm_v80 * 3]
            keypoint_v82[pfm_idx * 3 + 1] = new_keypoints[orig_v82_to_pfm_v80 * 3 + 1]
            keypoint_v82[pfm_idx * 3 + 2] = new_keypoints[orig_v82_to_pfm_v80 * 3 + 2]
    
    return keypoint_v82

def correct_keypoint_mapping(json_data: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    
    """Correct the keypoint mapping for a specific dataset"""
    if dataset_name.lower() not in PROBLEMATIC_DATASETS:
        return json_data  # No correction needed
    
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
            logging.info(f"__________________Processing {filename}__________________")        
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
                    
                    # Correct the keypoint mapping, and return the corrected json data
                    json_data = correct_keypoint_mapping(json_data, dataset_name)
                    
                    # Save the processed data
                    with open(output_file_path, 'w') as f:
                        json.dump(json_data, f, indent=4)
                    
                    logging.info(f"Saved corrected file to {output_file_path}")
                    corrected_files += 1
                else:
                    # delete?
                    logging.info(f"No correction needed for {filename}, copying file")
                    # Copy non-problematic dataset directly here instead of calling a function
                    shutil.copy2(input_file_path, output_file_path)
                    # logging.info(f"Copied file from {input_file_path} to {output_file_path}")
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
    input_path = os.path.join(base_path, "PFM_v8.0")
    
    # Create output directory directly in main()
    output_path = os.path.join(base_path, "PFM_V8.2")
    os.makedirs(output_path, exist_ok=True)
    
    # Configure logging to save to the target folder
    log_file_path = os.path.join(output_path, "v8.2_keypoint_correction.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting V8.2 keypoint mapping correction")
    logging.info(f"Input path: {input_path}")
    logging.info(f"Log file path: {log_file_path}")
    
    # Verify input path exists
    if not os.path.exists(input_path):
        logging.error(f"Input path {input_path} does not exist. Aborting.")
        return
    
    # Create subdirectories for train, test, val
    for subset in ["splitted_train_datasets", "splitted_test_datasets", "splitted_val_datasets"]:
        os.makedirs(os.path.join(output_path, subset), exist_ok=True)
    
    logging.info(f"Created output directory structure at {output_path}")
    
    # Process all dataset files
    process_dataset_files(input_path, output_path)
    
    logging.info(f"V8.2 keypoint mapping correction completed. Results saved to {output_path}")

if __name__ == "__main__":
    main()