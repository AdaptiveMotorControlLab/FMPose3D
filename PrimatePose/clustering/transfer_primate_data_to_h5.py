import json
import pandas as pd
import numpy as np
import os

def analyse_json_structure(json_path):
    """
    Analyze the structure of a JSON file and ensure it's a dictionary.
    
    Args:
        json_path (str): Path to the JSON file
        
    Returns:
        bool: True if data is a dictionary with expected structure, False otherwise
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("\nRoot structure: Dictionary")
    keys = set(data.keys())
    # print(keys)
    
    # Analyze the structure
    for key, value in data.items():
        value_type = type(value).__name__
        if isinstance(value, (list, dict)):
            value_len = len(value)
            print(f"  {key}: {value_type} (length: {value_len})")
            
            # If it's a list and not empty, show the type of first element
            if isinstance(value, list) and value:
                first_elem_type = type(value[0]).__name__
                print(f"    First element type: {first_elem_type}")
                
                # Show structure of first element if it's a dict
                if isinstance(value[0], dict):
                    print(f"    First element keys: {list(value[0].keys())}")
        else:
            print(f"  {key}: {value_type} = {value}")
    
    return True

def process_json_data(json_path):
    """
    Process JSON data and prepare it for H5 conversion
    """
    # 1. Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 2. Extract required data
    # Get bodyparts
    bodyparts = data["categories"][0]["keypoints"]
    
    # Create image ID to name mapping
    image_ids_to_image_name = {}
    for img in data["images"]:
        image_ids_to_image_name[img["id"]] = img["file_name"]
    
    # Create image ID to annotations mapping
    image_to_annotations = {}
    for ann in data["annotations"]:
        if ann["image_id"] not in image_to_annotations:
            image_to_annotations[ann["image_id"]] = []
        image_to_annotations[ann["image_id"]].append(ann)
    
    # Find max number of individuals
    max_individuals = max(len(anns) for anns in image_to_annotations.values())
    individuals = [f"monkey{i+1}" for i in range(max_individuals)]
    print(individuals)
    # Prepare data for DataFrame
    prediction_data = []
    index = []
    
    # Process each image
    for img_id, img_name in image_ids_to_image_name.items():
        index_name = f"{img_id}_{img_name}"
        index.append(index_name)
        
        # Get annotations for this image
        img_annotations = image_to_annotations.get(img_id, [])
        
        # Process keypoints for each individual
        img_data = []
        for i in range(max_individuals):
            if i < len(img_annotations):
                # Get keypoints from annotation and convert to float type
                keypoints = np.array(img_annotations[i]["keypoints"], dtype=np.float32)
                keypoints = keypoints.reshape(-1, 3)  # reshape to (num_keypoints, 3)
                
                # Create mask for non-visible or non-existent keypoints
                non_visible_mask = (keypoints[:, 2] == -1) # visibility is -1
                invalid_mask = non_visible_mask
                
                # Set coordinates to np.nan for invalid keypoints
                # keypoints[invalid_mask, :2] = np.nan
                keypoints[invalid_mask, :2] = 0
                
                # Add only x, y coordinates
                img_data.extend(keypoints[:, :2].flatten())
            else:
                # Pad with np.nan for missing individuals
                num_coords = len(bodyparts) * 2  # x,y for each bodypart
                # img_data.extend([np.nan] * num_coords)
                img_data.extend([0] * num_coords)
                
        prediction_data.append(img_data)
    
    return np.array(prediction_data), index, bodyparts, individuals

def build_dlc_dataframe_columns(scorer, bodyparts, individuals):
    """Build multi-index columns in DLC format"""
    columns = []
    for ind in individuals:
        for bp in bodyparts:
            columns.extend([(scorer, ind, bp, 'x'), 
                          (scorer, ind, bp, 'y')])
    
    return pd.MultiIndex.from_tuples(columns, names=['scorer', 'individuals', 'bodyparts', 'coords'])

def transfer_json_to_h5(json_path, output_dir):
    """
    Transfer data from JSON to H5 format and CSV format
    
    Args:
        json_path (str): Path to input JSON file
        output_dir (str): Directory to save output files
    """
    # Process JSON data
    prediction_data, index, bodyparts, individuals = process_json_data(json_path)
    
    # Set scorer
    scorer = ['ti']
    
    # Build DataFrame
    columns = build_dlc_dataframe_columns(scorer[0], bodyparts, individuals)
    df = pd.DataFrame(prediction_data, index=index, columns=columns)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filenames
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    h5_path = os.path.join(output_dir, f"{base_name}.h5")
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    
    # Save to H5
    df.to_hdf(h5_path, key='data', mode='w')
    print(f"Saved H5 file to: {h5_path}")
    
    # Save to CSV
    df.to_csv(csv_path)
    print(f"Saved CSV file to: {csv_path}")

    return df

def test_function():
    """Test the transfer function"""
    
    file_name = "ap10k"
    mode = "test"
    json_path = f"/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_{mode}_datasets/{file_name}_{mode}.json"
    output_dir = "/home/ti_wang/Ti_workspace/PrimatePose/clustering/data"
    
    print("Converting JSON to H5 and CSV...")
    df = transfer_json_to_h5(json_path, output_dir)
    print(f"\nCreated files with shape: {df.shape}")
    print("\nFirst few rows of the converted data:")
    print(df.head())

if __name__ == "__main__":
    test_function()