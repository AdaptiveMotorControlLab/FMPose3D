import pandas as pd
import json

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

if __name__ == "__main__":
    
    json_path = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_test_datasets/mbw_test.json"
    analyse_json_structure(json_path)
