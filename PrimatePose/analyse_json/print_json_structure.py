import json

# Function to recursively print the structure of a JSON object
def print_json_structure(data, indent=0):
    # If the data is a dictionary, iterate through its keys
    if isinstance(data, dict):
        for key, value in data.items():
            print(" " * indent + str(key))
            # Recursively print nested dictionaries or lists
            print_json_structure(value, indent + 4)
    # If the data is a list, print the size and process the first element
    elif isinstance(data, list):
        print(" " * indent + "[List of {} items]".format(len(data)))
        if len(data) > 0:
            print_json_structure(data[0], indent + 4)
    # For any other data type (like int, str, etc.), just print the type
    else:
        print(" " * indent + "(" + str(type(data).__name__) + ")")

if __name__ == "__main__":
    # Define the path to the JSON file
    # json_file_path = "/home/ti/projects/PrimatePose/ti_data/test.json"

    # json_file_path = "/mnt/data/tiwang/primate_data/pfm_test_v8.json" # v8
    # json_file_path = "/mnt/data/tiwang/v7/annotations/pfm_test_apr15.json" # v7
    # json_file_path = "/app/project/split/ak_2_train/results/snapshot-200snapshot-detector-020-test-predictions.json"
    json_file_path = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_v8.0/splitted_test_datasets/oap_test.json"
    
    # Read and parse the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Print the structure of the JSON file
    print("JSON structure:")
    print_json_structure(data)
