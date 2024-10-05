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


# Define the path to the JSON file
# json_file_path = "/home/ti/projects/PrimatePose/ti_data/test.json"

json_file_path = "/mnt/tiwang/v7/annotations/pfm_test_apr15.json"

# json_file_path = "/home/ti/projects/PrimatePose/ti_data/primate_val_1.1.json"

# Read and parse the JSON file
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Print the structure of the JSON file
print("JSON structure:")
print_json_structure(data)