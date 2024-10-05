import json

# Function to recursively print the structure of a JSON object
def print_json_structure(data, indent=0):
    print(data)
    # If the data is a dictionary, iterate through its keys
    if isinstance(data, dict):
        for key in data:
            print(" " * indent + str(key))
            # Recursively print nested dictionaries or lists
            print_json_structure(data[key], indent + 4)
    # If the data is a list, iterate through its items
    elif isinstance(data, list) and len(data) > 0:
        print(" " * indent + "[List with " + str(len(data)) + " items]")
        # Recursively print the first item in the list (assuming homogeneous lists)
        print_json_structure(data[0], indent + 4)
        

# Print the top-level keys
# print("Top-level keys:", data.keys())
# dict_keys(['images', 'annotations', 'categories'])

# print(data["images"][0].keys())
# dict_keys(['file_name', 'width', 'height', 'id', 'source_dataset'])

# print(data["annotations"][0].keys())
# dict_keys(['image_id', 'num_keypoints', 'keypoints', 'id', 'category_id', 'area', 'bbox', 'iscrowd'])

# print(data["categories"][0].keys())
# ['name', 'id', 'supercategory', 'keypoints'])

# print(data["categories"][0])
# print(len(data["categories"]))

# Assuming the category information is stored under the "categories" key
# Adjust according to the actual structure of your file
# if "categories" in data:
#     categories = data["categories"]
#     # Extract category names
#     category_names = [category["name"] for category in categories]
#     print("Categories in the file:", category_names)
# else:
#     print("'categories' key not found, check the JSON file structure.")

if __name__ == "__main__":
            
    # Define the path to the JSON file
    # json_file_path = "/home/ti/projects/PrimatePose/ti_data/test.json"
    # json_file_path = "/home/ti"
    json_file_path = "/home/ti/projects/PrimatePose/ti_data/primate_val_1.1.json"
    # json_file_path = "/home/ti/projects/PrimatePose/data/datasets/final_datasets/v7/annotations/pfm_train_apr15.json"    
    # json_file_path = "/mediaPFM/data/datasets/final_datasets/v7/annotations/pfm_test_apr15.json"

    # Read and parse the JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    print_json_structure(data)