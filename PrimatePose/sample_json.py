import json
import random

# Specify your JSON file path
json_file_path = '/home/ti_wang/Ti_workspace/PrimatePose/data/splitted_val_datasets/chimpact_val.json'

def print_json_structure(data, indent=0):
    # Check if data is a dictionary
    if isinstance(data, dict):
        for key in data:
            print(' ' * indent + f'Key: "{key}"')
            print_json_structure(data[key], indent + 4)
    # Check if data is a list
    elif isinstance(data, list):
        print(' ' * indent + 'List of items:')
        if len(data) > 0:
            print_json_structure(data[0], indent + 4)  # Print structure of the first item
    else:
        print(' ' * indent + f'Type: {type(data).__name__}')


def sample_json(nums, input_json_file, output_json_file):
        
    # Load the JSON data
    with open(input_json_file, 'r') as f:
        data = json.load(f)

    # Get all images and annotations
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    
    # Randomly sample 100 images
    sampled_images = random.sample(images, nums)

    # Get the IDs of the sampled images
    sampled_image_ids = {img['id'] for img in sampled_images}

    # Filter the annotations that correspond to the sampled images
    sampled_annotations = [ann for ann in annotations if ann['image_id'] in sampled_image_ids]

    # Create the sampled JSON structure
    sampled_data = {
        'images': sampled_images,
        'annotations': sampled_annotations,
        'categories': categories  # Include all categories since they don't change with sampling
    }

    # Save the sampled JSON to a new file
    with open(output_json_file, 'w') as f:
        json.dump(sampled_data, f, indent=4)


if __name__ == "__main__":


    # Print the JSON structure
    # print_json_structure(data)


    nums = 2
    input_path = '/home/ti_wang/Ti_workspace/PrimatePose/data/splitted_val_datasets/chimpact_val.json'
    output_path = f'/home/ti_wang/Ti_workspace/PrimatePose/data/splitted_val_datasets/chimpact_val_sampled_{nums}.json' 
    sample_json(nums, input_path, output_path)