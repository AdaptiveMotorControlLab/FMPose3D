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

def select_sample_by_image_id(source_json_file, image_id, target_json_file):
    # Load the JSON data from the source file
    with open(source_json_file, 'r') as file:
        data = json.load(file)

    # Filter the images by the specified image_id
    filtered_images = [image for image in data['images'] if image['id'] == image_id]

    # If the image_id does not exist in the dataset, print a warning and exit
    if not filtered_images:
        print(f"No image found with image_id: {image_id}")
        return

    # Filter annotations by the specified image_id
    filtered_annotations = [annotation for annotation in data['annotations'] if annotation['image_id'] == image_id]

    # If there are no annotations for the image_id, print a warning
    if not filtered_annotations:
        print(f"No annotations found for image_id: {image_id}")

    # Construct the target JSON structure with the filtered data
    filtered_data = {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": data["categories"]  # Retain the full categories list
    }

    # Save the filtered data to the target JSON file
    with open(target_json_file, 'w') as target_file:
        json.dump(filtered_data, target_file, indent=2)

    print(f"Filtered data has been saved to {target_json_file}")

if __name__ == "__main__":

    # Print the JSON structure
    # print_json_structure(data)

    nums = 50
    subset = "ap10k_val"
    input_path = f'/home/ti_wang/Ti_workspace/PrimatePose/data/splitted_val_datasets/{subset}.json'
    output_path = f'/home/ti_wang/Ti_workspace/PrimatePose/data/splitted_val_datasets/{subset}_sampled_nums_{nums}.json' 
    sample_json(nums, input_path, output_path)
    
    # image_id = 36681
    # source_json_file = f'/home/ti_wang/Ti_workspace/PrimatePose/data/splitted_val_datasets/{subset}.json'
    # target_json_file = f'/home/ti_wang/Ti_workspace/PrimatePose/data/splitted_val_datasets/{subset}_{image_id}.json'
    # select_sample_by_image_id(source_json_file, image_id, target_json_file)