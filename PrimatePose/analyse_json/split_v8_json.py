# instruction
# 1. split the v8 json file into train, test, val from V7 format

import json
import os
from collections import defaultdict

def convert_v7_to_v8_and_split_datasets(input_file, output_dir, model):
    """Convert V7 format JSON to V8 and split into separate JSON files by source datasets

    Args:
        input_file (str): Path to the input JSON file in V7 format
        output_dir (str): Output directory for saving split V8 format JSON files
        model (str): Dataset type (train/test/val)
        
    Function details:
    1. Reads the V7 format JSON file
    2. Converts data to V8 format structure
    3. Splits data based on different source datasets (dataset_id)
    4. Generates separate V8 format JSON files for each source dataset
    5. Preserves integrity of keypoints annotations, image information, and category data
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the JSON file
    with open(input_file, 'r') as f:
        primate_data = json.load(f)

    # Create a mapping from dataset_id to dataset name
    dataset_id_to_name = {}
    for dataset in primate_data['datasets']:
        dataset_id_to_name[dataset['id']] = dataset['prefix']
    
    # v7 version dataset' datasets lake the dataset['id'] for oms dataset
    dataset_id_to_name["17"] = "oms"
        # print(type(dataset['id']))
    # Initialize a dictionary to hold split data
    print("dataset_id_to_name", dataset_id_to_name)
    split_data = {}

    # Process the images and group them by source_dataset
    for image in primate_data['images']:
        dataset_id = image['dataset_id']

        # print("dataset_id:", type(dataset_id))

        source_dataset = dataset_id_to_name[str(dataset_id)]
        # dataset_id_to_name.get(str(dataset_id), '')

        if source_dataset not in split_data:
            print("source_dataset not in split_data", source_dataset)
            
        output_image = {
            'file_name': image['file_name'],
            'width': image['width'],
            'height': image['height'],
            'id': image['id'],
            'source_dataset': source_dataset,
            'dataset_id': image['dataset_id']
        }
        split_data[source_dataset]['images'].append(output_image)

    # Process the annotations and add them to the corresponding source_dataset
    for annotation in primate_data['annotations']:
        # print("annotation", annotation)
        # Find the image associated with this annotation to determine its source_dataset
        # dataset_id = image['dataset_id']
        # source_dataset = dataset_id_to_name.get(str(dataset_id), '')
        
        dataset_id = annotation['dataset_id']
        source_dataset = dataset_id_to_name[str(dataset_id)]
        
        output_annotation = {
                'id': annotation['id'],
                'image_id': annotation['image_id'],
                'category_id': 1,  # fixed, assume the whole dataset is one category
                'dataset_id': annotation['dataset_id'],
                'bbox': annotation['bbox'],
                'keypoints': annotation['keypoints'],
                'num_keypoints': 37,
                'area': annotation['area'],
                'iscrowd': annotation['iscrowd'],
                'keypoints_orig': annotation['keypoints_orig'],
                'bbox_orig': annotation['bbox_orig'],
        }
        split_data[source_dataset]['annotations'].append(output_annotation)

    # Process the categories section (use the same categories for all subsets)
    pfm_keypoints = primate_data['pfm_keypoints']
    output_category = {
        'id': 1,
        'name': 'pfm',
        'supercategory': 'animal',
        'keypoints': pfm_keypoints
    }
    # Assign categories to each split dataset
    for source_dataset in split_data:
        split_data[source_dataset]['categories'].append(output_category)

    # Save each split dataset to a separate JSON file
    for source_dataset, data in split_data.items():
        output_file = os.path.join(output_dir, f"{source_dataset}_{model}.json")
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved {source_dataset} data to {output_file}")
        

def check_image_exist_in_json(json_path, image_dir):
    
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Extract the list of images
    images = data.get('images', [])
    total_images = len(images)
    missing_images = []
    
    # Iterate over each image entry
    for image in images:
        file_name = image['file_name']
        # Construct the full path to the image file
        image_path = os.path.join(image_dir, file_name)
        
        # Check if the image file exists
        if not os.path.isfile(image_path):
            # missing_images.append(file_name)
            missing_images.append(image_path)
            
    # Output the results
    print(f"Total images listed in JSON: {total_images}")
    if missing_images:
        print(f"Missing images: {len(missing_images)}")
        # for img in missing_images:
        #     print(f"  - {img}")
    else:
        print("All images exist in the specified directory.")
    
def get_file_name_from_image_id(json_path, image_id):
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Extract the list of images
    images = data.get('images', [])
    
    # Iterate over each image entry
    for image in images:
        if image['id'] == image_id:
            return image['file_name']
    
    return None    

if __name__ == "__main__":
    # Define your input and output paths
    # input_file = '/mnt/tiwang/primate_data/primate_test_1.2.json'

    model = "test" # train test val
     # val
    input_file = f'/mnt/data/tiwang/v7/annotations/pfm_{model}_apr15.json'
    output_dir = f'/mnt/data/tiwang/primate_data/data_v8.1/splitted_{model}_datasets/'

    print(input_file)
    convert_v7_to_v8_and_split_datasets(input_file, output_dir, model)

    # extract_json(json_file_path)
    # print_json(json_file_path)
    # search_file_by_id(json_file_path, 1
    
    # json_dir = "/mnt/tiwang/primate_data/splitted_val_datasets/oms_val.json" 
    # json_dir = "/mnt/tiwang/primate_data/splitted_val_datasets/kinka_val.json"
    
    data_dir_base = "/mnt/tiwang/v8_coco/images/"

    # check_image_exist_in_json(json_dir, data_dir_base)

    # check all json for each subdataset
    # json_dir = "/mnt/tiwang/primate_data/splitted_val_datasets"
    # json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    # for json_file in json_files:
    #     json_path = os.path.join(json_dir, json_file)  
    #     check_image_exist_in_json(json_path, data_dir_base)  