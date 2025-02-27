import json

# Define the file paths
mode = 'val'
input_file = f'/mnt/data/tiwang/v7/annotations/pfm_{mode}_apr15.json'
output_file = f'/mnt/data/tiwang/primate_data/pfm_{mode}_v8.1.json'

# output_file = '/home/ti/projects/PrimatePose/ti_data/pfm_test_3_transformed1.json'
# input_file = '/home/ti/projects/PrimatePose/ti_data/data/pfm_test_3_items.json'

# Load primate.json
with open(input_file, 'r') as f:
    primate_data = json.load(f)

# Initialize the output data structure
output_data = {
    'images': [],
    'annotations': [],
    'categories': []
}

# Create a mapping from dataset_id to dataset name
dataset_id_to_name = {}
for dataset in primate_data['datasets']:
    dataset_id_to_name[dataset['id']] = dataset['name']
    # print(type(dataset['id']))
# print(dataset_id_to_name)

# Process the images section
for image in primate_data['images']:
    dataset_id = image['dataset_id']
    # print(type(dataset_id))
    source_dataset = dataset_id_to_name.get(str(dataset_id), '')
    # print(source_dataset)
    output_image = {
        'file_name': image['file_name'],
        'width': image['width'],
        'height': image['height'],
        'id': image['id'],
        'source_dataset': source_dataset,
        'dataset_id': image['dataset_id']
    }
    output_data['images'].append(output_image)
    
# Process the annotations section
for annotation in primate_data['annotations']:
    output_annotation = {
        'id': annotation['id'],
        'image_id': annotation['image_id'],
        'category_id': 1, # fixed, assume the whole dataset is one category
        'dataset_id': annotation['dataset_id'],
        'bbox': annotation['bbox'],
        'keypoints': annotation['keypoints'],
        'num_keypoints': 37,
        'area': annotation['area'],
        'iscrowd': annotation['iscrowd'],
        'keypoints_orig': annotation['keypoints_orig'],
        'bbox_orig': annotation['bbox_orig'],
    }
    output_data['annotations'].append(output_annotation)

# Process the categories section
pfm_keypoints = primate_data['pfm_keypoints']

# for category in primate_data['categories']:
#     output_category = {
#         'id': category['id'],
#         'name': category['common_name'],
#         'supercategory': category['superfamily'],
#         'keypoints': pfm_keypoints
#     }
#     output_data['categories'].append(output_category)
#     break

output_category = { 
    'id': 1,
    'name': 'pfm',
    'supercategory': 'animal',
    'keypoints': pfm_keypoints
}

output_data['categories'].append(output_category)

# Save the converted data to a new JSON file
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=4)
    
def split_v8_json(json_file, output_dir):
    pass