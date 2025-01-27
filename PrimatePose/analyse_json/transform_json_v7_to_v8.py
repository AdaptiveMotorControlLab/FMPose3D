import json

# Define the file paths
# input_file = '/mnt/tiwang/v7/annotations/pfm_train_apr15.json'
# output_file = '/mnt/tiwang/primate_data/pfm_train_v8.json'

input_file = '/mnt/tiwang/v7/annotations/pfm_val_apr15.json'
output_file = '/mnt/tiwang/primate_data/pfm_val_v8.json'

# input_file = '/mnt/tiwang/v7/annotations/pfm_test_apr15.json'
# output_file = '/mnt/tiwang/primate_data/pfm_test_v8.json'

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
        'source_dataset': source_dataset
    }
    output_data['images'].append(output_image)
    
# Process the annotations section
for annotation in primate_data['annotations']:
    output_annotation = {
        'id': annotation['id'],
        'image_id': annotation['image_id'],
        'category_id': 1, # fixed, assume the whole dataset is one category
        'bbox': annotation['bbox'],
        'keypoints': annotation['keypoints'],
        'num_keypoints': 37,
        'area': annotation['area'],
        'iscrowd': annotation['iscrowd']
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