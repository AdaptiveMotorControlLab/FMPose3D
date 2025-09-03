import json
import random

def cal_number_of_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return len(data['annotations'])

def cal_number_of_images(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return len(data['images'])

def small_dataset_filter(json_path, output_path, sample_rate=1/20):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # randomly sample 5% of images (1/20)
    all_images = data['images']
    sample_size = int(len(all_images) * sample_rate)
    filtered_images = random.sample(all_images, sample_size)
    data['images'] = filtered_images
    
    # get filtered image ids
    filtered_image_ids = set(img['id'] for img in filtered_images)
    
    # filter annotations to keep only those that correspond to filtered images
    data['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] in filtered_image_ids]
    
    # save to output_path
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
        
def annotation_has_keypoints(annotation):
    """Return True if an annotation contains any labeled keypoints.

    Logic:
    - If 'num_keypoints' exists, require > 0
    - Else, require any visibility flag > 0 among the keypoints' (v) entries
    - Robust to missing or malformed 'keypoints' field
    """
    keypoints = annotation.get('keypoints')
    if not isinstance(keypoints, list) or len(keypoints) == 0 or len(keypoints) % 3 != 0:
        return False

    num_keypoints = annotation.get('num_keypoints')
    if isinstance(num_keypoints, int):
        return num_keypoints > 0

    try:
        # Visibility flags are every third value starting at index 2
        return any(int(v) > 0 for v in keypoints[2::3])
    except Exception:
        return False

def cal_number_of_valid_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return sum(1 for ann in data.get('annotations', []) if annotation_has_keypoints(ann))

def get_the_number_of_valid_images(json_path):
    """Return the number of valid images (with at least one valid keypoint annotation)."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    valid_image_ids = set()
    for ann in data.get('annotations', []):
        if annotation_has_keypoints(ann):
            image_id = ann.get('image_id')
            if image_id is not None:
                valid_image_ids.add(image_id)
    return len(valid_image_ids)

        
if __name__ == "__main__":
    
    # convert original coco json files to indent format
    # original_json_path = "/home/ti_wang/Ti_workspace/projects/COCO_data/cocoapi/COCO/annotations"
    # output_path = "/home/ti_wang/Ti_workspace/projects/COCO_data/cocoapi/COCO/annotations_indent"
    # convert_json_indent_folder_level(original_json_path, output_path)
    
    val_json_path = "/home/ti_wang/Ti_workspace/sapiens/COCO_data/annotations_indent/person_keypoints_val2017.json" 
    train_json_path = "/home/ti_wang/Ti_workspace/sapiens/COCO_data/annotations_indent/person_keypoints_train2017.json"
    print("val: ", cal_number_of_valid_annotations(val_json_path)) # 6352
    print("val images: ", get_the_number_of_valid_images(val_json_path)) # 2346
    print("train: ", cal_number_of_valid_annotations(train_json_path)) # 149813
    print("train images: ", get_the_number_of_valid_images(train_json_path)) # 56599
    
    # train_json_path = "/home/ti_wang/Ti_workspace/sapiens/COCO_data/annotations_indent/person_keypoints_train2017.json"
    # print("train: ", cal_number_of_annotations(train_json_path)) # 262465

    # Create a small dataset with 1/20 of the validation data
    # mode = "val"
    # mode = "train"
    # json_path = f"/home/ti_wang/Ti_workspace/sapiens/COCO_data/annotations_indent/person_keypoints_{mode}2017.json"
    # small_output_path = f"/home/ti_wang/Ti_workspace/sapiens/COCO_data/annotations_indent/person_keypoints_{mode}2017_small_1_20.json"
    # small_dataset_filter(json_path, small_output_path)
    
    # print("small val dataset created: ", cal_number_of_annotations(small_output_path))




