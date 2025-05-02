# from deeplabcut.common_tools.json_tools import convert_json_indent_folder_level
import json
import random

def cal_number_of_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return len(data['annotations'])


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
        
        
if __name__ == "__main__":
    
    # convert original coco json files to indent format
    original_json_path = "/home/ti_wang/Ti_workspace/projects/COCO_data/cocoapi/COCO/annotations"
    output_path = "/home/ti_wang/Ti_workspace/projects/COCO_data/cocoapi/COCO/annotations_indent"
    # convert_json_indent_folder_level(original_json_path, output_path)
    
    val_json_path = "/home/ti_wang/Ti_workspace/sapiens/COCO_data/annotations_indent/person_keypoints_val2017.json" 
    # print("val: ", cal_number_of_annotations(val_json_path)) # 11004
    # train_json_path = "/home/ti_wang/Ti_workspace/sapiens/COCO_data/annotations_indent/person_keypoints_train2017.json"
    # print("train: ", cal_number_of_annotations(train_json_path)) # 262465


    # Create a small dataset with 1/20 of the validation data
    # mode = "val"
    mode = "train"
    json_path = f"/home/ti_wang/Ti_workspace/sapiens/COCO_data/annotations_indent/person_keypoints_{mode}2017.json"
    small_output_path = f"/home/ti_wang/Ti_workspace/sapiens/COCO_data/annotations_indent/person_keypoints_{mode}2017_small_1_20.json"
    small_dataset_filter(json_path, small_output_path)
    
    print("small val dataset created: ", cal_number_of_annotations(small_output_path))




