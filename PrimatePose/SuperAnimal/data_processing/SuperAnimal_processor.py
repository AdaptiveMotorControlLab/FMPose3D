import json
import os

class SuperAnimalProcessor:
    """
    Helper class providing helper functions for Super-Animal datasets
    """
    def __init__(self, data_path):
        pass

    @staticmethod
    def print_all_dataset_names(data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        images = data['images']
        unique_datasets = set()
        for image in images:
            dataset_name = image['source_dataset']
            unique_datasets.add(dataset_name)
        
        for dataset_name in sorted(unique_datasets):
            print(dataset_name)


    @staticmethod
    def leave_one_dataset_out(full_dataset_json_path, OOD_dataset_name, output_folder):
        with open(full_dataset_json_path, 'r') as f:
            data = json.load(f)
        
        images = data["images"]
        annotations = data["annotations"]
        
        # Get mode from filename
        mode = full_dataset_json_path.split("/")[-1].split(".")[0]
        
        # Split images
        OOD_images = []
        IID_images = []
        OOD_image_ids = set()
        IID_image_ids = set()
        
        for image in images:
            if image["source_dataset"] == OOD_dataset_name:
                OOD_images.append(image)
                OOD_image_ids.add(image["id"])
            else:
                IID_images.append(image)
                IID_image_ids.add(image["id"])
        
        # Split annotations
        OOD_annotations = []
        IID_annotations = []
        uncategorized_annotations = []
        
        for annotation in annotations:
            if annotation["image_id"] in OOD_image_ids:
                OOD_annotations.append(annotation)
            elif annotation["image_id"] in IID_image_ids:
                IID_annotations.append(annotation)
            else:
                uncategorized_annotations.append(annotation)
        
        # Report any uncategorized annotations
        if uncategorized_annotations:
            print(f"WARNING: Found {len(uncategorized_annotations)} annotations that don't belong to either OOD or IID images")
            # print(f"First few uncategorized annotation IDs: {[a['id'] for a in uncategorized_annotations[:5]]}")
            # print(f"Their image IDs: {[a['image_id'] for a in uncategorized_annotations[:5]]}")
        
        # Create output JSONs
        OOD_data = {
            "images": OOD_images,
            "annotations": OOD_annotations,
            "categories": data["categories"]
        }
        
        IID_data = {
            "images": IID_images,
            "annotations": IID_annotations,
            "categories": data["categories"]
        }
        
        # Save output files
        os.makedirs(output_folder, exist_ok=True)
        
        ood_filename = f"{mode}_OOD_{OOD_dataset_name}.json"
        iid_filename = f"{mode}_IID_wo_{OOD_dataset_name}.json"
        
        with open(os.path.join(output_folder, ood_filename), 'w') as f:
            json.dump(OOD_data, f, indent=4)
        
        with open(os.path.join(output_folder, iid_filename), 'w') as f:
            json.dump(IID_data, f, indent=4)
            

if __name__ == "__main__":
    
    data_path = "/home/ti_wang/Ti_workspace/PrimatePose/SuperAnimal/data/SuperAnimal/Quadruped80K/annotations/test.json"
    # SuperAnimalProcessor.print_all_dataset_names(data_path)

    # TVM_data_path = "/home/ti_wang/Ti_workspace/PrimatePose/SuperAnimal/data/SuperAnimal/TopViewMouse5K_20240913/annotations/train.json"
    # SuperAnimalProcessor.print_all_dataset_names(TVM_data_path)
    mode = ['train', 'test']
    for m in mode:
        TVM5k_data_path ="/home/ti_wang/Ti_workspace/PrimatePose/SuperAnimal/data/SuperAnimal/TopViewMouse5K_20240913/annotations/" + m + ".json"
        SuperAnimalProcessor.leave_one_dataset_out(TVM5k_data_path, OOD_dataset_name="openfield-Pranav-2018-08-20", \
                                                output_folder= "/home/ti_wang/Ti_workspace/PrimatePose/SuperAnimal/data/SuperAnimal/TopViewMouse5K_20240913/annotations")