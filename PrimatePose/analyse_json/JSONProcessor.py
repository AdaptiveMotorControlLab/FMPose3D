import json
import os
import random
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

class JSONProcessor:
    
    def __init__(self, json_path):
        self.json_path = json_path
        self.data = self._load_json()

    def _load_json(self):
        """Load JSON data from the specified path."""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            return data
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return None

    def print_structure(self, data=None, indent=0):
        """Print the structure of JSON data."""
        if data is None:
            data = self.data

        if isinstance(data, dict):
            for key, value in data.items():
                print('  ' * indent + str(key) + ":")
                self.print_structure(value, indent + 1)
        elif isinstance(data, list):
            print('  ' * indent + f"List of {len(data)} items")
            if len(data) > 0:
                self.print_structure(data[0], indent + 1)
        else:
            print('  ' * indent + str(type(data).__name__))

    def get_file_name_from_image_id(self, image_id):
        """Retrieve the filename of an image by its ID from JSON data."""
        images = self.data.get("images", [])
        for image in images:
            if image.get("id") == image_id:
                return image.get("file_name")
        return None

    def check_image_exist_in_json(self, image_id):
        """Check if an image exists in the JSON data by its ID."""
        images = self.data.get("images", [])
        for image in images:
            if image.get("id") == image_id:
                return True
        return False

    def get_annotations_for_image(self, image_id):
        """Retrieve annotations associated with an image by its ID."""
        annotations = self.data.get("annotations", [])
        image_annotations = [ann for ann in annotations if ann.get("image_id") == image_id]
        return image_annotations

    def print_all_datasets_content(self):
        """Print all the content of the 'datasets' field in the JSON data."""
        datasets = self.data.get("datasets", [])
        if not datasets:
            print("No datasets found in the JSON data.")
            return
        print(f"Number of datasets: {len(datasets)}")
        print("Datasets Content:")
        for dataset in datasets:
            print("-" * 40)
            for key, value in dataset.items():
                print(f"{key}: {value}")
        print("-" * 40)
    
    def analyze_annotations(self):
        """Analyze annotations to find different types of errors and count them."""
        error_counts = {
            'w_h_negative_or_zero': 0,
            'xmin+w_out_of_bounds': 0,
            'ymin+h_out_of_bounds': 0,
            'missing_image_reference': 0,
            'no_keypoints_annotations': 0,
            'keypoints_annotations': 0
        }
        
        image_dims = {image['id']: (image['width'], image['height']) for image in self.data.get('images', [])}
        
        for annotation in self.data.get('annotations', []):
            img_id = annotation.get('image_id')
            if img_id not in image_dims:
                error_counts['missing_image_reference'] += 1
                continue
            
            img_width, img_height = image_dims[img_id]
            xmin, ymin, width, height = annotation.get('bbox', [0, 0, 0, 0])

            if width <= 0 or height <= 0:
                error_counts['w_h_negative_or_zero'] += 1
                continue

            xmax = xmin + width
            ymax = ymin + height
            if xmax > img_width:
                error_counts['xmin+w_out_of_bounds'] += 1
            if ymax > img_height:
                error_counts['ymin+h_out_of_bounds'] += 1
            
            keypoints = annotation.get('keypoints', [])
            visibility_labels = keypoints[2::3]
            if all(v == -1 for v in visibility_labels):
                error_counts['no_keypoints_annotations'] += 1
            else:
                error_counts['keypoints_annotations'] += 1
        
        return error_counts

    def visualize_error_counts(self, error_counts, num_annotations, filename, image_path):
        """Visualize the annotation error counts using a bar chart."""
        error_types = list(error_counts.keys())
        error_values = list(error_counts.values())
        
        plt.figure(figsize=(8, 5))
        bars = plt.bar(error_types, error_values, color='skyblue')
        
        for bar in bars:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01 * max(error_values),
                f'{int(bar.get_height()):,}',
                ha='center', va='bottom',
                fontsize=10, color='black',
            )
        
        plt.title('Annotations from ' + filename, fontsize=14)
        plt.xlabel("Total number of annotations: " + str(num_annotations))
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, fontsize=10, ha='right')
        plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(image_path)
        print(f"Visualization saved at {image_path}")

    def save_json(self, output_path):
        """Save the current JSON data to a new file."""
        try:
            with open(output_path, 'w') as file:
                json.dump(self.data, file, indent=4)
            print(f"JSON data saved to {output_path}")
        except Exception as e:
            print(f"Error saving JSON: {e}")

    def sample_json(self, nums, output_json_file):
        """Sample a specified number of images and corresponding annotations.
        This is for v8 version.
        """
        images = self.data['images']
        annotations = self.data['annotations']
        categories = self.data['categories']
        
        sampled_images = random.sample(images, nums)
        sampled_image_ids = {img['id'] for img in sampled_images}
        sampled_annotations = [ann for ann in annotations if ann['image_id'] in sampled_image_ids]

        sampled_data = {
            'images': sampled_images,
            'annotations': sampled_annotations,
            'categories': categories
        }

        with open(output_json_file, 'w') as f:
            json.dump(sampled_data, f, indent=4)
        print(f"Sampled data saved to {output_json_file}")

    def sample_json_v7(self, nums, output_json_file):
        """Sample a specified number of images and corresponding annotations.
        This is for v7 version.
        """
        images = self.data['images']
        annotations = self.data['annotations']
        categories = self.data['categories']
        datasets = self.data['datasets']
        pfm_skeleton = self.data['pfm_skeleton']
        pfm_keypoints = self.data['pfm_keypoints']
        
        sampled_images = random.sample(images, nums)
        sampled_image_ids = {img['id'] for img in sampled_images}
        sampled_annotations = [ann for ann in annotations if ann['image_id'] in sampled_image_ids]

        sampled_data = {
            'images': sampled_images,
            'annotations': sampled_annotations,
            'categories': categories,
            'datasets': datasets,
            'pfm_skeleton': pfm_skeleton,
            'pfm_keypoints': pfm_keypoints
        }

        with open(output_json_file, 'w') as f:
            json.dump(sampled_data, f, indent=4)
        print(f"Sampled data saved to {output_json_file}")
        

    def select_sample_by_image_id(self, image_id, target_json_file):
        """Select and save sample by a specific image ID."""
        filtered_images = [image for image in self.data['images'] if image['id'] == image_id]
        
        if not filtered_images:
            print(f"No image found with image_id: {image_id}")
            return

        filtered_annotations = [annotation for annotation in self.data['annotations'] if annotation['image_id'] == image_id]
        
        if not filtered_annotations:
            print(f"No annotations found for image_id: {image_id}")

        filtered_data = {
            "images": filtered_images,
            "annotations": filtered_annotations,
            "categories": self.data["categories"]
        }

        with open(target_json_file, 'w') as target_file:
            json.dump(filtered_data, target_file, indent=2)
        print(f"Filtered data saved to {target_json_file}")

    def print_keypoints_names(self):
        """Print all keypoints defined in the categories."""
        categories = self.data.get('categories', [])
        if not categories:
            print("No categories found in the JSON data.")
            return

        for category in categories:
            keypoints = category.get('keypoints', [])
            if not keypoints:
                print(f"No keypoints found in category {category.get('name', 'unknown')}")
                continue
            
            print(f"\nKeypoints for category '{category.get('name', 'unknown')}':")
            print("-" * 40)
            for idx, keypoint in enumerate(keypoints, 1):
                print(f"{idx:2d}. {keypoint}")
            print("-" * 40)
            print(f"Total number of keypoints: {len(keypoints)}")

    def merge_json_files(self, json_folder_path, output_path, exclude_datasets=None):
        """Merge multiple JSON files from a folder into a single JSON file.
        Dataset names are extracted from filenames (first part before '_').
        
        Args:
            json_folder_path (str): Path to folder containing JSON files to merge
            output_path (str): Path where the merged JSON file will be saved
            exclude_datasets (list, optional): List of dataset names to exclude from merging
        """
        # Setup logging
        log_dir = os.path.dirname(output_path)
        output_name = os.path.splitext(os.path.basename(output_path))[0]
        log_file = os.path.join(log_dir, f"{output_name}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        if not os.path.exists(json_folder_path):
            logging.error(f"Folder path does not exist: {json_folder_path}")
            raise ValueError(f"Folder path does not exist: {json_folder_path}")
            
        exclude_datasets = set(exclude_datasets or [])
        merged_data = {
            'images': [],
            'annotations': [],
            'categories': None,  # Will be set from first valid file
            'datasets': []
        }
            
        json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]
        if not json_files:
            logging.warning(f"No JSON files found in {json_folder_path}")
            return
        
        logging.info(f"Found {len(json_files)} JSON files to merge")
        if exclude_datasets:
            logging.info(f"Excluding datasets: {', '.join(exclude_datasets)}")
        else:
            logging.info("No datasets excluded")
        
        processed_files = 0
        
        for file_name in json_files:
            dataset_name = file_name.split('_')[0] 
            if dataset_name in exclude_datasets:
                logging.info(f"Skipping {file_name} (dataset: {dataset_name})")
                continue
            
            with open(os.path.join(json_folder_path, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Set categories from first valid file
            if not merged_data['categories'] and 'categories' in data:
                merged_data['categories'] = data['categories']
                logging.debug(f"Categories set from {file_name}")
            
            # Add dataset info from filename
            merged_data['datasets'].append(dataset_name)
            
            # Simply append images and annotations
            merged_data['images'].extend(data.get('images', []))
            merged_data['annotations'].extend(data.get('annotations', []))
            
            processed_files += 1
            logging.info(f"Processed {file_name}")
        
        if processed_files == 0:
            logging.warning("No files were successfully processed")
            return
            
        # Log merge summary
        logging.info("\nMerge Summary:")
        logging.info(f"Files processed: {processed_files}")
        logging.info(f"Total images: {len(merged_data['images'])}")
        logging.info(f"Total annotations: {len(merged_data['annotations'])}")
        logging.info(f"Datasets included: {merged_data['datasets']}")
        
        # Save merged data
        with open(output_path, 'w') as f:
            json.dump(merged_data, f, indent=4)
        logging.info(f"Merged data saved to {output_path}")
        logging.info(f"Log file saved to {log_file}")

    def find_annotations_with_pose(self, input_json_path: str, output_json_path: str) -> None:
        """
        Find annotations with pose data and save them to a new JSON file while maintaining structure.
        An annotation is considered to have pose if NOT ALL visibility labels are -1.
        
        Args:
            input_json_path: Path to input JSON file
            output_json_path: Path to save filtered JSON file
        """
        # Read input JSON
        with open(input_json_path, 'r') as f:
            data = json.load(f)
        
        # Pre-allocate data structures
        valid_image_ids: Set[int] = set()
        filtered_annotations: List[Dict] = []
        
        # Process annotations in a single pass
        for ann in data['annotations']:
            # Get visibility labels (every 3rd element starting from index 2)
            vis_labels = ann['keypoints'][2::3]
            
            # Check if NOT ALL visibility labels are -1
            if not all(label == -1 for label in vis_labels):
                filtered_annotations.append(ann)
                valid_image_ids.add(ann['image_id'])
        
        # Create output JSON with same structure using dict comprehension for images
        output_data = {
            'images': [img for img in data['images'] if img['id'] in valid_image_ids],
            'annotations': filtered_annotations,
            'categories': data['categories']
        }
        
        # Include datasets if they exist in the input data
        if 'datasets' in data:
            output_data['datasets'] = data['datasets']
        
        # Ensure output directory exists and save
        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)
            print(f"Output saved to {output_json_path}")

    def remove_annotaions_w_wrong_bbox(self, input_json_path: str, output_json_path: str) -> None:
        """
        Remove annotations with wrong bbox.
        if we find the bbox is out of the image or has invalid dimensions (width/height <= 0), we remove the annotation.
        """
        with open(input_json_path, 'r') as f:
            data = json.load(f)
        
        image_id_to_image_info = {img['id']: img for img in data['images']}
        
        # Create a new list of valid annotations instead of modifying the list while iterating
        valid_annotations = []
        for ann in data['annotations']:
            img_id = ann['image_id']
            img_width = image_id_to_image_info[img_id]['width']
            img_height = image_id_to_image_info[img_id]['height']
            xmin, ymin, width, height = ann['bbox']
            
            # Keep annotation only if all conditions are met
            if (xmin >= 0 and ymin >= 0 and 
                width > 0 and height > 0 and 
                xmin + width <= img_width and 
                ymin + height <= img_height):
                valid_annotations.append(ann)
        
        # Update the annotations list with valid ones
        data['annotations'] = valid_annotations
                
        with open(output_json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Output saved to {output_json_path}")

    @staticmethod
    def cal_number_of_annotations(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return len(data['annotations'])

    def small_dataset_filter(self, json_path, output_path, sample_rate=1/20):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
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
    
    def remove_redudant_images(self, input_json_path, output_json_path):
        """
        Remove redundant images from JSON data.
        make sure every image in this json file is mentioned in the annotations.
        """
        # Load JSON data
        with open(input_json_path, 'r') as f:
            data = json.load(f)
        
        # Get all image IDs referenced in annotations
        referenced_image_ids = set()
        for annotation in data['annotations']:
            referenced_image_ids.add(annotation['image_id'])
        
        # Filter images list to keep only referenced images
        filtered_images = [img for img in data['images'] if img['id'] in referenced_image_ids]
        
        # Update the images list in the data
        data['images'] = filtered_images
        
        # Save the updated JSON data
        with open(output_json_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"Original images count: {len(data['images'])}")
        print(f"Images referenced in annotations: {len(referenced_image_ids)}")
        print(f"Removed {len(data['images']) - len(filtered_images)} redundant images")
        print(f"Saved filtered JSON to {output_json_path}")


# Example usage:

if __name__ == "__main__":
    
    # processor = JSONProcessor("/app/data/v7/annotations/pfm_train_apr15.json")
    # processor = JSONProcessor("/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/pfm_train_v8.json")
    # processor = JSONProcessor("/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_test_datasets/oms_test.json")
    # processor = JSONProcessor("/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/pfm_test_merged.json")

    processor = JSONProcessor("/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_test_datasets/lote_test.json")

    species = "riken"
    train_file= f"/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/splitted_train_datasets/{species}_train.json"
    test_file= f"/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/splitted_test_datasets/{species}_test.json"
    print(JSONProcessor.cal_number_of_annotations(train_file))
    print(JSONProcessor.cal_number_of_annotations(test_file))
    # Print JSON structure
    # processor.print_structure()
    
    # mode_list = ["train", "test"] #  "val"]
    # species = "oms"
    # for mode in mode_list:
        
    #     input_json_path = f"/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/splitted_{mode}_datasets/{species}_{mode}.json"
    #     output_json_path = f"/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/samples/{species}_{mode}_small_1_500.json"
    #     processor.small_dataset_filter(json_path=input_json_path, \
    #                                    output_path=output_json_path,
    #                                    sample_rate=1/500)
    #     num_annotations = processor.cal_number_of_annotations(json_path=output_json_path)
    #     print(f"Number of annotations in {mode} dataset: {num_annotations}")
        
        
        # input_json_path = f"/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/8.21_sapiens/{species}_{mode}_pose_v8_21.json"
        # output_json_path = f"/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/8.21_sapiens/{species}_{mode}_pose_v8_21_rm_useless_images.json"
        # processor.remove_redudant_images(input_json_path, output_json_path)
    
    # for mode in mode_list:
    #     processor.merge_json_files(json_folder_path=f"/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/splitted_{mode}_datasets", \
    #                            output_path=f"/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/pfm_{mode}_wo_riken_V82.json",
    #                            exclude_datasets=["riken"])
    #     processor.find_annotations_with_pose(input_json_path=f"/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/pfm_{mode}_wo_riken_V82.json", \
    #                                     output_json_path=f"/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/pfm_{mode}_pose_wo_riken_V82.json")
    
        # processor.remove_annotaions_w_wrong_bbox(input_json_path=f"/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/pfm_{mode}_pose_V82.json", \
                                                # output_json_path=f"/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/pfm_{mode}_pose_V82_no_wrong_bbox.json")
    
    
    # processor.merge_json_files(json_folder_path="/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/test_goodpose_datasets", output_path="/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/pfm_test_goodpose_merged.json")
    
    # processor.print_all_datasets_content()
    # processor.print_keypoints_names()
    # processor.sample_json(nums=20, output_json_file="/home/ti_wang/data/tiwang/primate_data/samples/lote_test_sampled_20.json")
    
    # Sample a subset of JSON data
    
    # processor.sample_json_v7(nums=10, output_json_file="/app/data/primate_data/samples/pfm_train_apr15_sampled_10.json")

    # Select a specific image ID sample
    # image_id = 123  # Example image ID
    # processor.select_sample_by_image_id(image_id, target_json_file="selected_sample_output.json")

    # # Analyze annotations and visualize error counts
    # error_counts = processor.analyze_annotations()
    # print("Annotation Errors Summary:", error_counts)
    # output_image_path = "annotations_errors.png"
    # processor.visualize_error_counts(error_counts, len(processor.data.get('annotations', [])), os.path.basename(processor.json_path), output_image_path)