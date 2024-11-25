import json
import os
import random
import matplotlib.pyplot as plt

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
            'no_keypoints_annotations': 0
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

    def merge_json_files(self, json_folder_path, output_path):
        """Merge multiple JSON files from a folder into a single JSON file.
        Preserves original image and annotation IDs.
        
        Args:
            json_folder_path (str): Path to folder containing JSON files to merge
            output_path (str): Path where the merged JSON file will be saved
        """
        merged_data = {
            'images': [],
            'annotations': [],
            'categories': []  # We'll take categories from the first file
        }
        
        # Get all JSON files in the folder
        json_files = [f for f in os.listdir(json_folder_path) if f.endswith('.json')]
        
        if not json_files:
            print(f"No JSON files found in {json_folder_path}")
            return
        
        print(f"Found {len(json_files)} JSON files to merge")
        
        # Process each JSON file
        for json_file in json_files:
            file_path = os.path.join(json_folder_path, json_file)
            print(f"Processing {json_file}...")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Take categories from first file if not yet set
                if not merged_data['categories'] and 'categories' in data:
                    merged_data['categories'] = data['categories']
                
                # Simply append images and annotations
                merged_data['images'].extend(data.get('images', []))
                merged_data['annotations'].extend(data.get('annotations', []))
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        # Save the merged JSON data to the output path
        with open(output_path, 'w') as f:
            json.dump(merged_data, f, indent=4)
        print(f"Merged JSON data saved to {output_path}")




# Example usage:
if __name__ == "__main__":
    
    # processor = JSONProcessor("/app/data/v7/annotations/pfm_train_apr15.json")

    # processor = JSONProcessor("/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/pfm_train_v8.json")
    # processor = JSONProcessor("/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_test_datasets/oms_test.json")
    processor = JSONProcessor("/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/pfm_test_merged.json")

    # Print JSON structure
    processor.print_structure()
    # processor.merge_json_files(json_folder_path="/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/train_datasets_checked", output_path="/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/pfm_train_merged.json")
    # processor.merge_json_files(json_folder_path="/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/test_datasets_checked", output_path="/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/pfm_test_merged.json")
    
    # processor.print_all_datasets_content()
    # processor.print_keypoints_names()
    # processor.sample_json(nums=50, output_json_file="/home/ti_wang/data/tiwang/primate_data/samples/oms_test_sampled_50.json")
    
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