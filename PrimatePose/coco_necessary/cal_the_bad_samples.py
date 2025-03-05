import json
import matplotlib.pyplot as plt

def visualize_error_counts(error_counts, num_annotations, filename, image_path):
    """Visualize the annotation error counts using a bar chart."""
    # Extract error types and counts for plotting
    error_types = list(error_counts.keys())
    error_values = list(error_counts.values())
    
    # Create a bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(error_types, error_values, color='skyblue')
    
    # Add value labels above the bars
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01 * max(error_values),
            f'{int(bar.get_height()):,}',  # Add commas to the numbers for better readability
            ha='center', va='bottom',
            fontsize=10, color='black',
        )
    
    # Set chart title and labels
    plt.title('Annotations from ' + filename, fontsize=14)
    plt.xlabel("total number of annotations:" + str(num_annotations))
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, fontsize=10, ha='right')
    plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
    # Remove top and right spines (borders)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(image_path)
    
def load_json(file_path):
    """Load JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def analyze_annotations(data):
    """Analyze annotations to find different types of errors and count them."""
    error_counts = {
        # 'missing_bbox': 0,  # Counter for annotations without bbox
        'w_h_negative_or_zero': 0,  # Combined count for negative or zero width/height
        'xmin+w_out_of_bounds': 0,   # Count of bboxes where width goes out of bounds
        'ymin+h_out_of_bounds': 0,   # Count of bboxes where height goes out of bounds
        'missing_image_reference': 0,
        'no_keypoints_annitations': 0,   # Count of annotations without keypoints
        'keypoints_annotations': 0    # Count of annotations with keypoints
    }
    # Create a dictionary of image dimensions for quick lookup
    image_dims = {image['id']: (image['width'], image['height']) for image in data['images']}
    
    for annotation in data['annotations']:
        img_id = annotation['image_id']
        if img_id not in image_dims:
            error_counts['missing_image_reference'] += 1
            continue
        
        # Check if 'bbox' is present and not empty
        # if 'bbox' not in annotation or not annotation['bbox']:
        #     error_counts['missing_bbox'] += 1
        #     continue
        
        img_width, img_height = image_dims[img_id]
        xmin, ymin, width, height = annotation['bbox']

        # Check for negative or zero width and height
        if width <= 0 or height <= 0:
            # print("width:", width, "height:", height)
            error_counts['w_h_negative_or_zero'] += 1
            continue

        # Calculate the bbox boundaries
        xmax = xmin + width
        ymax = ymin + height
        # Check if the entire bbox is out of image boundaries
        if xmax > img_width:
            error_counts['xmin+w_out_of_bounds'] += 1
            continue
            
        elif ymax > img_height:
            error_counts['ymin+h_out_of_bounds'] += 1
            continue
        
        # Check if keypoints are missing or contain only zeros
        keypoints = annotation.get('keypoints', [])

        visibility_labels = keypoints[2::3]  # Extract visibility values (every third item)
        if all(v == -1 for v in visibility_labels):
            error_counts['no_keypoints_annitations'] += 1
        else:
            error_counts['keypoints_annotations'] += 1
            
    return error_counts

import logging
def cal_different_kinds_of_samples(input_file, txt_path):
    logging.basicConfig(
    filename=txt_path,
    filemode='a',  # 'a' for append, 'w' for overwrite
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO)
    
    # Load data from the JSON file
    data = load_json(input_file)
    file_name = input_file.split('/')[-1]
    print("JSON file:", file_name)
    logging.info("JSON file: {}".format(file_name))
    num_annotations = len(data['annotations'])
    print("total number of annotations:", num_annotations)
    logging.info("total number of annotations: {}".format(num_annotations))
    # Analyze annotations and get the count of different types of errors
    error_counts = analyze_annotations(data)
    
    # Print the summary of errors found
    print("Annotation Errors Summary:")
    for category, count in error_counts.items():
        print(f"{category}: {count}")
        logging.info("{}: {}".format(category, count))

    # Visualize the error counts using a bar chart
    # image_path = "/app/analyse_json/annotations_{}.png".format(file_name.split('.')[0])
    # visualize_error_counts(error_counts, num_annotations, file_name, image_path)

def main():
    
    # input_file = '/app/data/primate_data/pfm_train_v8.json'
    # input_file = '/app/data/v7/annotations/pfm_train_apr15.json'
    # input_file = '/app/data/primate_data/pfm_train_v8.json'
    # input_file = '/app/data/primate_data/splitted_test_datasets/chimpact_test.json'
    input_file = '/app/data/primate_data/splitted_train_datasets/ak_new_train.json'
    
    # Load data from the JSON file
    data = load_json(input_file)
    file_name = input_file.split('/')[-1]
    print("JSON file:", file_name)
    num_annotations = len(data['annotations'])
    print("total number of annotations:", num_annotations)
    # Analyze annotations and get the count of different types of errors
    error_counts = analyze_annotations(data)
    
    # Print the summary of errors found
    print("Annotation Errors Summary:")
    for category, count in error_counts.items():
        print(f"{category}: {count}")

    # Visualize the error counts using a bar chart
    image_path = "/app/analyse_json/annotations_{}.png".format(file_name.split('.')[0])
    visualize_error_counts(error_counts, num_annotations, file_name, image_path)

if __name__ == "__main__":
    main()