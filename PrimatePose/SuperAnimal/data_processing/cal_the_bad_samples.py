import json
import os 
import logging

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
            error_counts['w_h_negative_or_zero'] += 1
            continue

        # Calculate the bbox boundaries
        xmax = xmin + width
        ymax = ymin + height
        # Check if the entire bbox is out of image boundaries
        if xmax >= img_width:
            error_counts['xmin+w_out_of_bounds'] += 1
            continue
            
        elif ymax >= img_height:
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

def cal_different_kinds_of_samples(input_file, txt_path):
    logging.basicConfig(
        filename=txt_path,
        filemode='a',
        format='%(message)s',
        level=logging.INFO
    )
    
    # Load data from the JSON file
    data = load_json(input_file)
    file_name = input_file.split('/')[-1].replace('.json', '')  # 移除 .json 后缀
    
    logging.info(f"===== Dataset: {file_name} =====")
    print(f"===== Dataset: {file_name} =====")
    
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