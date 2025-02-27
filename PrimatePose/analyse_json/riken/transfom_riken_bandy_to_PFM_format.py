import json
import os

def transform_riken_to_pfm(mode):
    # Define paths based on mode
    input_path = f'/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/Marmosets-Banty-2022-07-01_coco/annotations/{mode}.json'
    categories_path = '/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/samples/riken_test_pose.json'
    output_dir = '/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/samples'
    output_path = os.path.join(output_dir, f'riken_bandy_pose_{mode}_pfm.json')

    # Keypoint mapping from Riken Bandy to PFM V8.2
    keypoint_mapping = [
        -1, -1, 1, 2, 0, 3, 4, -1, -1, -1, -1,
        -1, 5, 6, -1, -1, -1, -1, 7, 8, 9, 10,
        -1, -1, 11, 12, -1, 13, 14, 15, 16, -1, -1,
        17, 18, -1, 19
    ]

    # Load source data
    with open(input_path, 'r') as f:
        source_data = json.load(f)
    
    # Load PFM categories
    with open(categories_path, 'r') as f:
        pfm_categories = json.load(f)['categories']

    # Process annotations
    new_annotations = []
    for ann in source_data['annotations']:
        # Initialize PFM keypoints (37 points * 3 values)
        pfm_keypoints = [-1] * (37 * 3)
        valid_count = 0
        
        # Map keypoints
        for pfm_idx, rb_idx in enumerate(keypoint_mapping):
            if rb_idx == -1:
                continue  # Keep as 0 with visibility 0
                
            try:
                # Get original keypoint values
                orig_idx = rb_idx * 3
                x = ann['keypoints'][orig_idx]
                y = ann['keypoints'][orig_idx+1]
                v = ann['keypoints'][orig_idx+2]
                
                # Set mapped values
                pfm_keypoints[pfm_idx*3] = x
                pfm_keypoints[pfm_idx*3+1] = y
                pfm_keypoints[pfm_idx*3+2] = v
                
                if v > 0:  # Count visible keypoints
                    valid_count += 1
            except IndexError:
                continue

        # Update annotation
        new_ann = ann.copy()
        new_ann['keypoints'] = pfm_keypoints
        # new_ann['num_keypoints'] = valid_count
        new_annotations.append(new_ann)

    # Create output data
    output_data = {
        'images': source_data['images'],
        'annotations': new_annotations,
        'categories': pfm_categories
    }

    # Save result
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

if __name__ == '__main__':
    for mode in ['train', 'test']:
        transform_riken_to_pfm(mode)

