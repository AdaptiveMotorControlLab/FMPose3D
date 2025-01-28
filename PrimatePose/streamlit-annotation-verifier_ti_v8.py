# for v8
import streamlit as st
import json
import os
import cv2
import numpy as np
from PIL import Image
import tempfile

# Define the skeletons
PFM_SKELETON = [
    [3, 5], [4, 5], [6, 3], [7, 4],
    [5, 12], [13, 12], [14, 12], [2, 17],
    [19, 13], [20, 14], [21, 19], [22, 20],
    [23, 21], [24, 22], [25, 12], [26, 12],
    [25, 27], [26, 27], [25, 28], [26, 29],
    [27, 28], [27, 29], [28, 30], [29, 31],
    [30, 32], [31, 33], [27, 34], [34, 35],
    [35, 36], [36, 37]
]

TOPVIEWMOUSE_COLOR_MAP = {
    "nose": (255, 0, 0),
    "left_ear": (0, 255, 0),
    "right_ear": (0, 0, 255),
    "left_ear_tip": (255, 255, 0),
    "right_ear_tip": (255, 0, 255),
    "left_eye": (0, 255, 255),
    "right_eye": (128, 0, 0),
    "neck": (0, 128, 0),
    "mid_back": (0, 0, 128),
    "mouse_center": (128, 128, 0),
    "mid_backend": (128, 0, 128),
    "mid_backend2": (0, 128, 128),
    "mid_backend3": (192, 192, 192),
    "tail_base": (128, 128, 128),
    "tail1": (64, 64, 64),
    "tail2": (255, 128, 0),
    "tail3": (128, 255, 0),
    "tail4": (0, 255, 128),
    "tail5": (0, 128, 255),
    "left_shoulder": (255, 0, 128),
    "left_midside": (255, 128, 128),
    "left_hip": (128, 255, 255),
    "right_shoulder": (128, 0, 64),
    "right_midside": (64, 0, 128),
    "right_hip": (128, 64, 0),
    "tail_end": (64, 128, 0),
    "head_midpoint": (0, 64, 128)
}

BIRD_COLOR_MAP = {
    "back": (128, 0, 128),
    "bill": (0, 102, 204),
    "belly": (255, 128, 0),
    "breast": (0, 255, 255),
    "crown": (0, 255, 0),
    "forehead": (255, 105, 180),
    "left_eye": (102, 0, 204),
    "left_leg": (139, 69, 19),
    "left_wing_tip": (75, 0, 130),
    "left_wrist": (255, 140, 0),
    "nape": (255, 51, 51),
    "right_eye": (255, 255, 102),
    "right_leg": (205, 133, 63),
    "right_wing_tip": (30, 144, 255),
    "right_wrist": (50, 205, 50),
    "tail_tip": (0, 255, 127),
    "throat": (255, 20, 147),
    "neck": (0, 191, 255),
    "tail_left": (218, 112, 214),
    "tail_right": (255, 165, 0),
    "upper_spine": (32, 178, 170),
    "upper_half_spine": (0, 128, 128),
    "lower_half_spine": (135, 206, 235),
    "right_foot": (255, 69, 0),
    "left_foot": (128, 128, 0),
    "left_half_chest": (233, 150, 122),
    "right_half_chest": (220, 20, 60),
    "chin": (127, 255, 0),
    "left_tibia": (72, 61, 139),
    "right_tibia": (60, 179, 113),
    "lower_spine": (106, 90, 205),
    "upper_half_neck": (199, 21, 133),
    "lower_half_neck": (210, 105, 30),
    "left_chest": (123, 104, 238),
    "right_chest": (85, 107, 47),
    "upper_neck": (47, 79, 79),
    "left_wing_shoulder": (188, 143, 143),
    "left_wing_elbow": (0, 255, 255),
    "right_wing_shoulder": (255, 20, 147),
    "right_wing_elbow": (105, 105, 105),
    "upper_cere": (0, 100, 0),
    "lower_cere": (100, 149, 237),
}

QUADRUPED_COLOR_MAP = {
    "nose": (255, 0, 0),
    "upper_jaw": (0, 255, 0),
    "lower_jaw": (0, 0, 255),
    "mouth_end_right": (255, 255, 0),
    "mouth_end_left": (255, 0, 255),
    "right_eye": (0, 255, 255),
    "right_earbase": (128, 0, 0),
    "right_earend": (0, 128, 0),
    "right_antler_base": (0, 0, 128),
    "right_antler_end": (128, 128, 0),
    "left_eye": (128, 0, 128),
    "left_earbase": (0, 128, 128),
    "left_earend": (192, 192, 192),
    "left_antler_base": (128, 128, 128),
    "left_antler_end": (64, 64, 64),
    "neck_base": (255, 128, 0),
    "neck_end": (128, 255, 0),
    "throat_base": (0, 255, 128),
    "throat_end": (0, 128, 255),
    "back_base": (255, 0, 128),
    "back_end": (255, 128, 128),
    "back_middle": (128, 255, 255),
    "tail_base": (128, 0, 64),
    "tail_end": (64, 0, 128),
    "front_left_thai": (128, 64, 0),
    "front_left_knee": (64, 128, 0),
    "front_left_paw": (0, 64, 128),
    "front_right_thai": (255, 64, 64),
    "front_right_knee": (64, 255, 64),
    "front_right_paw": (64, 64, 255),
    "back_left_paw": (255, 255, 64),
    "back_left_thai": (255, 64, 255),
    "back_right_thai": (64, 255, 255),
    "back_left_knee": (192, 64, 192),
    "back_right_knee": (192, 192, 64),
    "back_right_paw": (64, 192, 192),
    "belly_bottom": (192, 192, 192),
    "body_middle_right": (128, 64, 64),
    "body_middle_left": (64, 128, 128)
}

PRIMATE_COLOR_MAP = {
    "head": (0, 180, 0), # wait
    "neck": (0, 0, 180), # wait
    "nose": (255, 0, 0), # "
    "mouth_front_top": (0, 255, 0), # "upper_jaw"
    "mouth_front_bottom": (0, 0, 255), # "lower_jaw"
    "mouth_back_right": (255, 255, 0), # "mouth_end_right"
    "mouth_back_left": (255, 0, 255), # "mouth_end_left"
    "right_ear": (128, 0, 0), # "right_earbase"
    "left_ear": (0, 128, 128), # "left_earbase": (0, 128, 128),
    "neck": (255, 128, 0), # "neck_base"
    "upper_back": (128, 255, 0), # "neck_end"
    "throat_base": (0, 255, 128), # "throat_base"
    "upper_back": (255, 0, 128), # "back_base"
    "lower_back": (255, 128, 128), # "back_end"
    "torso_mid_back": (128, 255, 255), # "back_middle"
    "root_tail": (128, 0, 64), # "tail_base"
    "end_tail": (64, 0, 128), # "tail_end"
    "left_shoulder": (128, 64, 0), # "front_left_thai"
    "left_elbow": (64, 128, 0), # "front_left_knee"
    "left_hand": (0, 64, 128), # "front_left_paw"
    "right_shoulder": (255, 64, 64), # "front_right_thai"
    "right_elbow": (64, 255, 64), # "front_right_knee"
    "left_foot": (255, 255, 64), # "back_left_paw"
    "left_hip": (255, 64, 255), # "back_left_thai"
    "left_knee": (192, 64, 192), # "back_left_knee"
    "right_knee": (192, 192, 64), # "back_right_knee"
    "right_foot": (64, 192, 192), # "back_right_paw"
    "body_center": (192, 192, 192), #  "belly_bottom"
    "right_hip": (128, 64, 64), # "body_middle_right"`
    "left_hip": (64, 128, 128),  # "body_middle_left"
    "right_hand": (64, 64, 255), # "front_right_paw"
    "left_wrist": (128, 0, 128),
    "right_wrist": (0, 255, 255),
    "forehead": (0, 128, 0),
    "center_hip": (64, 255, 255),
    "left_ankle": (128, 128, 128),
    "right_ankle": (0, 0, 128),
    "mid_tail": (192, 192, 192),
    "mid_end_tail": (0, 128, 255), 
    "right_eye": (0, 255, 255),
    "left_eye": (128, 0, 128),
    # "right_earend": (0, 128, 0),
    # "right_antler_base": (0, 0, 128),
    # "right_antler_end": (128, 128, 0),
    # "left_earend": (192, 192, 192),
    # "left_antler_base": (128, 128, 128),
    # "left_antler_end": (64, 64, 64),
    # "throat_end": (0, 128, 255), 
    # "front_right_paw": (64, 64, 255),
    # "back_right_thai": (64, 255, 255),
}

# Define dataset configurations
DATASET_CONFIGS = {
    "aptv2": {
        "skeleton": [
            [1, 2], [1, 3], [2, 3], [3, 4], [4, 5],
            [4, 6], [6, 7], [7, 8], [4, 9], [9, 10],
            [10, 11], [5, 12], [12, 13], [13, 14],
            [5, 15], [15, 16], [16, 17]
        ],
        "keypoint_mapping": [
            -1, -1,  # 0-1
            0, 1, 2,  # 2-4
            -1, -1, -1, -1, -1, -1,  # 5-10
            3, 5, 8,  # 11-13
            -1, -1, -1, -1,  # 14-17
            6, 9, 7, 10,  # 18-21
            -1, -1,  # 22-23
            11, 14, 4,  # 24-26
            12, 15, 13, 16,  # 27-30
            -1, -1, -1, -1, -1, -1  # 31-36
        ]
    },
    "mit": {
            "skeleton": [
                [1, 2], [3, 4], [1, 3], [3, 13], [13, 14],
                [14, 5], [5, 8], [8, 10], [6, 9], [5, 6],
                [13, 7], [7, 12], [13, 5], [5, 11]
            ],
    "keypoint_mapping": None,  # No mapping needed for PFM format
    "keypoints": [
                "Front",
                "Right",
                "Middle",
                "Left",
                "FL1",
                "BL1",
                "FR1",
                "BR1",
                "BL2",
                "BR2",
                "FL2",
                "FR2",
                "Body1",
                "Body2",
                "Body3"
            ]
    },
    "pfm": {
        "skeleton": PFM_SKELETON,
        "keypoint_mapping": None  # No mapping needed for PFM format
    }
}

def get_dataset_config(image_id, images):
    """
    Get dataset configuration based on image_id
    Args:
        image_id: ID of the image
        images: list of image information
    Returns:
        dataset configuration dictionary
    """
    # Find image info
    image_info = next((img for img in images if img["id"] == image_id), None)
    if not image_info:
        return DATASET_CONFIGS["pfm"]
        
    # Get dataset name from image info
    dataset_name = image_info.get("source_dataset", "pfm").lower()
    return DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["pfm"])

def find_connections(pfm_idx, dataset_config):
    
    """
    Find all connections for a keypoint in PFM format
    Args:
        pfm_idx: index in PFM format
        keypoints: array of keypoint coordinates
        dataset_config: configuration of the dataset
    Returns:
        list of connected keypoint indices in PFM format
    """
    
    mapping = dataset_config["keypoint_mapping"]
    if mapping is None or pfm_idx >= len(mapping):
        return []
        
    # Get original dataset index for this PFM keypoint
    orig_idx = mapping[pfm_idx]
    if orig_idx == -1:  # Skip if this keypoint doesn't exist in original format
        return []
        
    # Look for all connections in original dataset skeleton
    connected_pfm_indices = []
    for [idx1, idx2] in dataset_config["skeleton"]:
        idx1 -= 1  # Convert to 0-based indexing
        idx2 -= 1
        # Check both directions of connection
        target_idx = None
        if idx1 == orig_idx:
            target_idx = idx2
            
        # Find corresponding PFM index
        for pfm_i, orig_i in enumerate(mapping):
            # Check if this keypoint exists in both formats
            if orig_i == target_idx:
                connected_pfm_indices.append(pfm_i)
                    
    return connected_pfm_indices

def compute_brightness(img, x, y, radius=20):
    crop = img[
        max(0, y - radius) : min(img.shape[0], y + radius),
        max(0, x - radius) : min(img.shape[1], x + radius),
        :,
    ]
    return np.mean(crop)

def get_contrasting_color(bg_color):
    # Calculate perceived luminance
    luminance = (0.299 * bg_color[2] + 0.587 * bg_color[1] + 0.114 * bg_color[0]) / 255
    if luminance > 0.5:
        return (0, 0, 0)  # Use black text
    else:
        return (255, 255, 255)  # Use white text
    
def visualize_annotation(img, annotation, color_map, categories, skeleton, image_id, annotation_id=None, dataset_config=None):
    # Bounding box visualization
    bbox = annotation["bbox"]
    x1, y1, width, height = bbox
    x2, y2 = x1 + width, y1 + height
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    print("________________________")
    
    id = annotation["id"]
    if "keypoints" in annotation:
        keypoints = np.array(annotation["keypoints"]).reshape(-1, 3)
        keypoint_names = categories[0]["keypoints"]  # Get keypoint names from categories
        # keypoint_names = categories  # Get keypoint names from categories
         
        # Calculate scaling factor based on image size
        img_height, img_width = img.shape[:2]
        scale_factor = max(img_width, img_height) / 1000
        
        # print("scale_factor:", scale_factor)
        # Set minimum and maximum limits for scaling factor
        # scale_factor = max(0.5, min(scale_factor, 2))
        
        existing_text_positions = []  # Track existing text positions to avoid overlap
        
        for i, (x_kp, y_kp, v) in enumerate(keypoints):
            if v > 0:
                keypoint_label = keypoint_names[i]
                # draw the keypoint
                cv2.circle(
                    img,
                    center=(int(x_kp), int(y_kp)),
                    radius=int(7 * scale_factor),
                    color=color_map[keypoint_label],
                    thickness=-1,
                )
                
                # bright = compute_brightness(img, int(x1), int(y1))
                # txt_color = (10, 10, 10) if bright > 128 else (255, 255, 255)
                # txt_color = (10, 10, 10) if bright > 128 else (235, 235, 215)
               
                # Get the background color at the text position
                # print("x_kp:", x_kp, "y_kp:", y_kp)
                # print("img.shape:", img.shape)
                # print("image_id", image_id)
                # print("id", id)
                
                bg_color = img[int(y_kp), int(x_kp)].astype(int)
                txt_color = get_contrasting_color(bg_color)
                
                # adjust font scale and thickness based on scale factor
                font_scale = max(0.2, min(scale_factor, 1))*0.8
                thickness = max(1, int(scale_factor))
                
                y_text = int(y_kp) - int(15 * scale_factor)
                x_text = int(x_kp) - int(10 * scale_factor)
                # Ensure text does not go out of image bounds
                x_text = min(max(0, x_text), img_width - 100)
                y_text = min(max(0, y_text), img_height - 10)
            
                # Avoid overlapping text: Adjust position if it overlaps with previously drawn text
                for (existing_x, existing_y) in existing_text_positions:
                    if abs(x_text - existing_x) < 50 and abs(y_text - existing_y) < 20:
                        y_text += int(20 * scale_factor)  # Move text slightly downward if overlap detected
                # Record this position
                existing_text_positions.append((x_text, y_text))
                
                cv2.putText(
                    img,
                    keypoint_label,
                    (int(x_kp), y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    txt_color,
                    thickness + 0,
                    cv2.LINE_AA,
                )
                
                # Draw the black text as an outline
                # cv2.putText(
                #     img,
                #     keypoint_label,
                #     (int(x_kp), y_text),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     font_scale,
                #     (0, 0, 0),  # Black text as the outline
                #     thickness + 2,  # The outline is thicker than the main text
                #     cv2.LINE_AA,
                # )
                
                # # Draw the white text on top
                # cv2.putText(
                #     img,
                #     keypoint_label,
                #     (int(x_kp), y_text),
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     font_scale,
                #     txt_color,  # The original white text
                #     thickness,
                #     cv2.LINE_AA,
                # )
                # Draw skeleton
                # for connection in skeleton:
                #     idx1, idx2 = connection
                #     x1_kp, y1_kp, v1 = keypoints[idx1 - 1]
                #     x2_kp, y2_kp, v2 = keypoints[idx2 - 1]
                #     if v1 > 0 and v2 > 0:
                #         keypoint_label1 = keypoint_names[idx1 - 1]
                #         color = color_map.get(keypoint_label1, (255, 255, 255))
                #         cv2.line(
                #             img,
                #             (int(x1_kp), int(y1_kp)),
                #             (int(x2_kp), int(y2_kp)),
                #             color,
                #             thickness=int(2 * scale_factor)
                #         )        
                 
                # Draw skeleton in original format
                connected_pfm_indices = find_connections(i, dataset_config)
                for pfm_idx in connected_pfm_indices:
                    x2_kp, y2_kp, v2 = keypoints[pfm_idx]
                    if v2 > 0:
                        cv2.line(img, (int(x_kp), int(y_kp)), (int(x2_kp), int(y2_kp)), (0, 255, 0), 2)
                        
    return img

def load_annotation_data(file):
    content = file.read()
    return json.loads(content)

def update_annotation_data(data, index, verified=True):
    data["annotations"][index]["verified"] = verified
    return data

def main():
    st.title("Annotation Verifier")

    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

    if 'color_map' not in st.session_state:
        # st.session_state.color_map = TOPVIEWMOUSE_COLOR_MAP
        st.session_state.color_map = PRIMATE_COLOR_MAP

    color_map_option = st.selectbox(
        "Select Color Map",
        [ "Primate", "Topview Mouse", "Bird", "Quadruped"]
    )

    if color_map_option == "Topview Mouse":
        st.session_state.color_map = TOPVIEWMOUSE_COLOR_MAP
    elif color_map_option == "Bird":
        st.session_state.color_map = BIRD_COLOR_MAP
    elif color_map_option == "Quadruped":
        st.session_state.color_map = QUADRUPED_COLOR_MAP    
    elif color_map_option == "Primate":
        st.session_state.color_map = PRIMATE_COLOR_MAP
        st.session_state.skeleton = PFM_SKELETON    
        
    # File uploader for annotation JSON
    # annotation_file = st.file_uploader("Upload annotation JSON file", type="json")
   
    # with open("/home/ti_wang/Ti_workspace/PrimatePose/data/splitted_val_datasets/chimpact_val_sampled_500.json", "r") as f:
    #     annotation_file = json.load(f)
        
    with open("/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_test_datasets/aptv2_test.json", "r") as f:
        annotation_file = json.load(f)
        
    if annotation_file is not None:
        # Load data only if a new file is uploaded
        if st.session_state.data is None:
            try:
                # st.session_state.data = load_annotation_data(annotation_file)
                st.session_state.data = annotation_file
                st.success("Annotation file loaded successfully!")
            except json.JSONDecodeError:
                st.error("Error: Invalid JSON file. Please upload a valid JSON file.")
                return

        # Image directory input
        image_dir = st.text_input("Enter the path to the image directory:")
        
        # image_dir = "/mediaPFM/data/datasets/final_datasets/v7/test"
        
        image_dir = "/mnt/data/tiwang/v8_coco/images"
        
        if image_dir and os.path.isdir(image_dir):
            # Navigation and verification
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Previous"):
                    st.session_state.current_index = (st.session_state.current_index - 1) % len(st.session_state.data["annotations"])
            with col2:
                if st.button("Next"):
                    st.session_state.current_index = (st.session_state.current_index + 1) % len(st.session_state.data["annotations"])
            with col3:
                if st.button("Verify Annotation"):
                    st.session_state.data = update_annotation_data(st.session_state.data, st.session_state.current_index)
                    st.success(f"Annotation {st.session_state.current_index} verified!")

            # Display current annotation
            annotation = st.session_state.data["annotations"][st.session_state.current_index]

            images = st.session_state.data['images']
            # print("images:", images)
            
            imageid2dataset = {image['id']: image['source_dataset'] for image in images}
            # imageid2dataset = {image['id']: image['dataset_id'] for image in images}
            
            image_id = annotation["image_id"]
            image_info = [img for img in st.session_state.data["images"] if img["id"] == image_id][0]
            image_path = os.path.join(image_dir, image_info["file_name"])

            if os.path.isfile(image_path):
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Get dataset configuration
                dataset_config = get_dataset_config(
                    image_id=image_id,
                    images=st.session_state.data["images"]
                )
                
                # Visualize annotation 
                img_with_annotation = visualize_annotation(
                    img.copy(), 
                    annotation, 
                    st.session_state.color_map, 
                    st.session_state.data["categories"], 
                    st.session_state.skeleton,
                    image_id,
                    dataset_config=dataset_config
                )
                # img_with_annotation = visualize_annotation(img.copy(), annotation, st.session_state.color_map, st.session_state.data["pfm_keypoints"])
             
                # Display image with annotation
                st.image(img_with_annotation, caption=f"{imageid2dataset[image_id]}   Image {st.session_state.current_index + 1}/{len(st.session_state.data['annotations'])}", use_column_width=True)
                
                # Display verification status
                if annotation.get("verified", False):
                    st.info("This annotation has been verified.")
                else:
                    st.warning("This annotation has not been verified yet.")
                
                # Display progress
                verified_count = sum(1 for ann in st.session_state.data["annotations"] if ann.get("verified", False))
                progress = verified_count / len(st.session_state.data["annotations"])
                st.progress(progress)
                st.text(f"Verified {verified_count} out of {len(st.session_state.data['annotations'])} annotations.")

            else:
                st.error(f"Image file not found: {image_path}")
                print(f"Image file not found: {image_path}")
        else:
            st.error("Please enter a valid image directory path.")
    else:
        st.info("Please upload an annotation JSON file to start.")

    # Add a download button for the modified data
    if st.session_state.data is not None:
        json_str = json.dumps(st.session_state.data, indent=2)
        st.download_button(
            label="Download modified annotation file",
            data=json_str,
            file_name="modified_annotations.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()