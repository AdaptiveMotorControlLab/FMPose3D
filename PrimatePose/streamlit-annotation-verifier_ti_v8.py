# for v8
import streamlit as st
import json
import os
import cv2
import numpy as np
from PIL import Image

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
}

primate_color_list = list(PRIMATE_COLOR_MAP.values())

import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

def get_cmap(n: int, name: str = "rainbow") -> Colormap:
    """
    Get a matplotlib colormap with n distinct colors.
    
    Args:
        n: number of distinct colors
        name: name of matplotlib colormap

    Returns:
        A function that maps each index in 0, 1, ..., n-1 to a distinct RGB color
    """
    return plt.cm.get_cmap(name, n)

# "left" → "L"
# "right" → "R"
# "mid" → "M"
# "back" → "B"
# "center" → "C"
keypoints_simplified = [
    "forehead", 
    "head",
    "L_eye",
    "R_eye",
    "nose",
    "L_E",
    "R_E",
    "mouth_front_top",
    "mouth_front_bottom",
    "mouth_B_L",
    "mouth_B_R",
    "neck",
    "L_S",
    "R_S",
    "upper_B",
    "torso_M_B",
    "body_C",
    "lower_B",
    "L_elbow",
    "R_elbow",
    "L_wrist",
    "R_wrist",
    "L_hand",
    "R_hand",
    "L_hip",
    "R_hip",
    "C_hip",
    "L_knee",
    "R_knee",
    "L_ankle",
    "R_ankle",
    "L_foot",
    "R_foot",
    "root_tail",
    "M_tail",
    "M_end_tail",
    "end_tail"
]
            
# Define dataset configurations
DATASET_CONFIGS = {
    "oap": {
        "skeleton": None,
        "keypoint_mapping": [
            -1, 3, 1, 2, 0, -1, -1, 
            -1, -1, -1, -1, 4, 5, 8, 
            -1, -1, -1, -1, 6, 9, 7, 
            10, -1, -1, -1, -1, 11, 12, 14, 
            13, 15, -1, -1, -1, -1, -1, -1
        ],
        "keypoints": [
                "nose",
                "left_eye",
                "right_eye",
                "head",
                "neck",
                "left_shoulder",
                "left_elbow",
                "left_wrist",                                
                "right_shoulder",
                "right_elbow",
                "right_wrist",
                "hip/sacrum",
                "left_knee",
                "left_foot",
                "right_knee",
                "right_foot"
            ],
        "keypoints_simplified": None
    },
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
        ],
        "keypoints": [
            "left_eye",
            "right_eye",
            "nose",
            "neck",
            "root_of_tail",
            "left_shoulder",
            "left_elbow",
            "left_front_paw",
            "right_shoulder",
            "right_elbow",
            "right_front_paw",
            "left_hip",
            "left_knee",
            "left_back_paw",
            "right_hip",
            "right_knee",
            "right_back_paw"
        ],
        "keypoints_simplified": [
            "L_eye",
            "R_eye",
            "nose",
            "neck",
            "R_tail",
            "L_S",
            "L_elbow",
            "L_F_paw",
            "R_S",
            "R_elbow",
            "RF_paw",
            "L_hip",
            "L_knee",
            "LB_paw",
            "R_hip",
            "R_knee",
            "RB_paw"
        ],
    },
    "mit": {
            "skeleton": [
                [1, 2], [3, 4], [1, 3], [3, 13], [13, 14],
                [14, 5], [5, 8], [8, 10], [6, 9], [5, 6],
                [13, 7], [7, 12], [13, 5], [5, 11]
            ],
    "keypoint_mapping": [
            0, 3, -1, -1, -1, 1, 2, -1, -1, -1,
            -1, -1, 7, 9, 4, -1, -1, 5, -1, -1,
            8, 10, -1, -1, 11, 13, -1, -1, -1, -1,
            -1, 12, 14, 6, -1, -1, -1
    ],  # No mapping needed for PFM format
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
    "oms": {
        "skeleton": None,
        "keypoint_mapping": [
            -1, 1, -1, -1, 0, -1, -1,
            -1, -1, -1, -1, 2, 5, 3,
            -1, -1, -1, -1, -1, -1, -1,
            -1, 6, 4, -1, -1, 7, 10, 8,
            -1, -1, 11, 9, -1, -1, -1, 12
            ],
        "keypoints": [
            "nose",
            "head",
            "neck",
            "right_shoulder",
            "right_hand",
            "left_shoulder",
            "left_hand",
            "hip",
            "right_knee",
            "right_foot",            
            "left_knee",
            "left_foot",
            "tail"],
        "keypoints_simplified": [
            "nose",
            "head",
            "neck",
            "R_S",
            "R_hand",
            "L_S",
            "L_hand",
            "hip",
            "R_knee",
            "R_foot",            
            "L_knee",
            "L_foot",
            "tail"]
    },
    "riken": {
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
    return DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["pfm"]), dataset_name

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
    
def visualize_annotation(img, annotation, color_map, categories, skeleton, image_id, annotation_id=None, dataset_config=None, use_simplified_keypoints=False, dataset_name=None):
    # Bounding box visualization
    bbox = annotation["bbox"]
    if bbox is not None:
        x1, y1, width, height = bbox
        x2, y2 = x1 + width, y1 + height
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    print("________________________")
    
    if "keypoints" in annotation:
        keypoints = np.array(annotation["keypoints"]).reshape(-1, 3)
        
        # print("dataset_config:", dataset_config)
        if 'keypoints' in dataset_config and dataset_config["keypoints"] is not None:
            # print("keypoint in dataset_config")
            if use_simplified_keypoints and dataset_config['keypoints_simplified'] is not None:
                keypoint_names = dataset_config['keypoints_simplified']  # Get simplified keypoint names from dataset config
            else:
                keypoint_names = dataset_config['keypoints']  # Get keypoint names from categories; pfm;
        else:
            keypoint_names = keypoints_simplified
        
        # Calculate scaling factor based on image size
        img_height, img_width = img.shape[:2]
        
        scale_factor = max(img_width, img_height) / 1000
        
        existing_text_positions = []
        
        # Create colormap
        cmap = get_cmap(len(keypoints_simplified), "rainbow")
        
        for i, (x_kp, y_kp, v) in enumerate(keypoints):
            if v > 0:
                # print(dataset_config['keypoint_mapping'])
                print("i:", i) 
                print("v:", v)
                print("idx:", dataset_config['keypoint_mapping'][i])
                print("x_kp:", x_kp, "y_kp:", y_kp)
                keypoint_label = keypoint_names[dataset_config['keypoint_mapping'][i]]
                print(keypoint_label)
                # Get color from colormap and convert to OpenCV BGR format
                color_rgb = cmap(i)[:3]  # Get RGB values (ignore alpha)
                color_bgr = tuple(int(c * 255) for c in color_rgb[::-1])  # Convert to BGR
                # print(dataset_config['keypoint_mapping'][i])
                # print(color_map)
                # use the primate_color
                # color = primate_color_list[dataset_config['keypoint_mapping'][i]]
                # color_bgr = color
                
                # print("color:", color)
                # draw the keypoint
                cv2.circle(
                    img,
                    center=(int(x_kp), int(y_kp)),
                    radius=int(7 * scale_factor),
                    # color=color_map[keypoint_label],
                    color=color_bgr,
                    thickness=-1,
                )
                # print("x_kp:", x_kp, "y_kp:", y_kp)
                # bg_color = img[int(y_kp), int(x_kp)].astype(int)
                # txt_color = get_contrasting_color(bg_color)
                
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
                
                thickness_dict = {"oap": 2, "oms": 1, "aptv2": 1, "mit": 2, "riken": 2, "pfm": 2}
                fontScale_dict = {"oap": 1.6, "oms": 0.9, "aptv2": 1.1, "mit": 1.2, "riken": 1.2, "pfm": 1.2}
                thickness_dataset = thickness_dict.get(dataset_name, 1)
                fontScale_dataset = fontScale_dict.get(dataset_name, 1.1)
                 
                # Draw colored text matching the keypoint color
                cv2.putText(
                    img=img,
                    text=keypoint_label,
                    # keypoint_idx,  # Use index instead of name
                    org=(int(x_kp), y_text),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=font_scale*fontScale_dataset,
                    color=color_bgr,  # Use same color as keypoint
                    thickness=thickness_dataset,
                    lineType=cv2.LINE_AA,
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
                if dataset_config["skeleton"] is not None:
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
        [ "Primate", "Topview Mouse"]
    )
 
    if color_map_option == "Primate":
        st.session_state.color_map = PRIMATE_COLOR_MAP
        st.session_state.skeleton = PFM_SKELETON    
        
    # File uploader for annotation JSON
    # annotation_file = st.file_uploader("Upload annotation JSON file", type="json")
   
    # with open("/home/ti_wang/Ti_workspace/PrimatePose/data/splitted_val_datasets/chimpact_val_sampled_500.json", "r") as f:
    #     annotation_file = json.load(f)
    
    with open("/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_test_datasets/oap_test.json", "r") as f:
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

        image_dir = "/mnt/data/tiwang/v8_coco/images"
        
        images = st.session_state.data['images']
        # imageid2dataset_name 
        imageid2dataset = {image['id']: image['source_dataset'] for image in images}
        # imageid2dataset = {image['id']: image['dataset_id'] for image in images}
        
        # Create a dictionary mapping image_id to file_name
        image_id_to_image = {image['id']: image for image in images}
        st.session_state.image_id2image = image_id_to_image

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

            # Display the current annotation with index
            annotation = st.session_state.data["annotations"][st.session_state.current_index]
            # keypoint = st.session_state.data["categories"][0][""]
            
            # todo optimize
            image_id = annotation["image_id"]
            # image_info = [img for img in st.session_state.data["images"] if img["id"] == image_id][0]
            image_info = st.session_state.image_id2image[image_id]
            image_path = os.path.join(image_dir, image_info["file_name"])
            print("image_path:", image_path)

            if os.path.isfile(image_path):
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Get dataset configuration
                # TODO: rewrite this function, get the dataset config from json_file_path
                dataset_config, dataset_name = get_dataset_config(
                    image_id=image_id,
                    images=st.session_state.data["images"]
                )
                
                # Visualize annotation 
                img_with_annotation = visualize_annotation(
                    img = img.copy(), 
                    annotation =  annotation, 
                    color_map = st.session_state.color_map, 
                    categories = st.session_state.data["categories"], 
                    skeleton = st.session_state.skeleton,
                    image_id = image_id,
                    dataset_config = dataset_config,
                    use_simplified_keypoints = True,
                    dataset_name = dataset_name
                )
            
                # img_with_annotation = visualize_annotation(img.copy(), annotation, st.session_state.color_map, st.session_state.data["pfm_keypoints"])
             
                # Display image with annotation
                
                quality = "High"
                # Convert quality settings to width
                quality_width = {
                    "Low": 800,
                    "Medium": 1200,
                    "High": 1600,
                    "Original": None  # None means use original size
                }
                
                # Display image with selected quality
                image_name = image_info['file_name'].split('.')[-2]
                image_type = image_info['file_name'].split('.')[-1]
                st.image(
                    img_with_annotation, 
                    caption=f"{image_name}, annotation_id: {annotation['id']}  Image {st.session_state.current_index + 1}/{len(st.session_state.data['annotations'])}", 
                    use_container_width=True,
                    width=quality_width[quality]
                )
                
                # Add download button for the current image                
                # Create three columns for the buttons
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("Save Annotated Image"):
                        # Define the directory path
                        save_dir = f"/home/ti_wang/data/tiwang/st_saved_images/{imageid2dataset[image_id]}"
                        # Create the directory if it doesn't exist
                        os.makedirs(save_dir, exist_ok=True)
                        # Define the file path
                        save_path = os.path.join(save_dir, f"{image_name}_anoID_{annotation['id']}.{image_type}")
                        # Save the image with high quality
                        cv2.imwrite(save_path, cv2.cvtColor(img_with_annotation, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        st.success(f"Image saved successfully at {save_path}")

                with col2:
                    if st.button("Save Wrong Annotated Image"):
                        # Define the directory path
                        save_dir = f"/home/ti_wang/data/tiwang/st_wrong_images/{imageid2dataset[image_id]}"
                        # Create the directory if it doesn't exist
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"wrong_{image_name}_anoID_{annotation['id']}.{image_type}")
                        cv2.imwrite(save_path, cv2.cvtColor(img_with_annotation, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        st.success(f"Image saved successfully at {save_path}")
                
                with col3:
                    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(img_with_annotation, cv2.COLOR_RGB2BGR))
                    if is_success:

                        print("image_name:", image_name)
                        btn = st.download_button(
                            label="Download annotated image",
                            data=buffer.tobytes(),
                            file_name=f"{image_name}_anoID_{annotation['id']}.{image_type}",
                            mime="image/png"
                        )
                    
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