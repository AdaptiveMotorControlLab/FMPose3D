# for v8

import streamlit as st
import json
import os
import cv2
import numpy as np
from PIL import Image
import tempfile

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
    
def visualize_annotation(img, annotation, color_map, categories):
    # Bounding box visualization
    bbox = annotation["bbox"]
    x1, y1, width, height = bbox
    x2, y2 = x1 + width, y1 + height
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Keypoint visualization
    if "keypoints" in annotation:
        keypoints = np.array(annotation["keypoints"]).reshape(-1, 3)
        # test
        # keypoint_names = categories[0]["keypoints"]  # Get keypoint names from categories
        keypoint_names = categories  # Get keypoint names from categories
        # except:
        #     print("error")
        #     print(categories[0])     
    
        # Calculate scaling factor based on image size
        img_height, img_width = img.shape[:2]
        scale_factor = max(img_width, img_height) / 1000
        print("scale_factor:", scale_factor)
        # Set minimum and maximum limits for scaling factor
        # scale_factor = max(0.5, min(scale_factor, 2))

        for i, (x_kp, y_kp, v) in enumerate(keypoints):
            if v > 0:
                keypoint_label = keypoint_names[i]
                
                cv2.circle(
                    img,
                    center=(int(x_kp), int(y_kp)),
                    radius=int(7 * scale_factor),
                    color=color_map[keypoint_label],
                    thickness=-1,
                )
                
                bright = compute_brightness(img, int(x1), int(y1))
                # txt_color = (10, 10, 10) if bright > 128 else (255, 255, 255)
                # txt_color = (10, 10, 10) if bright > 128 else (235, 235, 215)
               
                # Get the background color at the text position
                bg_color = img[int(y_kp), int(x_kp)].astype(int)
                print("bg_color:", bg_color)
                txt_color = get_contrasting_color(bg_color)
                
                # adjust font scale and thickness based on scale factor
                font_scale = 0.5 * scale_factor
                thickness = max(1, int(2 * scale_factor))
                y_text = int(y1) - int(15 * scale_factor)
                
                cv2.putText(
                    img,
                    keypoint_label,
                    (int(x_kp), y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    txt_color,
                    thickness + 2,
                    cv2.LINE_AA,
                )
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
    else:
        st.session_state.color_map = PRIMATE_COLOR_MAP    
        
    # File uploader for annotation JSON
    # annotation_file = st.file_uploader("Upload annotation JSON file", type="json")
   
    with open("/home/ti_wang/Ti_workspace/PrimatePose/data/splitted_val_datasets/anipose_val.json", "r") as f:
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
        
        image_dir = "/mnt/tiwang/v8_coco/images"
        
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
            print("images:", images)
            
            imageid2dataset = {image['id']: image['source_dataset'] for image in images}
            # imageid2dataset = {image['id']: image['dataset_id'] for image in images}
            
            image_id = annotation["image_id"]
            image_info = [img for img in st.session_state.data["images"] if img["id"] == image_id][0]
            image_path = os.path.join(image_dir, image_info["file_name"])

            if os.path.isfile(image_path):
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Visualize annotation 
                img_with_annotation = visualize_annotation(img.copy(), annotation, st.session_state.color_map, st.session_state.data["categories"])
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