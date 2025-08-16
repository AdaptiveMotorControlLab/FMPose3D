# for v8
import streamlit as st
import json
import os
import cv2
import numpy as np
from PIL import Image

# Bring skeletons from module
from skeletons import PFM_SKELETON

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
import plotly.graph_objects as go

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
from datasets import DATASET_CONFIGS
from skeletons import PFM_SKELETON

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
    
def visualize_annotation(img, annotation, color_map, categories, skeleton, image_id, annotation_id=None, dataset_config=None, use_simplified_keypoints=False, dataset_name=None, draw_text_labels=False):
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
        
        # only for riken
        scale_factor = scale_factor * 0.5 
        
        existing_text_positions = []
        
        # Create colormap
        cmap = get_cmap(len(keypoints_simplified), "rainbow")
        
        for i, (x_kp, y_kp, v) in enumerate(keypoints):
            if v > 0:
                # print(dataset_config['keypoint_mapping'])
                # print("i:", i) 
                # print("v:", v)
                # print("idx:", dataset_config['keypoint_mapping'][i])
                # print("x_kp:", x_kp, "y_kp:", y_kp)
                keypoint_label = keypoint_names[dataset_config['keypoint_mapping'][i]]
                # print(keypoint_label)
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
                circle_radius_dict = {"oap": 1, "oms": 1, "omc": 1, "mp": 1, "ap10k": 1, "aptv2": 1, "lote": 1, "ak": 5, "deepwild": 1, "chimpact": 1, "mit": 1, "riken": 1, "mbw": 1, "mit": 1, "pfm": 1}
                circle_radius = circle_radius_dict.get(dataset_name, 1)
                # print("circle_radius:", circle_radius)
                
                cv2.circle(
                    img,
                    center=(int(x_kp), int(y_kp)),
                    radius=int(4 * scale_factor*circle_radius),
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
                    if abs(x_text - existing_x) < 30 and abs(y_text - existing_y) < 20:
                        y_text += int(8 * scale_factor)  # Move text slightly downward if overlap detected
                # Record this position
                existing_text_positions.append((x_text, y_text))
                
                thickness_dict = {"oap": 3,   "oms": 1,   "omc": 1, "mp": 1,     "ap10k": 1,   "aptv2": 1,   "lote": 1,   "ak": 1,   "deepwild": 1,   "chimpact": 1,   "mit": 2,   "riken": 1,   "mbw": 1, "mit": 1, "pfm": 2}
                fontScale_dict = {"oap": 2.6, "oms": 0.9, "omc": 0.9, "mp": 1.3, "ap10k": 0.8, "aptv2": 1.1, "lote": 0.8, "ak": 1.0, "deepwild": 0.6, "chimpact": 0.9, "mit": 1.2, "riken": 0.6, "mbw": 0.6, "mit": 1.4, "pfm": 1.2}
                thickness_dataset = thickness_dict.get(dataset_name, 1)
                fontScale_dataset = fontScale_dict.get(dataset_name, 1.1)

                # scale_factor = 2  # Increase the resolution
                # img_high_res = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
                # img = img_high_res
                if draw_text_labels:
                    cv2.putText(
                        img=img,
                        text=keypoint_label,
                        org=(int(x_kp), y_text),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale*fontScale_dataset,
                        color=color_bgr,
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


def build_interactive_figure(img, annotation, dataset_config, dataset_name=None, use_simplified_keypoints=False):
    img_height, img_width = img.shape[:2]
    keypoints = np.array(annotation.get("keypoints", [])).reshape(-1, 3)

    # Determine keypoint names
    if "keypoints" in dataset_config and dataset_config["keypoints"] is not None:
        keypoint_names = dataset_config['keypoints_simplified'] if (use_simplified_keypoints and dataset_config.get('keypoints_simplified') is not None) else dataset_config['keypoints']
    else:
        keypoint_names = keypoints_simplified

    # Colors
    cmap = get_cmap(len(keypoints_simplified), "rainbow")

    x_vals, y_vals, hover_texts, colors = [], [], [], []
    for i, (x_kp, y_kp, v) in enumerate(keypoints):
        if v > 0:
            # Label mapping
            kp_idx = dataset_config['keypoint_mapping'][i] if dataset_config.get('keypoint_mapping') is not None else i
            if kp_idx != -1 and kp_idx < len(keypoint_names):
                label = keypoint_names[kp_idx]
            else:
                label = f"kp_{i}"

            x_vals.append(float(x_kp))
            y_vals.append(float(y_kp))
            hover_texts.append(label)
            rgb = cmap(i)[:3]
            colors.append(f"rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})")

    fig = go.Figure()

    # Use an image trace so Streamlit renders the bitmap reliably
    fig.add_trace(go.Image(z=img))

    # Skeleton lines (if available)
    if dataset_config.get("skeleton") is not None and len(keypoints) > 0:
        for i, (x_kp, y_kp, v) in enumerate(keypoints):
            if v <= 0:
                continue
            connected_pfm_indices = find_connections(i, dataset_config)
            for pfm_idx in connected_pfm_indices:
                x2, y2, v2 = keypoints[pfm_idx]
                if v2 > 0:
                    fig.add_shape(
                        type="line",
                        x0=float(x_kp), y0=float(y_kp), x1=float(x2), y1=float(y2),
                        line=dict(color="rgba(0,255,0,0.8)", width=2),
                        layer="above",
                    )

    # Bounding box
    bbox = annotation.get("bbox")
    if bbox is not None:
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        fig.add_shape(
            type="rect",
            x0=float(x1), y0=float(y1), x1=float(x2), y1=float(y2),
            line=dict(color="rgba(0,255,0,0.9)", width=2),
            fillcolor="rgba(0,0,0,0)",
        )

    # Keypoints (hoverable)
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            marker=dict(
                color=colors,
                size=8,
                line=dict(width=0),
                symbol="circle"
            ),
        )
    )

    # Use image coordinate system: origin at top-left (reverse y-axis)
    fig.update_xaxes(visible=False, range=[0, img_width])
    fig.update_yaxes(visible=False, range=[img_height, 0], scaleanchor="x", scaleratio=1, autorange=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), dragmode=False)

    return fig

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
        st.session_state.color_map = PRIMATE_COLOR_MAP

    color_map_option = st.selectbox(
        "Select Color Map",
        ["Primate"]
    )
 
    if color_map_option == "Primate":
        st.session_state.color_map = PRIMATE_COLOR_MAP
        st.session_state.skeleton = PFM_SKELETON    
        
    # Add input fields for dataset name and mode
    # dataset_name = st.text_input("Enter Dataset Name:")
    # mode = st.radio("Select Mode:", options=["train", "val", "test"], index=2)  # Default to 'test'

    # Add a button to load the annotation file
    # if st.button("Load Annotation File"):
    #     annotation_file_path = f"/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_{mode}_datasets/{dataset_name}_{mode}.json"
    #     with open(annotation_file_path, "r") as f:
    #         annotation_file = json.load(f)
    #     st.session_state.data = annotation_file
    #     st.success("Annotation file loaded successfully!")
            
    # todo: add a button to input dataset name, and model
    dataset_name = "oms"
    mode = "test"
    annotation_file_path = f"/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_{mode}_datasets/{dataset_name}_{mode}.json"
    # annotation_file_path = f"/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_test_datasets/mit_train_test.json"
    # annotation_file_path = "/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/PFM_V8.2/splitted_test_datasets/ap10k_test.json"
    with open(annotation_file_path, "r") as f:
        annotation_file = json.load(f)
    
    # with open("/home/ti_wang/Ti_workspace/PrimatePose/data/tiwang/primate_data/splitted_val_datasets/omc_val.json", "r") as f:
        # annotation_file = json.load(f)
            
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

        # image_dir = "/mnt/data/tiwang/v8_coco/images"
        image_dir = "/home/ti_wang/data/tiwang/v8_coco/images"
        
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
             
                # Interactive hoverable keypoints (no text drawn on the image)
                fig = build_interactive_figure(
                    img=img,
                    annotation=annotation,
                    dataset_config=dataset_config,
                    dataset_name=dataset_name,
                    use_simplified_keypoints=True,
                )

                image_name = image_info['file_name'].split('.')[-2]
                image_type = image_info['file_name'].split('.')[-1]

                tab_interactive, tab_image = st.tabs(["Interactive", "Image"])
                with tab_interactive:
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    st.caption(f"{image_name}, annotation_id: {annotation['id']}  Image {st.session_state.current_index + 1}/{len(st.session_state.data['annotations'])}")
                with tab_image:
                    st.image(
                        img,
                        caption=f"{image_name}, annotation_id: {annotation['id']}  Image {st.session_state.current_index + 1}/{len(st.session_state.data['annotations'])}",
                        use_container_width=True,
                    )
                
                # Add download button for the current image                
                # Create four columns for the buttons
                col1, col2, col3, col4 = st.columns(4)

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
                with col4:
                    if st.button("Save Original Image"):
                        # Define the directory path
                        save_dir = f"/home/ti_wang/data/tiwang/st_original_images/{imageid2dataset[image_id]}"
                        # Create the directory if it doesn't exist
                        os.makedirs(save_dir, exist_ok=True)
                        # Define the file path
                        save_path = os.path.join(save_dir, f"original_{image_name}_anoID_{annotation['id']}.{image_type}")
                        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])
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