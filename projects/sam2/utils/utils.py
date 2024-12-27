import json
import numpy as np
from typing import Dict, Any, List
import os.path as osp
from pathlib import Path


def read_bbox(bbox_path):
    """Read initial bbox from file"""
    with open(bbox_path, 'r') as f:
        x, y, w, h = map(float, f.read().strip().split(','))
        return [int(x), int(y), int(w), int(h)]


def load_json_data(json_path: str) -> Dict[str, Any]:
    """Load and return the JSON data"""
    with open(json_path, 'r') as f:
        return json.load(f)

def convert_bbox_to_coco(bbox):
    """Convert bbox from [x_min, y_min, x_max, y_max] to COCO format [x, y, w, h]"""
    x_min, y_min, x_max, y_max = bbox
    return np.array([[
        x_min,
        y_min,
        x_max - x_min,  # width
        y_max - y_min   # height
    ]])
    

def get_video_data(json_data: Dict[str, Any], video_id: int) -> tuple[List[Dict], List[Dict]]:
    """
    Extract images and annotations for a specific video_id
    Returns:
        tuple: (video_images, video_annotations)
    """
    # Filter images and annotations by video_id
    video_images = [img for img in json_data['images'] if img['video_id'] == video_id]
    video_annotations = [ann for ann in json_data['annotations'] if ann['video_id'] == video_id]
    
    # Sort images by frame number (extracted from file_name)
    video_images.sort(key=lambda x: int(x['file_name'].split('/')[-1].split('.')[0]))
    
    return video_images, video_annotations

def get_frame_path(image_data: Dict, base_image_path: str) -> str:
    """Construct full frame path from image data"""
    return osp.join(base_image_path, image_data['file_name'])

def save_processed_data_to_json(
    output_dir: Path,
    video_id: int,
    video_images: List[Dict],
    video_annotations: List[Dict],
    predictions: List[List[Dict]],  # List of predictions for each object
    masks: List[List[np.ndarray]],  # List of masks for each object
    bboxes: List[List[np.ndarray]]  # List of bboxes for each object
) -> str:
    processed_data = {
        "video_id": int(video_id),
        "images": [],
        "annotations": [],
        "predictions": []
    }
    
    num_objects = len(predictions)
    
    for frame_idx, (image_data, annotation) in enumerate(zip(video_images, video_annotations)):
        # Save image info
        processed_data["images"].append({
            "id": int(image_data["id"]),
            "file_name": image_data["file_name"],
            "width": int(image_data["width"]),
            "height": int(image_data["height"]),
            "frame_idx": int(frame_idx)
        })
        
        # Save original annotation
        processed_data["annotations"].append({
            "image_id": int(annotation["image_id"]),
            "bbox": [float(x) for x in annotation["bbox"]],
            "keypoints": [float(x) for x in annotation["keypoints"].ravel()] if isinstance(annotation["keypoints"], np.ndarray) else annotation["keypoints"],
            "frame_idx": int(frame_idx)
        })
        
        # Save predictions for each object
        frame_predictions = []
        for obj_id in range(num_objects):
            pred = predictions[obj_id][frame_idx]
            if pred:
                bodyparts = pred["bodyparts"]
                if bodyparts is not None:
                    frame_predictions.append({
                        "object_id": obj_id,
                        "frame_idx": int(frame_idx),
                        "bbox": [float(x) for x in bboxes[obj_id][frame_idx]] if bboxes[obj_id][frame_idx] is not None else None,
                        "keypoints": bodyparts[..., :2].tolist(),
                        "confidences": bodyparts[..., 2].tolist()
                    })
        
        processed_data["predictions"].extend(frame_predictions)
    
    json_path = output_dir / f"videoID_{video_id}_processed.json"
    with open(json_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    return str(json_path)

def save_video_data_to_json(json_data: Dict, video_id: int, output_dir: Path) -> str:
    """Extract and save video-specific data to a new JSON file"""
    video_data = {
        "info": json_data.get("info", {}),
        "licenses": json_data.get("licenses", []),
        "categories": json_data.get("categories", []),
        "images": [],
        "annotations": []
    }
    
    # Get all images for this video
    video_images = [img for img in json_data['images'] if img['video_id'] == video_id]
    video_data['images'] = video_images
    
    # Get all annotations for this video
    image_ids = {img['id'] for img in video_images}
    video_annotations = [ann for ann in json_data['annotations'] 
                        if ann['image_id'] in image_ids]
    video_data['annotations'] = video_annotations
    
    # Save to JSON file
    json_path = output_dir / f"videoID_{video_id}_original.json"
    with open(json_path, 'w') as f:
        json.dump(video_data, f, indent=2)
    
    return str(json_path)