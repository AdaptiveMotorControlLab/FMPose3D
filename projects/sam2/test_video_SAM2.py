import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2_video_predictor
from sam2.utils.misc import load_video_frames
import os

def read_bbox(bbox_path):
    """Read initial bbox from file"""
    with open(bbox_path, 'r') as f:
        x, y, w, h = map(float, f.read().strip().split(','))
        return [int(x), int(y), int(w), int(h)]
    
def save_video_frames(video_path, frame_dir):
    """
    Load video and save individual frames to the target directory at original size
    
    Args:
        video_path (str): Path to the video file
        frame_dir (str): Directory to save the frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame with just the number as filename
        frame_path = os.path.join(frame_dir, f"{frame_count:05d}.jpg")  # Changed from frame_XXXXX to just XXXXX
        cv2.imwrite(frame_path, frame)
        frame_count += 1
        
        if frame_count % 10 == 0:  # Progress update every 10 frames
            print(f"Saved {frame_count} frames...")
    
    cap.release()
    print(f"Successfully saved {frame_count} frames to {frame_dir}")
    return frame_count

def apply_mask_to_image(image, mask, alpha=0.5):
    """
    Apply a colored mask overlay on the image
    
    Args:
        image: Original image (numpy array)
        mask: Binary mask (torch tensor on GPU)
        alpha: Transparency of the overlay (0-1)
    """
    # Convert image to GPU tensor with correct dtype
    image_tensor = torch.from_numpy(image).cuda().float()
    
    # Create a colored overlay for the mask on GPU
    color_mask = torch.zeros_like(image_tensor)
    green_color = torch.tensor([0, 255, 0], dtype=torch.float32, device='cuda')
    mask_bool = mask.squeeze() > 0.5  # Convert to boolean mask
    
    # Apply the color to masked regions
    color_mask[mask_bool] = green_color
    
    # Combine the image with the colored mask on GPU
    output = (image_tensor + alpha * color_mask).clamp(0, 255)
    
    # Convert back to numpy for cv2.imwrite
    return output.cpu().numpy().astype(np.uint8)

# Load model
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)
print("SAM2 model initialized")

# Set parameters for video loading
image_size = (720, 1280)  # (height, width) - adjust based on your needs
offload_video_to_cpu = False  # Set to True if you want to save GPU memory

# Load video frames

# Load your video - use SAM2's built-in video loader
video_path = "./demo/data/gallery/01_dog.mp4"
video_path = "./demo/data/gallery/dance_15fps.mp4"
ori_frame_dir = ""
target_frame_dir = ""
video_name = os.path.basename(video_path).split('.')[0]
output_folder = os.path.join(os.path.dirname(video_path), video_name)
if ori_frame_dir == "":
    ori_frame_dir = os.path.join(output_folder, "ori_frames")
    print(f"Frame directory will be: {ori_frame_dir}")
if target_frame_dir =="":
    target_frame_dir = os.path.join(output_folder, "target_frames")
    
# Create the directory if it doesn't exist
os.makedirs(ori_frame_dir, exist_ok=True)
os.makedirs(target_frame_dir, exist_ok=True)

# Save video frames
num_frames = save_video_frames(video_path, ori_frame_dir)
print(f"Total frames extracted: {num_frames}")

# prompt
bbox_path = "./bbox.txt" # x1,y1,x2,y2
prompt_bbox = read_bbox(bbox_path)
frame_idx=0

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    # Initialize state with video frames
    state = predictor.init_state(video_path=str(ori_frame_dir))
    masks = []
    
    # add new prompts and instantly get the output on the same frame
    frame_idx, object_ids, mask = predictor.add_new_points_or_box(
        inference_state=state,
        box=prompt_bbox,
        frame_idx=frame_idx,
        obj_id=0   
    )
    masks.extend(mask)

    # Process first frame
    first_frame = cv2.imread(os.path.join(ori_frame_dir, f"{frame_idx:05d}.jpg"))
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    masked_frame = apply_mask_to_image(first_frame, masks[0])
    masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR
    cv2.imwrite(os.path.join(target_frame_dir, f"{frame_idx:05d}.jpg"), masked_frame)

    # propagate the prompts to get masklets throughout the video
    for frame_idx, object_ids, output_masks in predictor.propagate_in_video(state):
        print(f"Processing frame {frame_idx}, tracking {len(object_ids)} objects")
        
        # Read original frame
        frame_path = os.path.join(ori_frame_dir, f"{frame_idx:05d}.jpg")
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
     
        # Apply mask overlay
        if len(output_masks) > 0:
            masked_frame = apply_mask_to_image(frame, output_masks[0])  # Using first mask if multiple objects
            masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR
            
            # Save the masked frame
            output_path = os.path.join(target_frame_dir, f"{frame_idx:05d}.jpg")
            cv2.imwrite(output_path, masked_frame)
            print(f"Saved masked frame to {output_path}")
            
print("Completed processing all frames")
        