import torch
from PIL import Image
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
import os

# Load model
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Check if checkpoint exists
if not os.path.exists(checkpoint):
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# Load and prepare an image
image_path = "/home/ti_wang/Ti_workspace/projects/sam2/demo/data/monkey.jpg"

# Check if image exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

image = Image.open(image_path)
image = np.array(image)

# Run inference
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(image)
    
    # Example: Add a point prompt in the center of the image
    input_point = np.array([[image.shape[1]//2, image.shape[0]//2]])
    input_label = np.array([1])
    
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label
    ) 
    print("masks.shape", masks.shape)

    # Visualize the results
    plt.figure(figsize=(10, 10))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.plot(input_point[:, 0], input_point[:, 1], 'rx', markersize=10)  # Show the point prompt
    plt.title('Original Image with Point Prompt')
    plt.axis('off')
    
    # Plot image with mask overlay
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    
    # Convert mask to numpy if it's a tensor
    if torch.is_tensor(masks):
        masks = masks.cpu().numpy()
    
    # Show mask overlay (assuming masks[0] is the first/best mask)
    mask = masks[0]
    plt.imshow(mask, alpha=0.5, cmap='jet')
    plt.title('Segmentation Mask Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the visualization
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'segmentation_result.png'))
    plt.show()