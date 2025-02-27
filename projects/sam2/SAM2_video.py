import torch
import numpy as np
import cv2
import os
from sam2.build_sam import build_sam2_video_predictor

def create_video_from_frames(frames_dir, output_dir, fps=15):
    """Create a video from a sequence of frame images
    
    Args:
        frames_dir (str): Directory containing the frame images (format: XXXXX.jpg)
        output_dir (str): Directory to save the output video
        fps (int, optional): Frames per second for output video. Defaults to 15.
    
    Returns:
        str: Path to the created video file (output.mp4)
        
    Raises:
        ValueError: If no frames are found in frames_dir
    """
    # Get list of frame files and sort them
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
    frame_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    
    if not frame_files:
        raise ValueError(f"No frames found in {frames_dir}")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width = first_frame.shape[:2]
    
    # Create output video file
    output_path = os.path.join(output_dir, "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames to video
    print("Creating video from frames...")
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    out.release()
    print(f"Video saved to: {output_path}")
    return output_path

class SAM2VideoSegmentor:
    """Video segmentation class using SAM2 model
    
    This class handles video segmentation tasks including:
    - Frame extraction from video
    - Mask generation using SAM2
    - Visualization of results
    - Saving masks and visualizations
    """

    def __init__(self, checkpoint="./checkpoints/sam2.1_hiera_large.pt", 
                 model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml"):
        """Initialize the video segmentation model
        
        Args:
            checkpoint (str): Path to model checkpoint
            model_cfg (str): Path to model config
        """
        # Initialize SAM2 model
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint)
        print("SAM2 model initialized")

    def process_video(self, video_path, bbox, output_dir=None):
        """Process a video to generate segmentation masks
        
        Args:
            video_path (str): Path to input video file
            bbox (list): Initial bounding box in format [x1, y1, x2, y2]
            output_dir (str, optional): Output directory. If None, uses video's directory
            
        Returns:
            tuple: (target_frame_dir, mask_dir)
                - target_frame_dir (str): Directory containing visualization frames
                - mask_dir (str): Directory containing binary masks as .npy files
        """
        # Setup directories
        video_name = os.path.basename(video_path).split('.')[0]
        if output_dir is None:
            output_dir = os.path.dirname(video_path)
        
        output_folder = os.path.join(output_dir, video_name)
        ori_frame_dir = os.path.join(output_folder, "ori_frames")
        target_frame_dir = os.path.join(output_folder, "target_frames")
        mask_dir = os.path.join(output_folder, "masks")  # New directory for masks
        
        # Create directories
        os.makedirs(ori_frame_dir, exist_ok=True)
        os.makedirs(target_frame_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)  # Create masks directory

        # Extract frames
        num_frames = self._extract_frames(video_path, ori_frame_dir)
        print(f"Extracted {num_frames} frames")

        # Process frames
        self._process_frames(ori_frame_dir, target_frame_dir, mask_dir, bbox)
        
        print("Completed processing all frames")
        
        # Create output video from processed frames
        video_path = create_video_from_frames(target_frame_dir, output_folder, fps=15)
        print(f"Created visualization video: {video_path}")
        
        return target_frame_dir, mask_dir

    def _extract_frames(self, video_path, frame_dir):
        """Extract individual frames from video file
        
        Args:
            video_path (str): Path to input video
            frame_dir (str): Directory to save extracted frames
            
        Returns:
            int: Number of frames extracted
            
        Raises:
            ValueError: If video file cannot be opened
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_path = os.path.join(frame_dir, f"{frame_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"Extracted {frame_count} frames...")
        
        cap.release()
        return frame_count

    @staticmethod
    def _apply_mask(image, mask, alpha=0.5):
        """Apply colored mask overlay on image
        
        Args:
            image (numpy.ndarray): Input image in RGB format
            mask (torch.Tensor): Binary mask tensor on GPU
            alpha (float, optional): Transparency of overlay. Defaults to 0.5
            
        Returns:
            numpy.ndarray: Image with colored mask overlay
        """
        image_tensor = torch.from_numpy(image).cuda().float()
        
        color_mask = torch.zeros_like(image_tensor)
        green_color = torch.tensor([0, 255, 0], dtype=torch.float32, device='cuda')
        mask_bool = mask.squeeze() > 0.5
        
        color_mask[mask_bool] = green_color
        output = (image_tensor + alpha * color_mask).clamp(0, 255)
        
        return output.cpu().numpy().astype(np.uint8)

    def _save_mask(self, mask_dir, frame_idx, mask):
        """Save binary mask as numpy array
        
        Args:
            mask_dir (str): Directory to save masks
            frame_idx (int): Frame index for filename
            mask (torch.Tensor): Binary mask tensor to save
        """
        mask_path = os.path.join(mask_dir, f"{frame_idx:05d}.npy")
        # Convert mask to CPU and numpy, then save
        mask_np = mask.cpu().numpy()
        np.save(mask_path, mask_np)

    def _process_frames(self, ori_frame_dir, target_frame_dir, mask_dir, bbox):
        """Process video frames with SAM2 segmentation
        
        Args:
            ori_frame_dir (str): Directory containing original frames
            target_frame_dir (str): Directory to save visualization frames
            mask_dir (str): Directory to save binary masks
            bbox (list): Initial bounding box [x1, y1, x2, y2]
        """
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Initialize state
            state = self.predictor.init_state(video_path=str(ori_frame_dir))
            
            # Process first frame with bbox
            frame_idx, object_ids, mask = self.predictor.add_new_points_or_box(
                inference_state=state,
                box=bbox,
                frame_idx=0,
                obj_id=0   
            )
            
            # Save first frame and mask
            self._save_masked_frame(ori_frame_dir, target_frame_dir, frame_idx, mask[0])
            self._save_mask(mask_dir, frame_idx, mask[0])

            # Process remaining frames
            for frame_idx, object_ids, masks in self.predictor.propagate_in_video(state):
                print(f"Processing frame {frame_idx}, tracking {len(object_ids)} objects")
                if len(masks) > 0:
                    self._save_masked_frame(ori_frame_dir, target_frame_dir, frame_idx, masks[0])
                    self._save_mask(mask_dir, frame_idx, masks[0])

        print("Completed processing all frames")

    def _save_masked_frame(self, ori_frame_dir, target_frame_dir, frame_idx, mask):
        """Save frame with mask visualization
        
        Args:
            ori_frame_dir (str): Directory containing original frames
            target_frame_dir (str): Directory to save visualization frames
            frame_idx (int): Frame index for filename
            mask (torch.Tensor): Binary mask tensor for visualization
        """
        # Read and convert frame
        frame_path = os.path.join(ori_frame_dir, f"{frame_idx:05d}.jpg")
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply and save mask
        masked_frame = self._apply_mask(frame, mask)
        masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
        
        output_path = os.path.join(target_frame_dir, f"{frame_idx:05d}.jpg")
        cv2.imwrite(output_path, masked_frame)
        print(f"Saved masked frame to {output_path}")


def main():
    """Example usage of SAM2VideoSegmentor
    
    Demonstrates how to:
    1. Initialize the segmentor
    2. Process a video with bounding box
    3. Get paths to results
    """
    # Example usage
    video_path = "./demo/data/gallery/dance_15fps.mp4"
    bbox = [192, 251, 267, 370]  # Example bbox [x1, y1, x2, y2]
    
    # Initialize and run
    segmentor = SAM2VideoSegmentor()
    vis_dir, mask_dir = segmentor.process_video(video_path, bbox)
    
    print(f"Results saved to:\n  Visualizations: {vis_dir}\n  Masks: {mask_dir}")

if __name__ == "__main__":
    main()