import torch
import numpy as np
import cv2
import os
from sam2.build_sam import build_sam2_video_predictor

class SAM2VideoProcessor:
    def __init__(self, video_path, bbox_path, checkpoint, model_cfg):
        """Initialize video processor with paths and model settings"""
        self.video_path = video_path
        self.bbox_path = bbox_path
        
        # Setup directories
        self.video_name = os.path.basename(video_path).split('.')[0]
        self.output_folder = os.path.join(os.path.dirname(video_path), self.video_name)
        self.ori_frame_dir = os.path.join(self.output_folder, "ori_frames")
        self.target_frame_dir = os.path.join(self.output_folder, "target_frames")
        
        # Initialize model
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint)
        print("SAM2 model initialized")
        
        # Create output directories
        os.makedirs(self.ori_frame_dir, exist_ok=True)
        os.makedirs(self.target_frame_dir, exist_ok=True)

    def read_bbox(self):
        """Read initial bbox from file"""
        with open(self.bbox_path, 'r') as f:
            x, y, w, h = map(float, f.read().strip().split(','))
            return [int(x), int(y), int(w), int(h)]

    def extract_frames(self):
        """Extract frames from video"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {self.video_path}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_path = os.path.join(self.ori_frame_dir, f"{frame_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"Saved {frame_count} frames...")
        
        cap.release()
        print(f"Successfully saved {frame_count} frames to {self.ori_frame_dir}")
        return frame_count

    @staticmethod
    def apply_mask_to_image(image, mask, alpha=0.5):
        """Apply colored mask overlay on image"""
        image_tensor = torch.from_numpy(image).cuda().float()
        
        color_mask = torch.zeros_like(image_tensor)
        green_color = torch.tensor([0, 255, 0], dtype=torch.float32, device='cuda')
        mask_bool = mask.squeeze() > 0.5
        
        color_mask[mask_bool] = green_color
        output = (image_tensor + alpha * color_mask).clamp(0, 255)
        
        return output.cpu().numpy().astype(np.uint8)

    def process_frame(self, frame_idx, mask):
        """Process a single frame with mask"""
        frame_path = os.path.join(self.ori_frame_dir, f"{frame_idx:05d}.jpg")
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        masked_frame = self.apply_mask_to_image(frame, mask)
        masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
        
        output_path = os.path.join(self.target_frame_dir, f"{frame_idx:05d}.jpg")
        cv2.imwrite(output_path, masked_frame)
        print(f"Saved masked frame to {output_path}")

    def process_video(self):
        """Main process to handle video segmentation"""
        # Extract frames
        num_frames = self.extract_frames()
        print(f"Total frames extracted: {num_frames}")

        # Get initial bbox
        prompt_bbox = self.read_bbox()
        frame_idx = 0

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Initialize state
            state = self.predictor.init_state(video_path=str(self.ori_frame_dir))
            
            # Process first frame with bbox
            frame_idx, object_ids, mask = self.predictor.add_new_points_or_box(
                inference_state=state,
                box=prompt_bbox,
                frame_idx=frame_idx,
                obj_id=0   
            )
            self.process_frame(frame_idx, mask[0])

            # Process remaining frames
            for frame_idx, object_ids, masks in self.predictor.propagate_in_video(state):
                print(f"Processing frame {frame_idx}, tracking {len(object_ids)} objects")
                if len(masks) > 0:
                    self.process_frame(frame_idx, masks[0])

        print("Completed processing all frames")

def main():
    # Configuration
    VIDEO_PATH = "./demo/data/gallery/00185_Inner_Take7_Capture_0004-skirt-15fps.mp4"
    # VIDEO_PATH = "./demo/data/gallery/01_dog.mp4"
    BBOX_PATH = "./bbox.txt"  # x1,y1,x2,y2
    CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
    MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # Process video
    processor = SAM2VideoProcessor(VIDEO_PATH, BBOX_PATH, CHECKPOINT, MODEL_CFG)
    processor.process_video()

if __name__ == "__main__":
    main()
        