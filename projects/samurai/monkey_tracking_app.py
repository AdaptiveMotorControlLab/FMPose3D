import os
import cv2
import gradio as gr
import numpy as np
from pathlib import Path
import subprocess
from inference_combined import process_video_with_tracking


def convert_to_h264(input_video, output_video):
    """Convert video to H.264 codec and MP4 format."""
    ffmpeg_cmd = [
        'ffmpeg', '-i', input_video,
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y',  # Overwrite output file if it exists
        output_video
    ]
    subprocess.run(ffmpeg_cmd, check=True, stderr=subprocess.PIPE, universal_newlines=True)
    
class MonkeyTrackingGUI:
    def __init__(self, video_root, output_root):
        self.video_root = video_root
        self.output_root = output_root
        self.selected_points = []
        self.current_frame = None
        self.current_video_path = None
        self.current_video_name = None
        self.current_frame_idx = 0
        self.video_info = None
        
    def get_video_list(self):
        """Get list of videos from video root directory"""
        if not os.path.exists(self.video_root):
            return []
        return sorted([f for f in os.listdir(self.video_root) 
                      if f.endswith(('.mp4', '.avi', '.mov'))])
    
    def load_first_frame(self, video_name):
        """Load first frame from selected video"""
        if not video_name:
            return None, "Please select a video first"
        
        self.current_video_name = video_name
        self.current_video_path = os.path.join(self.video_root, video_name)
        cap = cv2.VideoCapture(self.current_video_path)
        
        # Store video info
        self.video_info = {
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None, "Failed to read video"
        
        self.current_frame = frame
        self.selected_points = []  # Clear previous points
        return frame, f"Loaded first frame from {video_name}. Click two points to define bbox."
    
    def add_point(self, img, evt: gr.SelectData):
        """Add point and draw bbox if two points are selected"""
        if self.current_frame is None:
            return img, "Please load a video first"
            
        x, y = evt.index[0], evt.index[1]
        self.selected_points.append((x, y))
        
        # Draw points and bbox
        frame_vis = self.current_frame.copy()
        
        # Draw all points
        for pt in self.selected_points:
            cv2.circle(frame_vis, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
        
        # If we have two points, draw bbox
        if len(self.selected_points) == 2:
            x1 = min(self.selected_points[0][0], self.selected_points[1][0])
            y1 = min(self.selected_points[0][1], self.selected_points[1][1])
            x2 = max(self.selected_points[0][0], self.selected_points[1][0])
            y2 = max(self.selected_points[0][1], self.selected_points[1][1])
            
            # Draw rectangle
            cv2.rectangle(frame_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            message = f"Bbox coordinates: x={x1}, y={y1}, w={x2-x1}, h={y2-y1}"
        else:
            message = f"Selected point {len(self.selected_points)}/2: ({x}, {y})"
            
        return frame_vis, message
    
    def clear_points(self):
        """Clear selected points"""
        self.selected_points = []
        if self.current_frame is not None:
            return self.current_frame.copy(), "Cleared points. Select two new points."
        return None, "Please load a video first"
    
    def save_bbox(self):
        """Save bbox coordinates to file"""
        if len(self.selected_points) != 2:
            return "Please select two points first"
            
        x1 = min(self.selected_points[0][0], self.selected_points[1][0])
        y1 = min(self.selected_points[0][1], self.selected_points[1][1])
        x2 = max(self.selected_points[0][0], self.selected_points[1][0])
        y2 = max(self.selected_points[0][1], self.selected_points[1][1])
        
        # Convert to x,y,w,h format
        bbox = [x1, y1, x2-x1, y2-y1]
        
        # Save to file
        save_path = Path("bbox.txt")
        with open(save_path, 'w') as f:
            f.write(','.join(map(str, bbox)))
            
        return f"Saved bbox to {save_path}"
    
    def process_video(self):
        """Process the video using inference_combined.py"""
        if not self.current_video_path or not os.path.exists("bbox.txt"):
            return None, "Please select a video and define bbox first"
        
        try:
            # Create output directory
            video_name = os.path.splitext(self.current_video_name)[0]
            output_dir = os.path.join(self.output_root, video_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Define paths for pose estimation models
            pose_config = "/home/ti_wang/Ti_workspace/projects/samurai/pre_trained_models/pytorch_config.yaml"
            pose_snapshot = "/home/ti_wang/Ti_workspace/projects/samurai/pre_trained_models/snapshot-best-056.pt"
            
            # Process video
            result_path = process_video_with_tracking(
                video_path=self.current_video_path,
                bbox_path="bbox.txt",
                output_path=output_dir,
                pose_config=pose_config,
                pose_snapshot=pose_snapshot,
                device="cuda"
            )
            
            if os.path.exists(result_path):
                if os.path.isdir(result_path):
                    return None, "Video creation failed, but frames were saved"
                
                # Create a web-compatible version
                web_output = os.path.join(output_dir, "web_output.mp4")
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-i', result_path,
                    '-c:v', 'mpeg4',
                    '-q:v', '1',
                    '-pix_fmt', 'yuv420p',
                    web_output
                ]
                
                try:
                    subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
                    return web_output, f"Video processed successfully! Saved to {web_output}"
                except subprocess.CalledProcessError:
                    return result_path, f"Video processed successfully! Saved to {result_path}"
            
            return None, "Error: No video file was produced"
                
        except Exception as e:
            return None, f"Error processing video: {str(e)}"

def make_demo(video_root, output_root):
    app = MonkeyTrackingGUI(video_root, output_root)
    
    with gr.Blocks(css="#video-controls { margin-top: 10px; }") as demo:
        gr.Markdown("# Monkey Video Processing System")
        
        with gr.Row():
            # Left column for video selection and bbox
            with gr.Column():
                video_list = gr.Dropdown(
                    choices=app.get_video_list(),
                    label="Select Video",
                    interactive=True
                )
                load_button = gr.Button("Load First Frame")
                message = gr.Textbox(label="Status", interactive=False)
                frame_display = gr.Image(label="First Frame")
                
                with gr.Row():
                    clear_button = gr.Button("Clear Points")
                    save_button = gr.Button("Save Bbox")
                
                process_button = gr.Button("Process Video", variant="primary")
            
            # Right column for processed video
            with gr.Column():
                output_message = gr.Textbox(label="Processing Status")
                processed_video = gr.Video(
                    label="Processed Video",
                    format="mp4",
                    width=800,
                    height=450,
                    autoplay=True
                )
                
                # Simplified video controls
                with gr.Row(elem_id="video-controls"):
                    video_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        label="Video Progress",
                        interactive=True
                    )
        
        # Event handlers
        load_button.click(
            app.load_first_frame,
            inputs=[video_list],
            outputs=[frame_display, message]
        )
        
        frame_display.select(
            app.add_point,
            inputs=[frame_display],
            outputs=[frame_display, message]
        )
        
        clear_button.click(
            app.clear_points,
            outputs=[frame_display, message]
        )
        
        save_button.click(
            app.save_bbox,
            outputs=[message]
        )
        
        process_button.click(
            app.process_video,
            outputs=[processed_video, output_message]
        )
    
    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str, required=True,
                       help="Root directory containing monkey videos")
    parser.add_argument("--output_root", type=str, required=True,
                       help="Root directory for saving processed videos")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port to run the gradio app")
    parser.add_argument("--share", action="store_true",
                       help="Create a public link")
    
    args = parser.parse_args()
    
    demo = make_demo(args.video_root, args.output_root)
    
    # Try different ports if the default one is occupied
    for port in range(args.port, args.port + 100):
        try:
            demo.launch(server_port=port, share=args.share)
            break
        except OSError:
            print(f"Port {port} is in use, trying next port...")
            continue 