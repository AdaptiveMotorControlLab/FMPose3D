import os
import cv2
import gradio as gr
import numpy as np
from pathlib import Path

class BboxGUI:
    def __init__(self, video_root):
        self.video_root = video_root
        self.selected_points = []
        self.current_frame = None
        
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
        
        video_path = os.path.join(self.video_root, video_name)
        cap = cv2.VideoCapture(video_path)
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
    
    def clear_points(self):
        """Clear selected points"""
        self.selected_points = []
        if self.current_frame is not None:
            return self.current_frame.copy(), "Cleared points. Select two new points."
        return None, "Please load a video first"

def make_demo(video_root):
    bbox_gui = BboxGUI(video_root)
    
    with gr.Blocks() as demo:
        gr.Markdown("# Monkey Video Bbox Selection")
        
        with gr.Row():
            # Left column for video selection
            with gr.Column():
                video_list = gr.Dropdown(
                    choices=bbox_gui.get_video_list(),
                    label="Select Video",
                    interactive=True
                )
                load_button = gr.Button("Load First Frame")
                
            # Right column for bbox selection
            with gr.Column():
                message = gr.Textbox(label="Status", interactive=False)
                frame_display = gr.Image(label="First Frame")
                clear_button = gr.Button("Clear Points")
                save_button = gr.Button("Save Bbox")
        
        # Event handlers
        load_button.click(
            bbox_gui.load_first_frame,
            inputs=[video_list],
            outputs=[frame_display, message]
        )
        
        frame_display.select(
            bbox_gui.add_point,
            inputs=[frame_display],
            outputs=[frame_display, message]
        )
        
        clear_button.click(
            bbox_gui.clear_points,
            outputs=[frame_display, message]
        )
        
        save_button.click(
            bbox_gui.save_bbox,
            outputs=[message]
        )
        
    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", type=str, required=True,
                       help="Root directory containing monkey videos")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port to run the gradio app")
    parser.add_argument("--share", action="store_true",
                       help="Create a public link")
    
    args = parser.parse_args()
    
    demo = make_demo(args.video_root)
    demo.launch(server_port=args.port, share=args.share) 