from pathlib import Path
from deeplabcut.utils.make_labeled_video import create_labeled_video

# Define video path
video_path = Path("/home/ti_wang/Ti_workspace/PrimatePose/data/demo_8s/8s_3840_2160_25fps.mp4")
import deeplabcut

# Create a new project
project_name = 'PrimatePose'
your_name = 'researcher'
video_path = ["/home/ti_wang/Ti_workspace/PrimatePose/data/demo_8s/8s_3840_2160_25fps.mp4"]

config_path = deeplabcut.create_new_project(
    project_name,
    your_name,
    video_path,
    working_directory='/home/ti_wang/Ti_workspace/PrimatePose/data/demo_8s',
    copy_videos=True
)

print(config_path)

# Define the skeleton structure with bodypart names
PFM_SKELETON = [
    ["right_eye", "nose"], ["left_eye", "nose"], ["left_ear", "right_eye"], ["right_ear", "left_eye"],
    ["nose", "neck"], ["left_shoulder", "neck"], ["right_shoulder", "neck"], ["head", "body_center"],
    ["left_elbow", "left_shoulder"], ["right_elbow", "right_shoulder"], ["left_wrist", "left_elbow"], ["right_wrist", "right_elbow"],
    ["left_hand", "left_wrist"], ["right_hand", "right_wrist"], ["left_hip", "neck"], ["right_hip", "neck"],
    ["left_hip", "center_hip"], ["right_hip", "center_hip"], ["left_hip", "left_knee"], ["right_hip", "right_knee"],
    ["center_hip", "left_knee"], ["center_hip", "right_knee"], ["left_knee", "left_ankle"], ["right_knee", "right_ankle"],
    ["left_ankle", "left_foot"], ["right_ankle", "right_foot"], ["center_hip", "root_tail"], ["root_tail", "mid_tail"],
    ["mid_tail", "mid_end_tail"], ["mid_end_tail", "end_tail"]
]

# Create labeled video
create_labeled_video(
    config=config_path,  # Add your actual config file path here
    videos=[str(video_path)],
    videotype="mp4",
    draw_skeleton=True,
    skeleton=PFM_SKELETON,
    skeleton_color="white",
    pcutoff=0.6,
    dotsize=8,
    colormap="rainbow",
    codec="mp4v",
    confidence_to_alpha=True,
)