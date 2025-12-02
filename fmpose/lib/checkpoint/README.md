## Demo Checkpoints

### Automatic Download (Recommended)
The checkpoint files will be **automatically downloaded** from Google Drive when you run the demo script for the first time. No manual download is required!

Checkpoints are saved to the user cache directory:
- **Linux**: `~/.cache/fmpose/checkpoints/`
- **macOS**: `~/Library/Caches/fmpose/checkpoints/`
- **Windows**: `%LOCALAPPDATA%\fmpose\checkpoints\`

### Manual Download
If automatic download fails, you can manually download YOLOv3 and HRNet pretrained models from [here](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA?usp=sharing) and put them in the cache directory above.

Required files:
- `yolov3.weights` - YOLOv3 weights for human detection
- `pose_hrnet_w48_384x288.pth` - HRNet model for 2D pose estimation
