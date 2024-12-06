# Define variables for paths
VIDEO_PATH="/home/ti_wang/Ti_workspace/projects/samurai/results/a_monkey.mp4"
BBOX_PATH="/home/ti_wang/Ti_workspace/projects/samurai/bbox.txt"
OUTPUT_PATH="/home/ti_wang/Ti_workspace/projects/samurai/results/processed"

# Run the script
python process_video.py --video "$VIDEO_PATH" --bbox "$BBOX_PATH" --output "$OUTPUT_PATH"