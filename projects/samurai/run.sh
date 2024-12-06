# Basic usage with default paths
# python split_video.py --frames 100

# With custom paths
# python results/split_video.py --input /home/ti_wang/Ti_workspace/projects/samurai/results/monkey_data/a_monkey.mp4 --output /home/ti_wang/Ti_workspace/projects/samurai/results/monkey_data/a_monkey_10frames.mp4 --frames 10

python monkey_tracking_app.py --video_root /home/ti_wang/Ti_workspace/projects/samurai/results/monkey_data --output_root /home/ti_wang/Ti_workspace/projects/samurai/results --port 7860 --share