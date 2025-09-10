#!/bin/bash

# Script to run the 3D Pose Estimation test (Global coordinate system)
# This script automatically copies source files and saves parameters for reproducibility

CHECKPOINT_PATH="./experiments/oms_20250724_1622/best_model.pth"
# CHECKPOINT_PATH="./experiments/oms_20250725_1130/best_model.pth"
# CHECKPOINT_PATH="./experiments/oms_20250726_1031/best_model.pth"
# CHECKPOINT_PATH="./experiments/oms_20250725_1232/best_model.pth"

# Default paths (modify these according to your setup)
DEFAULT_TEST_JSON="/home/ti_wang/Ti_workspace/PrimatePose/Pose3D/data/oms_test_small.json"
DEFAULT_IMAGE_ROOT="/home/ti_wang/data/tiwang/v8_coco/images"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

# Check if test JSON exists
if [ ! -f "$DEFAULT_TEST_JSON" ]; then
    echo "Error: Test JSON file not found: $DEFAULT_TEST_JSON"
    exit 1
fi

echo "Running 3D Pose Estimation Test..."
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Test JSON: $DEFAULT_TEST_JSON"
echo ""

# Run the test script
python test.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --test_json "$DEFAULT_TEST_JSON" \
    --image_root "$DEFAULT_IMAGE_ROOT" \
    --backbone resnet50 \
    --num_keypoints 37 \
    --image_size 256 256 \
    --batch_size 16 \
    --num_workers 4 \
    --num_vis_samples 30

echo ""
echo "Test completed! Check the generated test_results directory for outputs."
echo ""
echo "📁 Test Results Structure:"
echo "  📊 Metrics: test_results.json"
echo "  🖼️  Visualizations: visualizations/"
echo "  📋 Source Code: source_code/"
echo "  📝 Test Arguments: test_args.txt"
echo "  🏆 Checkpoint Info: checkpoint_info.txt"