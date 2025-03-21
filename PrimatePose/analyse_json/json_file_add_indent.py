import json

def change_json_indent(source_json_path, output_path):
    """Read JSON data from source file and save it to a new file with indentation."""
    with open(source_json_path, 'r') as file:
        data = json.load(file)
    
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"JSON data saved to {output_path}")
    
if __name__ == "__main__":
    json_path = "/home/ti_wang/Ti_workspace/PrimatePose/Vis/samples/video_adaption_SA_20250317_1956/videoID_1000013_original_superanimal_topviewmouse_pfm_det_pfm_pose_before_adapt.json"
    output_path = "/home/ti_wang/Ti_workspace/PrimatePose/Vis/samples/video_adaption_SA_20250317_1956/videoID_1000013_original_superanimal_topviewmouse_pfm_det_pfm_pose_before_adapt.json"
    change_json_indent(json_path, output_path)