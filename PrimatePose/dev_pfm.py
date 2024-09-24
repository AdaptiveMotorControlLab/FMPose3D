import json
from pathlib import Path

def main(project_root: Path):
    for subset, ann_filename in [
        ("train", "pfm_train_apr15.json"),
        ("valid", "pfm_val_apr15.json"),
        ("test", "pfm_test_apr15.json"),
    ]:
        print(f"Subset {subset}")
        with open(project_root / "annotations" / ann_filename, "r") as f:
            data = json.load(f)
        
        print(f"Number of annotated images: {len(data['images'])}")
        print(f"Number of annotatations: {len(data['annotations'])}")
        # images = [
        #     p
        #     for p in (project_root / subset).iterdir()
        #     if p.suffix in (".png", ".jpeg", ".jpg")
        # ]
        # print("num images in folder", len(images))
        
        print(f"Number of categories: {len(data['categories'])}")
        # print(f"categories: {data['categories']}")
        
        print(f"Number of datasets: {len(data['datasets'])}")
        # print(f"datasets: {data['datasets']}")

        print(f"Number of pfm_skeleton: {len(data['pfm_skeleton'])}")
        # print(f"pfm_skeleton: {data['pfm_skeleton']}")
        
        print(f"Number of pfm_keypoints: {len(data['pfm_keypoints'])}")
        # print(f"pfm_keypoints: {data['pfm_keypoints']}")
        
        print()

    # train.json -> annotations for the training set
    # valid.json -> annotations for the validation set
    # test.json -> annotations for the test set
    
    # Subset train
    # Numbers annotated images: 416265;
    # Numbers annotatations: 804465;
    # numbers images in folder 396056;

    # Subset valid
    # Num annotated images: 74991
    # Num annotatations: 116745
    # num images in folder 70903

    # Subset test
    # Num annotated images: 98029
    # Num annotatations: 144023
    # num images in folder 94376

def test_image(path):
    with open(path, "r") as f:
        train_data = json.load(f)
    # sample = train_data[0]
    print(train_data.keys())
    # sample_image = train_data["images"][3230]
    # select the specific image by id
    sample_image = next((image for image in train_data["images"] if image["id"] == 160490), None)
    
    print("sample_image", sample_image.keys())
    sample_annotation = train_data["annotations"][1]    
    print("sample_image", sample_image)
    # print(sample_annotation)
    img_name = sample_image["file_name"]
    img_path = Path("/mediaPFM/data/datasets/final_datasets/v7/test") / img_name
    img_keypoint = sample_annotation["keypoints"]
    num_keypoints = 16
    print("len(img_keypoint):", len(img_keypoint))
    # print(img_path)
    # # print(sample_image) 
    area = sample_annotation["area"]
    # print("area:", area)
    keypoints_original = sample_annotation["keypoints_orig"]
    # print("keypoints_orig:", len(keypoints_original))
    bbox = sample_annotation["bbox"]
    # print("bbox:", bbox)  
    
import json
from pathlib import Path

def inspect_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    
    # Print the top-level keys
    print("Top-level keys:", data.keys())
    
    # Print a sample of the images and annotations
    if "images" in data and len(data["images"]) > 0:
        print("\nSample image entry:")
        print(json.dumps(data["images"][0], indent=2))
    
    if "annotations" in data and len(data["annotations"]) > 0:
        print("\nSample annotation entry:")
        print(json.dumps(data["annotations"][0], indent=2))
        
    if "categories" in data and len(data["categories"]) > 0:
        print("\nSample categories entry:")
        print(json.dumps(data["categories"][0], indent=2))


    if "datasets" in data and len(data["datasets"]) > 0:
        print("\nSample datasets entry:")
        print(json.dumps(data["datasets"][0], indent=2))

    if "pfm_skeleton" in data and len(data["pfm_skeleton"]) > 0:
        print("\nSample pfm_skeleton entry:")
        print(json.dumps(data["pfm_skeleton"][0], indent=2))

    if "pfm_keypoints" in data and len(data["pfm_keypoints"]) > 0:
        print("\nSample pfm_keypoints entry:")
        print(json.dumps(data["pfm_keypoints"][0], indent=2))

def select_sample_by_id(path, id):
    with open(path, "r") as f:
        data = json.load(f)
         
    image = next((image for image in data["images"] if image["id"] == id), None)
    print("image", image)
    # 查找所有 id=xx 的数据
    results = [item for item in data["annotations"] if item.get("id") == id]
    print("results", len(results))

def extract_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    
    len = 8
    images = data["images"][:len] 
    annotations = data["annotations"][:len]
    categories = data["categories"][:len]
    # datasets = data["datasets"][:]
    # pfm_keypoints = data["pfm_keypoints"][:]
    
    new_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
        # "datasets": datasets,
        # "pfm_keypoints": pfm_keypoints
    }
    if new_data:
        output_file_path = "/home/ti/projects/PrimatePose/ti_data/data/pfm_test_{}_items.json".format(str(len))
        
        with open(output_file_path, "w") as f:
            json.dump(new_data, f, indent=4)
        print("Extracted top x items to", output_file_path)
    else:
        print("No data to extract")
          
def print_json(path):
           
    with open(path, "r") as f:
        data = json.load(f)
        print(data.keys())
        print(data)

def search_file_by_id(path, id):
    with open(path, "r") as f:
        data = json.load(f)
        
    image = next((image for image in data["images"] if image["id"] == id), None)

    print("image", image["file_name"])

    print("image", image)

if __name__ == "__main__":
    # main(project_root=Path("/mediaPFM/data/datasets/final_datasets/v7"))
    # test_image()
    
    # Path to the JSON file
    # json_file_path = "/mediaPFM/data/datasets/final_datasets/v7/annotations/pfm_test_apr15.json"
    # json_file_path = "/home/ti/projects/PrimatePose/ti_data/primate_test_1.1.json"
    # json_file_path_sample = "/home/ti/projects/PrimatePose/ti_data/data/pfm_test_10_items.json"
    # inspect_json(json_file_path)
    # test_image(json_file_path)
    # extract_json(json_file_path)
    test_json_path = "/mnt/tiwang/v7/annotations/pfm_test_apr15.json"
    search_file_by_id(test_json_path, 82506)
