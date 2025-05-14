import json

class SuperAnimalProcessor:
    
    def __init__(self, data_path):
        pass

    @staticmethod
    def print_all_dataset_names(data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        images = data['images']
        unique_datasets = set()
        for image in images:
            dataset_name = image['source_dataset']
            unique_datasets.add(dataset_name)
        
        for dataset_name in sorted(unique_datasets):
            print(dataset_name)


if __name__ == "__main__":
    
    data_path = "/home/ti_wang/Ti_workspace/PrimatePose/SuperAnimal/data/SuperAnimal/Quadruped80K/annotations/test.json"
    SuperAnimalProcessor.print_all_dataset_names(data_path)